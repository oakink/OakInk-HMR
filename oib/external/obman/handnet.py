import functools
import imp
import os
import os.path as osp
from collections import OrderedDict
from turtle import forward
from typing import Any, Dict, List
from copy import deepcopy
from jinja2 import pass_context

import torch
import torch.nn as nn
import torch.nn.functional as torch_f
import torch.optim

from oib.criterions.criterion import TensorLoss
from oib.metrics.basic_metric import AverageMeter, Metric, VisMetric
from oib.metrics.evaluator import Evaluator
from oib.models.model_abstraction import ModuleAbstract
from oib.utils.builder import MODEL
from oib.utils.logger import logger
from oib.utils.misc import CONST, enable_lower_param, param_size
from oib.utils.transform import denormalize
from oib.viztools.draw import concat_imgs, draw_batch_joint_images

from .branches import resnet
from .branches.manobranch import ManoBranch, ManoLoss
from .branches.atlasbranch import AtlasBranch, AtlasLoss
from .branches.contactloss import (
    compute_contact_loss,
    batch_pairwise_dist,
    meshiou,
)
from .branches.absolutebranch import AbsoluteBranch
from .branches.queries import TransQueries, BaseQueries


@MODEL.register_module
class ObMan(ModuleAbstract):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.name = type(self).__name__

        self.handnet = HandNet(cfg)

        self.metric = "val"
        logger.info("{}".format(cfg.PRETRAINED))
        self.init_weights(pretrained=cfg.PRETRAINED)
        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def forward(self, inputs, mode="train", **kwargs):
        if mode == "train":
            return self.training_step(inputs, **kwargs)
        elif mode == "val":
            return self.validation_step(inputs, **kwargs)
        elif mode == "test":
            return self.testing_step(inputs, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")
    
    def training_step(self, batch):
        preds = self.handnet(batch)
        final_loss, final_loss_dict = self.compute_loss(preds, batch)

        with torch.no_grad():
            self.evaluator.feed_all(preds, batch, final_loss_dict)

        return preds, final_loss_dict

    def validation_step(self, batch):
        preds = self.handnet(batch)
        final_loss_dict = {}  # * we don't need to compute loss for validation
        with torch.no_grad():
            self.evaluator.feed_all(preds, batch, final_loss_dict)
        return preds, final_loss_dict

    def testing_step(self, batch):
        preds = self._forward_impl(batch)
        final_loss_dict = {}  # * we don't need to compute loss for validation
        with torch.no_grad():
            self.evaluator.feed_all(preds, batch, final_loss_dict)
        return preds, final_loss_dict



    class Hand_Loss(TensorLoss):
        def __init__(self, cfg) -> None:
            super().__init__()
            if (
                cfg.LAMBDA_VERTS
                or cfg.LAMBDA_JOINTS3D
                or cfg.AMBDA_JOINTS2D
                or cfg.LAMBDA_PCA
            ):
                self.mano_lambdas = True
            else:
                self.mano_lambdas = False
            self.mano_loss = ManoLoss(
                lambda_verts=cfg.LAMBDA_VERTS,
                lambda_joints3d=cfg.LAMBDA_JOINTS3D,
                lambda_shape=cfg.LAMBDA_SHAPE,
                lambda_pose_reg=cfg.LAMBDA_POSE_REG,
                lambda_pca=cfg.LAMBDA_PCA,
            )
            self.lambda_joints2d = cfg.LAMBDA_JOINTS2D
        
        def forward(self, preds, gt):
            mano_results = preds["mano_results"]
            mano_total_loss, mano_losses = self.mano_loss.compute_loss(mano_results, gt)
            loss_dict = {}
            for key in mano_losses:
                loss_dict[key] = mano_losses[key]
            
            if self.lambda_joints2d:
                proj_joints2d = preds["joints2d"]
                gt_joints2d = gt["target_joints_2d"].cuda().float()
                joints2d_loss = torch_f.mse_loss(
                    proj_joints2d, gt_joints2d
                )
                loss_dict["joints2d"] = joints2d_loss
                mano_total_loss += self.lambda_joints2d * joints2d_loss
            
            return mano_total_loss, loss_dict

    class Object_Loss(TensorLoss):
        def __init__(self, cfg) -> None:
            super().__init__()
            self.contact_target = cfg.CONTACT.TARGET
            self.contact_zones = cfg.CONTACT.ZONES
            self.contact_lambda = cfg.CONTACT.LAMBDA
            self.contact_thresh = cfg.CONTACT.THRESH
            self.contact_mode = cfg.CONTACT.MODE
            self.collision_lambda = cfg.COLLISION.LAMBDA
            self.collision_thresh = cfg.COLLISION.THRESH
            self.collision_mode = cfg.COLLISION.MODE
            if cfg.CONTACT.LAMBDA or cfg.COLLISION.LAMBDA:
                self.need_collisions = True
            else:
                self.need_collisions = False
        
        def forward(self, preds, gt):
            # ============= Initiate >>>>>>>>>>>>>
            loss_dict = {}
            total_loss = 0
            mano_results = preds["mano_results"]
            atlas_results = preds["atlas_results"]
            atlas_branch_test_faces = preds["obj_utils"]["atlas_branch.test_faces"]
            atlas_loss = preds["obj_utils"]["atlas_loss"]
            mano_branch_faces = preds["obj_utils"][ "mano_branch.faces"]
            # ============= compute loss >>>>>>>>>>>>>
            if self.need_collisions:
                (
                    attr_loss,
                    penetr_loss,
                    contact_infos,
                    contact_metrics,
                ) = compute_contact_loss(
                    mano_results["verts"],
                    mano_branch_faces,
                    atlas_results["objpoints3d"],
                    atlas_branch_test_faces,
                    contact_thresh=self.contact_thresh,
                    contact_mode=self.contact_mode,
                    collision_thresh=self.collision_thresh,
                    collision_mode=self.collision_mode,
                    contact_target=self.contact_target,
                    contact_zones=self.contact_zones,
                )
                if (
                    TransQueries.verts3d in gt
                    and TransQueries.objpoints3d in gt
                ):
                    h2o_dists = batch_pairwise_dist(
                        gt[TransQueries.verts3d],
                        gt[TransQueries.objpoints3d],
                    )
                    dist_h2o_gt, _ = torch.min(h2o_dists, 2)
                    contact_ious, contact_auc = meshiou(
                        dist_h2o_gt, contact_infos["min_dists"]
                    )
                    contact_infos["batch_ious"] = contact_ious
                    loss_dict["contact_auc"] = contact_auc
                contact_loss = (
                    self.contact_lambda * attr_loss
                    + self.collision_lambda * penetr_loss
                )
                total_loss += contact_loss
                loss_dict["penetration_loss"] = penetr_loss
                loss_dict["attraction_loss"] = attr_loss
                loss_dict["contact_loss"] = contact_loss
                for metric_name, metric_val in contact_metrics.items():
                    loss_dict[metric_name] = metric_val
            
            atlas_total_loss, atlas_losses = atlas_loss.compute_loss(
                atlas_results, gt
            )
            total_loss += atlas_total_loss

            for key, val in atlas_losses.items():
                loss_dict[key] = val
            
            return total_loss, loss_dict

    def init_weights(self, pretrained=""):
        if pretrained == "":
            logger.warning(f"=> Init {self.name} weights in backbone and head")
            """
            Add init for other modules
            ...
            """
        elif os.path.isfile(pretrained):
            # pretrained_state_dict = torch.load(pretrained)
            logger.info(f"=> Loading {self.name} pretrained model from: {pretrained}")
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained, map_location=torch.device("cpu"))
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict_old = checkpoint["state_dict"]
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith("module."):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]  # delete "module." (in nn.parallel)
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                logger.error(f"=> No state_dict found in checkpoint file {pretrained}")
                raise RuntimeError()
            self.load_state_dict(state_dict, strict=False)
            logger.info(f"=> Loading SUCCEEDED")
        else:
            logger.error(f"=> No {self.name} checkpoints file found in {pretrained}")
            raise FileNotFoundError()
    

class HandNet(nn.Module):
    def __init__(self,cfg,):
        """
        Args:
            atlas_mesh (bool): Whether to get points on the mesh instead or
                randomling generating a point cloud. This allows to use
                regularizations that rely on an underlying triangulation
            atlas_ico_division: Granularity of the approximately spherical mesh
                see https://en.wikipedia.org/wiki/Geodesic_polyhedron.
                if 1, 42 vertices, if 2, 162 vertices, if 3 (default), 642
                vertices, if 4, 2562 vertices
            mano_root (path): dir containing mano pickle files
            mano_neurons: number of neurons in each layer of base mano decoder
            mano_use_pca: predict pca parameters directly instead of rotation
                angles
            mano_comps (int): number of principal components to use if
                mano_use_pca
            mano_lambda_pca: weight to supervise hand pose in PCA space
            mano_lambda_pose_reg: weight to supervise hand pose in axis-angle
                space
            mano_lambda_verts: weight to supervise vertex distances
            mano_lambda_joints3d: weight to supervise distances
            adapt_atlas_decoder: add layer between encoder and decoder, usefull
                when finetuning from separately pretrained encoder and decoder
        """
        super(HandNet, self).__init__()
        if int(cfg.RESNET_VERSION) == 18:
            img_feature_size = 512
            base_net = resnet.resnet18(pretrained=True)
        elif int(cfg.RESNET_VERSION) == 50:
            img_feature_size = 2048
            base_net = resnet.resnet50(pretrained=True)
        else:
            raise NotImplementedError(
                "Resnet {} not supported".format(cfg.RESNET_VERSION)
            )
        self.adapt_atlas_decoder = cfg.ADAPT_ATLAS_DECODER
        self.atlas_separate_encoder = cfg.ATLAS.SEPARATE_ENCODER
        if self.adapt_atlas_decoder:
            self.atlas_adapter = torch.nn.Linear(
                img_feature_size, img_feature_size
            )
        mano_base_neurons = [img_feature_size] + cfg.MANO.NEURONS
        self.contact_target = cfg.CONTACT.TARGET
        self.contact_zones = cfg.CONTACT.ZONES
        self.contact_lambda = cfg.CONTACT.LAMBDA
        self.contact_thresh = cfg.CONTACT.THRESH
        self.contact_mode = cfg.CONTACT.MODE
        self.collision_lambda = cfg.COLLISION.LAMBDA
        self.collision_thresh = cfg.COLLISION.THRESH
        self.collision_mode = cfg.COLLISION.MODE
        if cfg.CONTACT.LAMBDA or cfg.COLLISION.LAMBDA:
            self.need_collisions = True
        else:
            self.need_collisions = False
        self.base_net = base_net
        if self.atlas_separate_encoder:
            self.atlas_base_net = deepcopy(base_net)

        self.absolute_lambda = cfg.ABSOLUTE_LAMBDA
        if cfg.MANO.JOINTS2D:
            self.scaletrans_branch = AbsoluteBranch(
                base_neurons=[img_feature_size, int(img_feature_size / 2)],
                out_dim=3,
            )
        self.lambda_joints2d = cfg.MANO.JOINTS2D

        self.mano_adapt_skeleton = cfg.MANO.ADAPT_SKELETON
        self.mano_branch = ManoBranch(
            ncomps=cfg.MANO.COMPS,
            base_neurons=mano_base_neurons,
            adapt_skeleton=cfg.MANO.ADAPT_SKELETON,
            dropout=cfg.FC_DROPOUT,
            use_trans=False,
            mano_root=cfg.MANO.ROOT,
            center_idx=cfg.MANO.CENTER_IDX,
            use_shape=cfg.MANO.USE_SHAPE,
            use_pca=cfg.MANO.USE_PCA,
        )

        # AtlasBranch
        self.atlas_mesh = cfg.ATLAS.MESH
        feature_size = img_feature_size
        self.atlas_branch = AtlasBranch(
            mode="sphere",
            use_residual=cfg.ATLAS.RESIDUAL,
            points_nb=cfg.ATLAS.POINTS_NB,
            predict_trans=cfg.ATLAS.PREDICT_TRANS,
            predict_scale=cfg.ATLAS.PREDICT_SCALE,
            inference_ico_divisions=cfg.ATLAS.ICO_DIVISIONS,
            bottleneck_size=feature_size,
            use_tanh=cfg.ATLAS.USE_TANH,
            out_factor=cfg.ATLAS.OUT_FACTOR,
            separate_encoder=self.atlas_separate_encoder,
        )
        # AtlasLoss Initiate
        # **NOTOCE**: AtlasLoss is based on Atlas_branch
        self.atlas_lambda = cfg.ATLAS.LAMBDA
        self.atlas_final_lambda = cfg.ATLAS.FINAL_LAMBDA
        self.atlas_trans_weight = cfg.ATLAS.TRANS_WEIGHT
        self.atlas_scale_weight = cfg.ATLAS.SCALE_WEIGHT
        self.atlas_loss = AtlasLoss(
            atlas_loss=cfg.ATLAS.LOSS,
            lambda_atlas=cfg.ATLAS.LAMBDA,
            final_lambda_atlas=cfg.ATLAS.FINAL_LAMBDA,
            trans_weight=cfg.ATLAS.TRANS_WEIGHT,
            scale_weight=cfg.ATLAS.SCALE_WEIGHT,
            edge_regul_lambda=cfg.ATLAS.LAMBDA_REGUL_EDGES,
            lambda_laplacian=cfg.ATLAS.LAMBDA_LAPLACIAN,
            laplacian_faces=self.atlas_branch.test_faces,
            laplacian_verts=self.atlas_branch.test_verts,
        )

    def decay_regul(self, gamma):
        if self.atlas_loss.edge_regul_lambda is not None:
            self.atlas_loss.edge_regul_lambda = (
                gamma * self.atlas_loss.edge_regul_lambda
            )
        if self.atlas_loss.lambda_laplacian is not None:
            self.atlas_loss.lambda_laplacian = (
                gamma * self.atlas_loss.lambda_laplacian
            )

    def forward(
        self, sample, no_loss=False, return_features=True, force_objects=False
    ):
        # ============= get hands & object features >>>>>>>>>>>>>
        if force_objects:
            if TransQueries.objpoints3d not in sample:
                sample[TransQueries.objpoints3d] = None
        total_loss = None
        results = {}
        losses = {}
        # image = sample[TransQueries.images].cuda()
        image = sample["image"].cuda()
        features, _ = self.base_net(image)
        if self.atlas_separate_encoder:
            atlas_infeatures, _ = self.atlas_base_net(image)
            if return_features:
                results["atlas_features"] = atlas_infeatures
        if return_features:
            results["img_features"] = features

        # ============= predict center3d >>>>>>>>>>>>>
        # **NOTICE**: unused part
        # if (
        #     self.absolute_lambda
        #     and TransQueries.center3d in sample
        #     and ("target_cam_intr" in sample)
        # ):
        #     predict_center = True
        #     supervise_center = True
        # elif ("target_cam_intr" in sample) and self.lambda_joints2d:
        #     predict_center = True
        #     supervise_center = False
        # else:
        #     predict_center = False
        #     supervise_center = False
        # if predict_center:
        #     focals = sample["target_cam_intr"][:, 0, 0]
        #     u_0 = sample["target_cam_intr"][:, 0, 2]
        #     v_0 = sample["target_cam_intr"][:, 1, 2]
        #     absolute_input = torch.cat(
        #         (
        #             focals.unsqueeze(1),
        #             u_0.unsqueeze(1),
        #             v_0.unsqueeze(1),
        #             features,
        #         ),
        #         dim=1,
        #     )
        #     pred_center3d = self.absolute_branch(absolute_input)
        #     results["center3d"] = pred_center3d
        #     if not no_loss and supervise_center:
        #         absolute_loss = torch_f.mse_loss(
        #             pred_center3d, sample[TransQueries.center3d]
        #         ).view(1)
        #         if total_loss is None:
        #             total_loss = absolute_loss
        #         else:
        #             total_loss += self.absolute_lambda * absolute_loss
        #         losses["absolute_loss"] = absolute_loss
        
        # ============= compute mano results >>>>>>>>>>>>>
        # if sample["root"] == "palm":
        #     root_palm = True
        # else:
        #     root_palm = False
        mano_results = self.mano_branch(
            features,
            # **NOTICE**: sieds and root_palm are changed
            # sides=sample[BaseQueries.sides],
            # root_palm = root_palm
            sides=(sample["image"].shape)[0]*["right"],
            root_palm=True,
            use_stereoshape=False,
            )
        results["mano_results"] = mano_results
        if self.lambda_joints2d:
            scaletrans = self.scaletrans_branch(features)
            trans = scaletrans[:, 1:]
            # Abs to make sure no inversion in scale
            scale = torch.abs(scaletrans[:, :1])

            # Trans is multiplied by 100 to make scale and trans updates
            # of same magnitude after 2d joints supervision
            # (100 is ~ the scale of the 2D joint coordinate values)
            proj_joints2d = mano_results["joints"][
                :, :, :2
            ] * scale.unsqueeze(1) + 100 * trans.unsqueeze(1)
            results["joints2d"] = proj_joints2d


        # ============= source code >>>>>>>>>>>>>
        # ======= predict atlas >>>>>>>>
        predict_atlas = TransQueries.objpoints3d in sample.keys() and (
            self.atlas_lambda or self.atlas_final_lambda
        )
        if predict_atlas:
            if self.atlas_mesh:
                if self.adapt_atlas_decoder:
                    atlas_features = self.atlas_adapter(features)
                else:
                    atlas_features = features
                if self.atlas_separate_encoder:
                    atlas_results = self.atlas_branch.forward_inference(
                        atlas_features,
                        separate_encoder_features=atlas_infeatures,
                    )
                else:
                    atlas_results = self.atlas_branch.forward_inference(
                        atlas_features
                    )
            else:
                atlas_results = self.atlas_branch(features)
            results["atlas_results"] = atlas_results
            results["obj_utils"] = {
                "atlas_branch.test_faces": self.atlas_branch.test_faces,
                "mano_branch.faces": self.mano_branch.faces,
                "atlas_loss": self.atlas_loss,
            }

        return results
