import torch
from torch import nn
from torch.nn import functional as torch_f
import numpy as np
from collections import OrderedDict
import os
from contextlib import nullcontext
from oib.models.model_abstraction import ModuleAbstract
from oib.criterions.criterion import TensorLoss
from oib.metrics.basic_metric import VisMetric
from oib.utils.builder import MODEL
from oib.utils.logger import logger
from oib.utils.misc import CONST, param_size, enable_lower_param
from oib.utils.transform import batch_uvd2xyz, rot6d_to_aa, batch_persp_project, batch_xyz2uvd
from oib.viztools.draw import draw_batch_joint_images, concat_imgs

from .resnet import ResNetBackbone
from .module import PoseNet, Pose2Feat, ParamRegressor, MeshNet
from .loss import CoordLoss, ParamLoss, NormalVectorLoss, EdgeLengthLoss

from manotorch.manolayer import ManoLayer


@MODEL.register_module
class I2L_MeshNet(ModuleAbstract):
    @enable_lower_param
    def __init__(self, cfg):
        super(I2L_MeshNet, self).__init__(cfg)

        self.center_idx = cfg["DATA_PRESET"].get("CENTER_IDX", 0)
        assert self.center_idx == 0, "I2L_MeshNet only support center_idx 0"

        self.inp_res = cfg["DATA_PRESET"]["IMAGE_SIZE"]
        self.output_hm_shape = cfg["DATA_PRESET"]["HEATMAP_SIZE"]
        assert len(self.output_hm_shape) == 3, f"wrong heatmap dim, got {len(self.output_hm_shape)}"
        self.sigma = cfg["HEATMAP_SIGMA"]

        self.stage = cfg["STAGE"]
        assert self.stage in {"lixel", "param"}, f"got unknown model stage {self.stage}"

        # get layers and nets
        self.pose_backbone = ResNetBackbone(cfg["POSE_BACKBONE"]["RESNET_TYPE"])
        self.pose_net = PoseNet(21, self.output_hm_shape)
        self.pose2feat = Pose2Feat(21, self.output_hm_shape)
        self.mesh_backbone = ResNetBackbone(cfg["MESH_BACKBONE"]["RESNET_TYPE"])
        self.mesh_net = MeshNet(778, self.output_hm_shape)
        self.param_regressor = ParamRegressor(21)

        self.human_model_layer = ManoLayer(
            center_idx=None, flat_hand_mean=False, use_pca=False, mano_assets_root="assets/mano_v1_2"
        )  # no center_idx
        # ======
        self.joint_regressor = self.human_model_layer.th_J_regressor.numpy()
        self.fingertip_vertex_idx = [745, 317, 444, 556, 673]  # mesh vertex idx (right hand)
        thumbtip_onehot = np.array(
            [1 if i == 745 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32
        ).reshape(1, -1)
        indextip_onehot = np.array(
            [1 if i == 317 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32
        ).reshape(1, -1)
        middletip_onehot = np.array(
            [1 if i == 445 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32
        ).reshape(1, -1)
        ringtip_onehot = np.array(
            [1 if i == 556 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32
        ).reshape(1, -1)
        pinkytip_onehot = np.array(
            [1 if i == 673 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32
        ).reshape(1, -1)
        self.joint_regressor = np.concatenate(
            (self.joint_regressor, thumbtip_onehot, indextip_onehot, middletip_onehot, ringtip_onehot, pinkytip_onehot)
        )
        self.joint_regressor = self.joint_regressor[
            [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20], :
        ]
        # >>>>>

        self.init_weights(pretrained=cfg["PRETRAINED"])

        logger.info(f"{type(self).__name__} uses center_idx {self.center_idx}")
        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters")

    def make_gaussian_heatmap(self, joint_coord_img):
        x = torch.arange(self.output_hm_shape[2])
        y = torch.arange(self.output_hm_shape[1])
        z = torch.arange(self.output_hm_shape[0])
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        xx = xx[None, None, :, :, :].to(joint_coord_img.device).float()
        yy = yy[None, None, :, :, :].to(joint_coord_img.device).float()
        zz = zz[None, None, :, :, :].to(joint_coord_img.device).float()

        x = joint_coord_img[:, :, 0, None, None, None]
        y = joint_coord_img[:, :, 1, None, None, None]
        z = joint_coord_img[:, :, 2, None, None, None]
        heatmap = torch.exp(
            -(((xx - x) / self.sigma) ** 2) / 2
            - (((yy - y) / self.sigma) ** 2) / 2
            - (((zz - z) / self.sigma) ** 2) / 2
        )
        return heatmap

    # ***** forward *****
    def forward(self, inputs, mode="train", **kwargs):
        if mode == "train":
            return self.training_step(inputs, **kwargs)
        elif mode == "val":
            return self.validation_step(inputs, **kwargs)
        elif mode == "test":
            return self.testing_step(inputs, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def _forward_impl(self, inputs):
        if self.stage == "lixel":
            cm = nullcontext()
        else:
            cm = torch.no_grad()

        with cm:
            # posenet forward
            shared_img_feat, pose_img_feat = self.pose_backbone(inputs["image"])
            joint_coord_img = self.pose_net(pose_img_feat)

            # make 3D heatmap from posenet output and convert to image feature
            with torch.no_grad():
                joint_heatmap = self.make_gaussian_heatmap(joint_coord_img.detach())
            shared_img_feat = self.pose2feat(shared_img_feat, joint_heatmap)

            # meshnet forward
            _, mesh_img_feat = self.mesh_backbone(shared_img_feat, skip_early=True)
            mesh_coord_img = self.mesh_net(mesh_img_feat)

            # joint coordinate outputs from mesh coordinates
            joint_img_from_mesh = torch.bmm(
                torch.from_numpy(self.joint_regressor)
                .to(mesh_coord_img.device)[None, :, :]
                .repeat(mesh_coord_img.shape[0], 1, 1),
                mesh_coord_img,
            )
            mesh_coord_cam = None

        if self.stage == "param":
            # parameter regression
            pose_param, shape_param = self.param_regressor(joint_img_from_mesh.detach())

            # get mesh and joint coordinates
            mesh_coord_cam, _ = self.human_model_layer(pose_param, shape_param)
            joint_coord_cam = torch.bmm(
                torch.from_numpy(self.joint_regressor)
                .to(mesh_coord_img.device)[None, :, :]
                .repeat(mesh_coord_cam.shape[0], 1, 1),
                mesh_coord_cam,
            )

            # root-relative 3D coordinates
            root_joint_cam = joint_coord_cam[:, self.center_idx, None, :]
            mesh_coord_cam = mesh_coord_cam - root_joint_cam
            joint_coord_cam = joint_coord_cam - root_joint_cam

        out = {}
        out["joint_coord_img"] = torch.nan_to_num(joint_coord_img)
        out["joint_img_from_mesh"] = torch.nan_to_num(joint_img_from_mesh)
        out["mesh_coord_img"] = torch.nan_to_num(mesh_coord_img)

        if self.stage == "param":
            out["pose_param"] = pose_param
            out["shape_param"] = shape_param
            out["joint_coord_cam"] = torch.nan_to_num(joint_coord_cam)
            out["mesh_coord_cam"] = torch.nan_to_num(mesh_coord_cam)

        # recover xyz from uvd
        # ! note these things does not run with grad!
        heatmap_size = torch.Tensor((self.output_hm_shape[2], self.output_hm_shape[1], self.output_hm_shape[0]))
        heatmap_size = heatmap_size.to(mesh_coord_img.device)

        pred_verts_3d_uvd = mesh_coord_img.detach().clone()
        pred_verts_3d_uvd = pred_verts_3d_uvd / heatmap_size[None, None, :]
        batch_size = pred_verts_3d_uvd.shape[0]

        root_joints = inputs["target_root_joint"]
        cam_intrs = inputs["target_cam_intr"]

        pred_verts_3d = batch_uvd2xyz(
            pred_verts_3d_uvd,
            root_joints,
            cam_intrs,
            inp_res=self.inp_res,
            depth_range=CONST.UVD_DEPTH_RANGE,
            ref_bone_len=None,
        )
        pred_verts_3d = torch.nan_to_num(pred_verts_3d)
        pred_joints_3d = (
            torch.from_numpy(self.joint_regressor)
            .to(pred_verts_3d.device)[None, :, :]
            .repeat(mesh_coord_img.shape[0], 1, 1)
            @ pred_verts_3d
        )
        pred_joints_3d = torch.nan_to_num(pred_joints_3d)

        # sanity check!
        assert torch.all(torch.isfinite(pred_verts_3d))
        assert torch.all(torch.isfinite(pred_joints_3d))

        # keys for compatibility
        out["joints_3d"] = pred_joints_3d
        out["joints_3d_rel"] = pred_joints_3d - pred_joints_3d[:, (self.center_idx,)]  # freihand use 0 for center_idx
        out["verts_3d"] = pred_verts_3d
        out["verts_3d_rel"] = pred_verts_3d - pred_joints_3d[:, (self.center_idx,)]  # freihand use 0 for center_idx
        out["2d_uvd"] = joint_coord_img.detach().clone() / heatmap_size[None, None, :]

        return out

    def training_step(self, batch, **kwargs):
        # forward your network
        preds = self._forward_impl(batch)

        # compute loss
        final_loss, final_loss_dict = self.compute_loss(preds, batch)
        preds["joints_2d"] = batch_persp_project(preds["joints_3d"], batch["target_cam_intr"])

        # evaluator
        with torch.no_grad():
            self.evaluator.feed_all(preds, batch, final_loss_dict)

        return preds, final_loss_dict

    def validation_step(self, batch, **kwargs):
        preds = self._forward_impl(batch)
        final_loss_dict = {}  # * we don't need to compute loss for validation
        return preds, final_loss_dict

    def testing_step(self, batch, **kwargs):
        preds = self._forward_impl(batch)

        preds["joints_2d"] = batch_persp_project(preds["joints_3d"], batch["target_cam_intr"])

        if kwargs.get("disable_evaluator", False):
            final_loss_dict = {}
        else:
            final_loss_dict = {}  # * we don't need to compute loss for testing
            with torch.no_grad():
                self.evaluator.feed_all(preds, batch, final_loss_dict)

        if "callback" in kwargs:
            kwargs["callback"](preds=preds, inputs=batch)

        return preds, final_loss_dict

    # ***** network loss *****
    class I2L_MeshNet_Loss(TensorLoss):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg

            self.inp_res = cfg["DATA_PRESET"]["IMAGE_SIZE"]
            self.output_hm_shape = cfg["DATA_PRESET"]["HEATMAP_SIZE"]

            self.lambda_joint_fit = cfg.get("LAMBDA_JOINT_FIT", 1.0)
            self.lambda_mesh_fit = cfg.get("LAMBDA_MESH_FIT", 1.0)
            self.lambda_mesh_joint_fit = cfg.get("LAMBDA_MESH_JOINT_FIT", 1.0)
            self.lambda_mesh_normal = cfg.get("LAMBDA_MESH_NORMAL", 0.1)
            self.lambda_mesh_edge = cfg.get("LAMBDA_MESH_EDGE", 1.0)

            self.lambda_pose_param = cfg.get("LAMBDA_POSE_PARAM", 1.0)
            self.lambda_shape_param = cfg.get("LAMBDA_SHAPE_PARAM", 1.0)
            self.lambda_joint_fit_cam = cfg.get("LAMBDA_JOINT_FIT_CAM", 1.0)

            self.coord_loss = CoordLoss()
            self.param_loss = ParamLoss()
            self.normal_loss = NormalVectorLoss()
            self.edge_loss = EdgeLengthLoss()

            logger.info(f"Construct {type(self).__name__} with lambda: ")
            logger.info(f"  |   LAMBDA_JOINT_FIT      : {self.lambda_joint_fit}")
            logger.info(f"  |   LAMBDA_MESH_FIT       : {self.lambda_mesh_fit}")
            logger.info(f"  |   LAMBDA_MESH_JOINT_FIT : {self.lambda_mesh_joint_fit}")
            logger.info(f"  |   LAMBDA_MESH_NORMAL    : {self.lambda_mesh_normal}")
            logger.info(f"  |   LAMBDA_MESH_EDGE      : {self.lambda_mesh_edge}")
            logger.info(f"  |   LAMBDA_POSE_PARAM     : {self.lambda_pose_param}")
            logger.info(f"  |   LAMBDA_SHAPE_PARAM    : {self.lambda_shape_param}")
            logger.info(f"  |   LAMBDA_JOINT_FIT_CAM  : {self.lambda_joint_fit_cam}")

        def init_loss(self, preds):
            target_device = None
            losses = {}

            # Get device
            for key in preds.keys():
                if isinstance(preds[key], torch.Tensor):
                    target_device = preds[key].device
                    break
            if target_device is None:
                logger.error("Cannot found valid Tensor with device")
                raise RuntimeError()

            final_loss = torch.Tensor([0.0]).float().to(target_device)
            return final_loss, losses

        def forward(self, preds, gt):
            final_loss, losses = self.init_loss(preds)  # TENSOR(0.), {}

            # transform uvd to heatmap version
            heatmap_size = torch.Tensor((self.output_hm_shape[2], self.output_hm_shape[1], self.output_hm_shape[0]))
            heatmap_size = heatmap_size.to(preds["mesh_coord_img"].device)
            root_joints = gt["target_root_joint"]
            cam_intrs = gt["target_cam_intr"]
            target_joints_uvd = batch_xyz2uvd(
                gt["target_joints_3d"],
                root_joints,
                cam_intrs,
                inp_res=self.inp_res,
                depth_range=CONST.UVD_DEPTH_RANGE,
                ref_bone_len=None,
            )
            target_joints_uvd = target_joints_uvd * heatmap_size[None, :]
            target_verts_uvd = batch_xyz2uvd(
                gt["target_verts_3d"],
                root_joints,
                cam_intrs,
                inp_res=self.inp_res,
                depth_range=CONST.UVD_DEPTH_RANGE,
                ref_bone_len=None,
            )
            target_verts_uvd = target_verts_uvd * heatmap_size[None, :]

            if self.lambda_joint_fit > 0:
                pred_joints_3d_uvd = preds["joint_coord_img"]
                joints_3d_uvd = target_joints_uvd  # TENSOR(B, NJOINTS, 3)
                joint_vis = gt["target_joints_vis"][..., None]
                joint_fit_loss = self.coord_loss(
                    pred_joints_3d_uvd,
                    joints_3d_uvd.to(final_loss.device),
                    joint_vis.to(final_loss.device),
                )
                joint_fit_loss = joint_fit_loss.mean()
                final_loss += self.lambda_joint_fit * joint_fit_loss
            else:
                joint_fit_loss = None
            losses["joint_fit_loss"] = joint_fit_loss

            if self.lambda_mesh_fit > 0:
                pred_mesh_3d_uvd = preds["mesh_coord_img"]
                mesh_3d_uvd = target_verts_uvd
                mesh_vis = gt["target_verts_vis"][..., None]
                mesh_fit_loss = self.coord_loss(
                    pred_mesh_3d_uvd,
                    mesh_3d_uvd.to(final_loss.device),
                    mesh_vis.to(final_loss.device),
                )
                mesh_fit_loss = mesh_fit_loss.mean()
                final_loss += self.lambda_mesh_fit * mesh_fit_loss
            else:
                mesh_fit_loss = None
            losses["mesh_fit_loss"] = mesh_fit_loss

            if self.lambda_mesh_joint_fit > 0:
                pred_joints_3d_uvd_fit = preds["joint_img_from_mesh"]
                joints_3d_uvd = target_joints_uvd  # TENSOR(B, NJOINTS, 3)
                joint_vis = gt["target_joints_vis"][..., None]
                mesh_joint_fit_loss = self.coord_loss(
                    pred_joints_3d_uvd_fit,
                    joints_3d_uvd.to(final_loss.device),
                    joint_vis.to(final_loss.device),
                )
                mesh_joint_fit_loss = mesh_joint_fit_loss.mean()
                final_loss += self.lambda_mesh_joint_fit * mesh_joint_fit_loss
            else:
                mesh_joint_fit_loss = None
            losses["mesh_joint_fit_loss"] = mesh_joint_fit_loss

            if self.lambda_mesh_normal > 0:
                pred_mesh_3d_uvd = preds["mesh_coord_img"]
                mesh_3d_uvd = target_verts_uvd
                mesh_vis = gt["target_verts_vis"][..., None]
                face = gt["hand_faces"].long()
                mesh_normal_loss = self.normal_loss(
                    pred_mesh_3d_uvd,
                    mesh_3d_uvd.to(final_loss.device),
                    mesh_vis.to(final_loss.device),
                    face,
                )
                mesh_normal_loss = mesh_normal_loss.mean()
                final_loss += self.lambda_mesh_normal * mesh_normal_loss
            else:
                mesh_normal_loss = None
            losses["mesh_normal_loss"] = mesh_normal_loss

            if self.lambda_mesh_edge > 0:
                pred_mesh_3d_uvd = preds["mesh_coord_img"]
                mesh_3d_uvd = target_verts_uvd
                mesh_vis = gt["target_verts_vis"][..., None]
                face = gt["hand_faces"].long()
                mesh_edge_loss = self.edge_loss(
                    pred_mesh_3d_uvd,
                    mesh_3d_uvd.to(final_loss.device),
                    mesh_vis.to(final_loss.device),
                    face,
                )
                mesh_edge_loss = mesh_edge_loss.mean()
                final_loss += self.lambda_mesh_edge * mesh_edge_loss
            else:
                mesh_edge_loss = None
            losses["mesh_edge_loss"] = mesh_edge_loss

            if self.lambda_pose_param > 0:
                pred_pose_param = preds["pose_param"]
                batch_size = pred_pose_param.shape[0]
                pose_param = gt["target_mano_pose"]
                pose_param_loss = self.param_loss(
                    pred_pose_param,
                    pose_param.to(final_loss.device),
                    torch.ones((batch_size, 1), device=final_loss.device),
                )
                pose_param_loss = pose_param_loss.mean()
                final_loss += self.lambda_pose_param * pose_param_loss
            else:
                pose_param_loss = None
            losses["pose_param_loss"] = pose_param_loss

            if self.lambda_shape_param > 0:
                pred_shape_param = preds["shape_param"]
                batch_size = pred_shape_param.shape[0]
                shape_param = gt["target_mano_shape"]
                shape_param_loss = self.param_loss(
                    pred_shape_param,
                    shape_param.to(final_loss.device),
                    torch.ones((batch_size, 1), device=final_loss.device),
                )
                shape_param_loss = shape_param_loss.mean()
                final_loss += self.lambda_shape_param * shape_param_loss
            else:
                shape_param_loss = None
            losses["shape_param_loss"] = shape_param_loss

            if self.lambda_joint_fit_cam > 0:
                pred_joints_3d_cam = preds["joint_coord_cam"]
                joints_3d_cam = gt["target_joints_3d_rel"]
                batch_size = pred_joints_3d_cam.shape[0]
                joint_fit_cam_loss = self.coord_loss(
                    pred_joints_3d_cam,
                    joints_3d_cam.to(final_loss.device),
                    torch.ones((batch_size, 1, 1), device=final_loss.device),
                )
                joint_fit_cam_loss = joint_fit_cam_loss.mean()
                final_loss += self.lambda_joint_fit_cam * joint_fit_cam_loss
            else:
                joint_fit_cam_loss = None
            losses["joint_fit_cam_loss"] = joint_fit_cam_loss

            return final_loss, losses

    # ***** network metrics *****
    class I2L_MeshNet_Vis_Metric(VisMetric):
        def feed(self, preds, targs):
            if self.images is not None:  # * only visualize the first batch
                return
            img = draw_batch_joint_images(preds["joints_2d"], targs["target_joints_2d"], targs["image"])
            self.images = [concat_imgs(img)]

    # ***** network initializer *****
    def init_weights(self, pretrained=""):
        if pretrained == "":
            logger.warning(f"=> Init {type(self).__name__} weights in backbone and head")
            """
            Add init for other modules
            ...
            """
            self.pose_backbone.init_weights()
            self.pose_net.apply(init_weights)
            self.pose2feat.apply(init_weights)
            self.mesh_backbone.init_weights()
            self.mesh_net.apply(init_weights)
            self.param_regressor.apply(init_weights)
        elif os.path.isfile(pretrained):
            # pretrained_state_dict = torch.load(pretrained)
            logger.info(f"=> Loading {type(self).__name__} pretrained model from: {pretrained}")
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained)
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
        else:
            logger.error(f"=> No {type(self).__name__} checkpoints file found in {pretrained}")
            raise FileNotFoundError()


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)
