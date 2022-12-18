import traceback
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from torch.utils.data import default_collate
from oib.utils.config import _DEFAULT_AUG_PARAM, _ZERO_AUG_PARAM
from oib.utils.logger import logger
from oib.utils.misc import CONST
from oib.utils.transform import (
    SimpleTransform2DHeatmap,
    SimpleTransform3DMANO,
    SimpleTransformUVD,
    SimpleTransformUVDOrthoMANO,
    fit_ortho_param,
    get_verts_2d_vis,
    ortho_project,
)

# https://github.com/lmb-freiburg/freihand/blob/master/utils/mano_utils.py
kpId2vertices = {4: [744], 8: [320], 12: [443], 16: [555], 20: [672]}  # ThumbT  # IndexT  # MiddleT  # RingT  # PinkT

OPENPOSE_JOINTS_NAME = [
    "loc_bn_palm_L",
    "loc_bn_thumb_L_01",
    "loc_bn_thumb_L_02",
    "loc_bn_thumb_L_03",
    "loc_bn_thumb_L_04",
    "loc_bn_index_L_01",
    "loc_bn_index_L_02",
    "loc_bn_index_L_03",
    "loc_bn_index_L_04",
    "loc_bn_mid_L_01",
    "loc_bn_mid_L_02",
    "loc_bn_mid_L_03",
    "loc_bn_mid_L_04",
    "loc_bn_ring_L_01",
    "loc_bn_ring_L_02",
    "loc_bn_ring_L_03",
    "loc_bn_ring_L_04",
    "loc_bn_pinky_L_01",
    "loc_bn_pinky_L_02",
    "loc_bn_pinky_L_03",
    "loc_bn_pinky_L_04",
]
OPENPOSE_JOINTS_NAME_TO_ID = {w: i for i, w in enumerate(OPENPOSE_JOINTS_NAME)}


class HDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.name = "HDataset"
        self.data_mode = cfg.DATA_MODE
        self.data_root = cfg.DATA_ROOT
        self.data_split = cfg.DATA_SPLIT
        self.use_cache = cfg.DATA_PRESET.USE_CACHE
        self.bbox_expand_ratio = float(cfg.DATA_PRESET.BBOX_EXPAND_RATIO)
        self.image_size = cfg.DATA_PRESET.IMAGE_SIZE  # (W, H)
        self.data_preset = cfg.DATA_PRESET

        self.include_mano = cfg.get("INCLUDE_MANO", True)

        self.center_idx = int(cfg.DATA_PRESET.CENTER_IDX)
        self.sides = CONST.SIDE
        self.njoints = CONST.NUM_JOINTS

        self.aug = cfg.AUG
        if cfg.AUG:
            if cfg.AUG_PARAM is None:
                self.aug_param = _DEFAULT_AUG_PARAM
            else:
                self.aug_param = cfg.AUG_PARAM
        else:
            self.aug_param = _ZERO_AUG_PARAM

        self.build_transform(self.aug_param)
        logger.info(f"Initialized abstract class: {self.name}")

    def __len__(self):
        return len(self)

    def build_transform(self, aug_param):
        basis_param = {
            "scale_jit_factor": aug_param.SCALE_JIT,
            "color_jit_factor": aug_param.COLOR_JIT,
            "rot_jit_factor": aug_param.ROT_JIT,
            "rot_prob": aug_param.ROT_PROB,
            "occlusion": aug_param.OCCLUSION,
            "occlusion_prob": aug_param.OCCLUSION_PROB,
            "output_size": self.image_size,
            "train": "train" in self.data_split,
            "aug": self.aug,
        }

        if self.data_mode == "2D":
            self.transform = SimpleTransform2DHeatmap(
                heatmap_size=self.data_preset.HEATMAP_SIZE, heatmap_sigma=self.data_preset.HEATMAP_SIGMA, **basis_param
            )
        elif self.data_mode == "UVD":
            self.transform = SimpleTransformUVD(center_idx=self.center_idx, **basis_param)
        elif self.data_mode == "3D":
            self.transform = SimpleTransform3DMANO(center_idx=self.center_idx, **basis_param)
        elif self.data_mode == "UVD_ortho":
            self.transform = SimpleTransformUVDOrthoMANO(center_idx=self.center_idx, **basis_param)
        else:
            raise ValueError(f"Unknown data mode: {self.data_mode}")

    @staticmethod
    def flip_2d(annot_2d, centerX):
        annot_2d = annot_2d.copy()
        annot_2d[:, 0] = centerX - annot_2d[:, 0]
        return annot_2d

    @staticmethod
    def flip_3d(annot_3d):
        annot_3d = annot_3d.copy()
        annot_3d[:, 0] = -annot_3d[:, 0]
        return annot_3d

    @staticmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def get_sample_idx(self) -> List[int]:
        pass

    @abstractmethod
    def get_image(self, idx):
        pass

    @abstractmethod
    def get_image_mask(self, idx):
        pass

    @abstractmethod
    def get_image_path(self, idx):
        pass

    @abstractmethod
    def get_joints_3d(self, idx):
        pass

    @abstractmethod
    def get_verts_3d(self, idx):
        pass

    @abstractmethod
    def get_verts_uvd(self, idx):
        pass

    @abstractmethod
    def get_bone_scale(self, idx):
        pass

    @abstractmethod
    def get_joints_2d(self, idx):
        pass

    @abstractmethod
    def get_joints_uvd(self, idx):
        pass

    @abstractmethod
    def get_cam_intr(self, idx):
        pass

    @abstractmethod
    def get_cam_center(self, idx):
        pass

    @abstractmethod
    def get_sides(self, idx):
        pass

    @abstractmethod
    def get_bbox_center_scale(self, idx):
        pass

    @abstractmethod
    def get_sample_identifier(self, idx):
        pass

    @abstractmethod
    def get_mano_pose(self, idx):
        pass

    @abstractmethod
    def get_mano_shape(self, idx):
        pass

    @abstractmethod
    def get_rawimage_size(self, idx):
        pass

    def get_joints_2d_vis(self, joints_2d=None, raw_size=None, **kwargs):
        joints_vis = ((joints_2d[:, 0] >= 0) & (joints_2d[:, 0] < raw_size[0])) & (
            (joints_2d[:, 1] >= 0) & (joints_2d[:, 1] < raw_size[1])
        )
        return joints_vis.astype(np.float32)

    def getitem_2d(self, idx):
        idx = self.get_sample_idx()[idx]
        hand_side = self.get_sides(idx)
        bbox_center, bbox_scale = self.get_bbox_center_scale(idx)
        bbox_scale = bbox_scale * self.bbox_expand_ratio  # extend bbox sacle
        # cam_center = self.get_cam_center(idx)
        joints_2d = self.get_joints_2d(idx)
        image_path = self.get_image_path(idx)
        image = self.get_image(idx)
        image_mask = self.get_image_mask(idx)

        raw_size = [image.shape[1], image.shape[0]]  # (W, H)
        joints_vis = self.get_joints_2d_vis(joints_2d=joints_2d, raw_size=raw_size)

        flip_hand = True if hand_side != self.sides else False

        if flip_hand:
            bbox_center[0] = raw_size[0] - bbox_center[0]  # image center
            joints_2d = self.flip_2d(joints_2d, raw_size[0])
            image = image[:, ::-1, :]
            image_mask = image_mask[:, ::-1]

        label = {
            "idx": idx,
            # "cam_center": cam_center,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "joints_2d": joints_2d,
            "joints_vis": joints_vis,
            "image_path": image_path,
            "raw_size": raw_size,
            "image_mask": image_mask,
        }
        return image, label

    def getitem_uvd(self, idx):
        # Support FreiHAND, HO3D, DexYCB, YT3D, TMANO,
        idx = self.get_sample_idx()[idx]
        hand_side = self.get_sides(idx)

        bbox_center, bbox_scale = self.get_bbox_center_scale(idx)
        bbox_scale = bbox_scale * self.bbox_expand_ratio  # extend bbox sacle
        verts_uvd = self.get_verts_uvd(idx)
        joints_uvd = self.get_joints_uvd(idx)
        joints_2d = self.get_joints_2d(idx)
        # cam_center = self.get_cam_center(idx)
        image_path = self.get_image_path(idx)
        image = self.get_image(idx)
        image_mask = self.get_image_mask(idx)

        raw_size = [image.shape[1], image.shape[0]]  # (W, H)
        joints_vis = get_verts_2d_vis(verts_2d=joints_2d, raw_size=raw_size)

        flip_hand = True if hand_side != self.sides else False

        ## Flip 2d if needed
        if flip_hand:
            bbox_center[0] = raw_size[0] - bbox_center[0]  # image center
            joints_2d = self.flip_2d(joints_2d, raw_size[0])
            joints_uvd = self.flip_2d(joints_uvd, raw_size[0])
            verts_uvd = self.flip_2d(verts_uvd, raw_size[0])
            image = image[:, ::-1, :]
            image_mask = image_mask[:, ::-1]

        label = {
            "idx": idx,
            # "cam_center": cam_center,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "joints_2d": joints_2d,
            "verts_uvd": verts_uvd,
            "joints_uvd": joints_uvd,
            "joints_vis": joints_vis,
            "image_path": image_path,
            "raw_size": raw_size,
            "image_mask": image_mask,
        }
        return image, label

    def getitem_3d(self, idx):
        # Support FreiHAND, HO3D, DexYCB
        idx = self.get_sample_idx()[idx]
        hand_side = self.get_sides(idx)
        bbox_center, bbox_scale = self.get_bbox_center_scale(idx)
        bbox_scale = bbox_scale * self.bbox_expand_ratio  # extend bbox sacle
        cam_intr = self.get_cam_intr(idx)
        cam_center = self.get_cam_center(idx)
        joints_3d = self.get_joints_3d(idx)
        verts_3d = self.get_verts_3d(idx)
        joints_2d = self.get_joints_2d(idx)
        verts_uvd = self.get_verts_uvd(idx)
        joints_uvd = self.get_joints_uvd(idx)

        if self.include_mano:
            mano_pose = self.get_mano_pose(idx)
            mano_shape = self.get_mano_shape(idx)

        image_path = self.get_image_path(idx)
        image = self.get_image(idx)
        raw_size = [image.shape[1], image.shape[0]]  # (W, H)
        joints_vis = get_verts_2d_vis(verts_2d=joints_2d, raw_size=raw_size)

        image_mask = self.get_image_mask(idx)

        flip_hand = True if hand_side != self.sides else False

        # Flip 2d if needed
        if flip_hand:
            bbox_center[0] = raw_size[0] - bbox_center[0]  # image center
            joints_3d = self.flip_3d(joints_3d)
            verts_3d = self.flip_3d(verts_3d)
            joints_uvd = self.flip_2d(joints_uvd, raw_size[0])
            verts_uvd = self.flip_2d(verts_uvd, raw_size[0])
            joints_2d = self.flip_2d(joints_2d, centerX=raw_size[0])
            image = image[:, ::-1, :]
            image_mask = image_mask[:, ::-1]

        label = {
            "idx": idx,
            "cam_center": cam_center,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "cam_intr": cam_intr,
            "joints_2d": joints_2d,
            "joints_3d": joints_3d,
            "verts_3d": verts_3d,
            "joints_vis": joints_vis,
            "joints_uvd": joints_uvd,
            "verts_uvd": verts_uvd,
            "image_path": image_path,
            "raw_size": raw_size,
            "image_mask": image_mask,
        }
        if self.include_mano:
            label.update({"mano_pose": mano_pose, "mano_shape": mano_shape})
        return image, label

    def getitem_uvd_ortho(self, idx):
        idx = self.get_sample_idx()[idx]
        hand_side = self.get_sides(idx)
        bbox_center, bbox_scale = self.get_bbox_center_scale(idx)
        bbox_scale = bbox_scale * self.bbox_expand_ratio  # extend bbox sacle

        joints_3d = self.get_joints_3d(idx)
        joints_2d = self.get_joints_2d(idx)
        joints_uvd = np.concatenate((joints_2d, joints_3d[:, 2:]), axis=1)

        CID = self.cfg.DATA_PRESET.CENTER_IDX
        ortho_intr = fit_ortho_param((joints_3d - joints_3d[CID, :]), joints_2d)

        verts_3d = self.get_verts_3d(idx)
        verts_uv = ortho_project(verts_3d - joints_3d[CID, :], ortho_intr)
        verts_d = verts_3d[:, 2:]
        verts_uvd = np.concatenate((verts_uv, verts_d), axis=1)

        if self.include_mano:
            mano_pose = self.get_mano_pose(idx)
            mano_shape = self.get_mano_shape(idx)

        image_path = self.get_image_path(idx)
        image = self.get_image(idx)
        raw_size = [image.shape[1], image.shape[0]]  # (W, H)

        joints_vis = get_verts_2d_vis(verts_2d=joints_2d, raw_size=raw_size)
        image_mask = self.get_image_mask(idx)

        flip_hand = True if hand_side != self.sides else False

        # Flip 2d if needed
        if flip_hand:
            bbox_center[0] = raw_size[0] - bbox_center[0]  # image center
            joints_3d = self.flip_3d(joints_3d)
            verts_3d = self.flip_3d(verts_3d)
            joints_uvd = self.flip_2d(joints_uvd, raw_size[0])
            verts_uvd = self.flip_2d(verts_uvd, raw_size[0])
            joints_2d = self.flip_2d(joints_2d, centerX=raw_size[0])
            image = image[:, ::-1, :]
            image_mask = image_mask[:, ::-1]

        label = {
            "idx": idx,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "ortho_intr": ortho_intr,
            "joints_2d": joints_2d,
            "joints_3d": joints_3d,
            "verts_3d": verts_3d,
            "joints_vis": joints_vis,
            "joints_uvd": joints_uvd,
            "verts_uvd": verts_uvd,
            "image_path": image_path,
            "raw_size": raw_size,
            "image_mask": image_mask,
        }
        if self.include_mano:
            label.update({"mano_pose": mano_pose, "mano_shape": mano_shape})

        return image, label

    def __getitem__(self, idx):
        if self.data_mode not in ["2D", "3D", "UVD", "UVD_ortho"]:
            raise NotImplementedError(f"Unknown data mode: {self.data_mode}")

        try:
            if self.data_mode == "2D":
                image, label = self.getitem_2d(idx)
            elif self.data_mode == "UVD":
                image, label = self.getitem_uvd(idx)
            elif self.data_mode == "3D":
                image, label = self.getitem_3d(idx)
            elif self.data_mode == "UVD_ortho":
                image, label = self.getitem_uvd_ortho(idx)

            results = self.transform(image, label)
            results.update(label)
            results.pop("image_mask", None)
            return results

        except:
            traceback.print_exc()
            _idx = np.random.randint(len(self))
            return self.__getitem__(_idx)

    def collate_fn(self, batch):
        return default_collate(batch)
