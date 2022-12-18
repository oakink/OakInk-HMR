import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import List

import imageio
import numpy as np
import torch
from manotorch.manolayer import ManoLayer
from oib.datasets.hdata import HDataset
from oib.utils.builder import DATASET
from oib.utils.etqdm import etqdm
from oib.utils.logger import logger
from oib.utils.transform import get_annot_center, get_annot_scale, persp_project
from termcolor import colored
from oib.utils.transform import (
    SimpleTransform2D,
    SimpleTransform3D,
    SimpleTransform3DMANO,
    SimpleTransformUVD,
    get_verts_2d_vis,
)
from ..utils.transform import quat_to_rotmat, rotmat_to_aa, quat_to_aa


@DATASET.register_module
class OakInk(HDataset):
    @staticmethod
    def _get_info_list(data_dir, split_key, data_split):
        if data_split == "train+val":
            info_list = json.load(open(os.path.join(data_dir, "anno", "split", split_key, "seq_train.json")))
        elif data_split == "train":
            info_list = json.load(
                open(os.path.join(data_dir, "anno", "split_train_val", split_key, "example_split_train.json"))
            )
        elif data_split == "val":
            info_list = json.load(
                open(os.path.join(data_dir, "anno", "split_train_val", split_key, "example_split_val.json"))
            )
        else:  # data_split == "test":
            info_list = json.load(open(os.path.join(data_dir, "anno", "split", split_key, "seq_test.json")))
        return info_list

    def __init__(self, cfg):
        super(OakInk, self).__init__(cfg)
        self.rMANO = ManoLayer(side="right", mano_assets_root="assets/mano_v1_2")
        assert self.data_mode in ["2D", "UVD", "3D"], f"OakInk does not dupport {self.data_mode} mode"
        assert self.data_split in [
            "all",
            "train+val",
            "train",
            "val",
            "test",
        ], "OakInk data_split must be one of ['train', 'val']"
        self.split_mode = cfg.SPLIT_MODE
        assert self.split_mode in [
            "default",
            "subject",
            "object",
        ], "OakInk split_mode must be one of ['default', 'subject', 'object]"
        self.use_pack = cfg.get("USE_PACK", False)
        if self.use_pack:
            self.getitem_3d = self._getitem_3d_pack
        else:
            self.getitem_3d = self._getitem_3d

        # determine whether to return extra stuff
        self.with_mano = cfg.get("WITH_MANO", False)
        self.with_verts_2d = cfg.get("WITH_VERTS_2D", False)
        self.with_hand_faces = cfg.get("WITH_HAND_FACES", False)

        self.load_dataset()
        logger.info(f"initialized child class: {self.name}")
        logger.info(f"\twith_mano: {self.with_mano}")
        logger.info(f"\twith_verts_2d: {self.with_verts_2d}")
        logger.info(f"\twith_hand_faces: {self.with_hand_faces}")

    def load_dataset(self):
        self.name = "oakink-image"
        self.root = os.path.join(self.data_root, "OakInk", "image")

        if self.data_split == "all":
            self.info_list = json.load(open(os.path.join(self.root, "anno", "seq_all.json")))
        elif self.split_mode == "default":
            self.info_list = self._get_info_list(self.root, "split0", self.data_split)
        elif self.split_mode == "subject":
            self.info_list = self._get_info_list(self.root, "split1", self.data_split)
        else:  # self.split_mode == "object":
            self.info_list = self._get_info_list(self.root, "split2", self.data_split)

        self.info_str_list = []
        for info in self.info_list:
            info_str = "__".join([str(x) for x in info])
            info_str = info_str.replace("/", "__")
            self.info_str_list.append(info_str)

        self.framedata_color_name = [
            "north_east_color",
            "south_east_color",
            "north_west_color",
            "south_west_color",
        ]

        self.n_samples = len(self.info_str_list)
        self.sample_idxs = list(range(self.n_samples))
        logger.info(
            f"{self.name} Got {colored(len(self.sample_idxs), 'yellow', attrs=['bold'])}"
            f"/{self.n_samples} samples for data_split {self.data_split}"
        )

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
            self.transform = SimpleTransform2D(
                heatmap_size=self.data_preset.HEATMAP_SIZE, heatmap_sigma=self.data_preset.HEATMAP_SIGMA, **basis_param
            )
        elif self.data_mode == "UVD":
            self.transform = SimpleTransformUVD(center_idx=self.center_idx, **basis_param)
        elif self.data_mode == "3D":
            self.transform = SimpleTransform3D(center_idx=self.center_idx, **basis_param)
        elif self.data_mode == "3DMANO":
            self.transform = SimpleTransform3DMANO(center_idx=self.center_idx, **basis_param)
        else:
            raise ValueError(f"Unknown data mode: {self.data_mode}")

    def __len__(self):
        return len(self.sample_idxs)

    def get_sample_idx(self) -> List[int]:
        return self.sample_idxs

    def get_image_path(self, idx):
        info = self.info_list[idx]
        # compute image path
        offset = os.path.join(info[0], f"{self.framedata_color_name[info[3]]}_{info[2]}.png")
        image_path = os.path.join(self.root, "stream_release_v2", offset)
        return image_path

    def get_image(self, idx):
        path = self.get_image_path(idx)
        image = np.array(imageio.imread(path, pilmode="RGB"), dtype=np.uint8)
        return image

    def get_rawimage_size(self, idx):
        # MUST (W, H)
        return (848, 480)

    def get_image_mask(self, idx):
        # mask_path = os.path.join(self.root, "mask", f"{self.info_str_list[idx]}.png")
        # mask = np.array(imageio.imread(mask_path, as_gray=True), dtype=np.uint8)
        # return mask
        return np.zeros((480, 848), dtype=np.uint8)

    def get_cam_intr(self, idx):
        cam_path = os.path.join(self.root, "anno", "cam_intr", f"{self.info_str_list[idx]}.pkl")
        with open(cam_path, "rb") as f:
            cam_intr = pickle.load(f)
        return cam_intr

    def get_cam_center(self, idx):
        intr = self.get_cam_intr(idx)
        return np.array([intr[0, 2], intr[1, 2]])

    def get_hand_faces(self, idx):
        return self.rMANO.get_mano_closed_faces().numpy()

    def get_joints_3d(self, idx):
        joints_path = os.path.join(self.root, "anno", "hand_j", f"{self.info_str_list[idx]}.pkl")
        with open(joints_path, "rb") as f:
            joints_3d = pickle.load(f)
        return joints_3d

    def get_verts_3d(self, idx):
        verts_path = os.path.join(self.root, "anno", "hand_v", f"{self.info_str_list[idx]}.pkl")
        with open(verts_path, "rb") as f:
            verts_3d = pickle.load(f)
        return verts_3d

    def get_joints_2d(self, idx):
        cam_intr = self.get_cam_intr(idx)
        joints_3d = self.get_joints_3d(idx)
        return persp_project(joints_3d, cam_intr)

    def get_joints_uvd(self, idx):
        uv = self.get_joints_2d(idx)
        d = self.get_joints_3d(idx)[:, 2:]  # (21, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_verts_uvd(self, idx):
        v3d = self.get_verts_3d(idx)
        intr = self.get_cam_intr(idx)
        uv = persp_project(v3d, intr)[:, :2]
        d = v3d[:, 2:]  # (778, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_verts_2d(self, idx):
        cam_intr = self.get_cam_intr(idx)
        verts_3d = self.get_verts_3d(idx)
        return persp_project(verts_3d, cam_intr)

    def get_sides(self, idx):
        return "right"

    def get_bone_scale(self, idx):
        raise NotImplementedError(f"{self.name} does not support bone scale")

    def get_bbox_center_scale(self, idx):
        joints_2d = self.get_joints_2d(idx)
        center = get_annot_center(joints_2d)
        scale = get_annot_scale(joints_2d)
        return center, scale

    def get_mano_pose(self, idx):
        general_info_path = os.path.join(
            self._data_dir, "image", "anno", "general_info", f"{self.info_str_list[idx]}.pkl"
        )
        with open(general_info_path, "rb") as f:
            general_info = pickle.load(f)
        raw_hand_anno = general_info["hand_anno"]

        raw_hand_pose = (raw_hand_anno["hand_pose"]).reshape((16, 4))  # quat (16, 4)
        _wrist, _remain = raw_hand_pose[0, :], raw_hand_pose[1:, :]
        cam_extr = general_info["cam_extr"]  # SE3 (4, 4))
        extr_R = cam_extr[:3, :3]  # (3, 3)

        wrist_R = extr_R.matmul(quat_to_rotmat(_wrist))  # (3, 3)
        wrist = rotmat_to_aa(wrist_R).unsqueeze(0).numpy()  # (1, 3)
        remain = quat_to_aa(_remain).numpy()  # (15, 3)
        hand_pose = np.concatenate([wrist, remain], axis=0)  # (16, 3)

        return hand_pose.astype(np.float32)

    def get_mano_shape(self, idx):
        general_info_path = os.path.join(
            self._data_dir, "image", "anno", "general_info", f"{self.info_str_list[idx]}.pkl"
        )
        with open(general_info_path, "rb") as f:
            general_info = pickle.load(f)
        raw_hand_anno = general_info["hand_anno"]
        hand_shape = raw_hand_anno["hand_shape"].numpy().astype(np.float32)
        return hand_shape

    def get_sample_identifier(self, idx):
        res = f"{self.name}__{self.info_str_list[idx]}"
        return res

    def _getitem_3d(self, idx):
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
        if self.with_mano:
            mano_pose = self.get_mano_pose(idx)
            mano_shape = self.get_mano_shape(idx)

        image_path = self.get_image_path(idx)
        image = self.get_image(idx)
        raw_size = [image.shape[1], image.shape[0]]  # (W, H)
        joints_vis = get_verts_2d_vis(verts_2d=joints_2d, raw_size=raw_size)
        if self.with_verts_2d:
            verts_2d = self.get_verts_2d(idx)
            verts_vis = get_verts_2d_vis(verts_2d=verts_2d, raw_size=raw_size)

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
            if self.with_verts_2d:
                self.flip_2d(verts_2d, raw_size[0])

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
        if self.with_mano:
            label.update(
                {
                    "mano_pose": mano_pose,
                    "mano_shape": mano_shape,
                }
            )
        if self.with_verts_2d:
            label.update(
                {
                    "verts_2d": verts_2d,
                    "verts_vis": verts_vis,
                }
            )
        if self.with_hand_faces:
            label["hand_faces"] = np.asarray(self.rMANO.th_faces)
        return image, label

    def _getitem_3d_pack(self, idx):
        idx = self.get_sample_idx()[idx]
        hand_side = self.get_sides(idx)
        # load pack
        pack_path = os.path.join(
            self.root, "anno_packed", self.split_mode, self.data_split, f"{self.info_str_list[idx]}.pkl"
        )
        with open(pack_path, "rb") as f:
            packed = pickle.load(f)

        cam_intr = np.array(packed["cam_intr"])
        joints_3d = np.array(packed["hand_j"])
        verts_3d = np.array(packed["hand_v"])
        joints_2d = persp_project(joints_3d, cam_intr)
        verts_2d = persp_project(verts_3d, cam_intr)
        joints_uvd = np.concatenate((joints_2d, joints_3d[:, 2:]), axis=1)
        verts_uvd = np.concatenate((verts_2d, verts_3d[:, 2:]), axis=1)

        bbox_center = get_annot_center(joints_2d)
        bbox_scale = get_annot_scale(joints_2d)
        bbox_scale = bbox_scale * self.bbox_expand_ratio  # extend bbox sacle
        cam_center = np.array([cam_intr[0, 2], cam_intr[1, 2]])
        if self.with_mano:
            mano_pose = np.array(packed["mano_pose"])
            mano_shape = np.array(packed["mano_shape"])

        image_path = self.get_image_path(idx)
        image = self.get_image(idx)
        raw_size = [image.shape[1], image.shape[0]]  # (W, H)
        joints_vis = get_verts_2d_vis(verts_2d=joints_2d, raw_size=raw_size)
        if self.with_verts_2d:
            verts_vis = get_verts_2d_vis(verts_2d=verts_2d, raw_size=raw_size)

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
            if self.with_verts_2d:
                self.flip_2d(verts_2d, raw_size[0])

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
        if self.with_mano:
            label.update(
                {
                    "mano_pose": mano_pose,
                    "mano_shape": mano_shape,
                }
            )
        if self.with_verts_2d:
            label.update(
                {
                    "verts_2d": verts_2d,
                    "verts_vis": verts_vis,
                }
            )
        if self.with_hand_faces:
            label["hand_faces"] = self.rMANO.th_faces
        return image, label
