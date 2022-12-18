# ! WARNING: unfinished & untested

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import List
import trimesh
import logging

import imageio
import numpy as np
import torch
from manotorch.manolayer import ManoLayer
from oib.datasets.hodata import HODataset
from oib.utils.builder import DATASET
from oib.utils.etqdm import etqdm
from oib.utils.logger import logger
from oib.utils.transform import get_annot_center, get_annot_scale, persp_project
from termcolor import colored
from oib.utils.transform import (
    SimpleTransform3D,
    SimpleTransformUVD,
    ObjectSimpleTransform3DMANO,
    ObjectSimpleTransformUVD,
    get_verts_2d_vis,
)


def suppress_trimesh_logging():
    logger = logging.getLogger("trimesh")
    logger.setLevel(logging.ERROR)


def load_object(obj_root, filename):
    # load object mesh
    try:
        mesh_file = os.path.join(obj_root, filename)
        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f"Cannot found valid object mesh file at {obj_root} for {filename}")
        obj = trimesh.load(mesh_file, process=False, skip_materials=True, force="mesh")
        bbox_center = (obj.vertices.min(0) + obj.vertices.max(0)) / 2
        obj.vertices = obj.vertices - bbox_center
    except Exception as e:
        raise RuntimeError(f"failed to load object {filename}! {e}")
    return obj


def decode_seq_cat(seq_cat):
    field_list = seq_cat.split("_")
    obj_id = field_list[0]
    action_id = field_list[1]
    if action_id == "0004":
        subject_id = tuple(field_list[2:4])
    else:
        subject_id = (field_list[2],)
    return obj_id, action_id, subject_id


@DATASET.register_module
class OakInkHO(HODataset):
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
        assert self.data_mode in ["UVD", "3D"], f"OakInk does not dupport {self.data_mode} mode"
        assert self.data_split in [
            "all",
            "train+val",
            "train",
            "val",
            "test",
        ], "OakInk data_split must be one of ['train', 'val']"
        self.split_mode = cfg.SPLIT_MODE
        assert self.split_mode in [
            "handobject",
        ], "OakInk split_mode must be one of ['default']"
        self.use_pack = cfg.get("USE_PACK", False)
        if self.use_pack:
            self.getitem_3d = self._getitem_3d_pack

        self.load_dataset()
        logger.info(f"initialized child class: {self.name}")

    def load_dataset(self):
        self.name = "oakink-image"
        self.root = os.path.join(self.data_root, self.name)

        if self.data_split == "all":
            self.info_list = json.load(open(os.path.join(self.root, "anno", "seq_all.json")))
        else:  # self.split_mode == "handobject":
            self.info_list = self._get_info_list(self.root, "split0_ho", self.data_split)

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

        # load obj
        suppress_trimesh_logging()

        self.obj_mapping = {}
        obj_root = os.path.join(self.root, "obj")
        all_obj_fn = sorted(os.listdir(obj_root))
        for obj_fn in all_obj_fn:
            obj_id = os.path.splitext(obj_fn)[0]
            obj_model = load_object(obj_root, obj_fn)
            self.obj_mapping[obj_id] = obj_model

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

        if self.data_mode == "UVD":
            self.transform = SimpleTransformUVD(center_idx=self.center_idx, **basis_param)
            if self.include_obj:
                self.transform = [
                    self.transform,
                    ObjectSimpleTransformUVD(center_idx=self.center_idx, **basis_param),
                ]
        elif self.data_mode == "3D":
            self.transform = SimpleTransform3D(center_idx=self.center_idx, **basis_param)
            if self.include_obj:
                self.transform = [
                    self.transform,
                    ObjectSimpleTransform3DMANO(center_idx=self.center_idx, **basis_param),
                ]
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
        raise NotImplementedError(f"{self.name} does not support mano pose")

    def get_mano_shape(self, idx):
        raise NotImplementedError(f"{self.name} does not support mano shape")

    def get_obj_idx(self, idx):
        info = self.info_list[idx][0]
        seq_cat, _ = info.split("/")
        obj_id, _, _ = decode_seq_cat(seq_cat)
        return obj_id

    def get_obj_faces(self, idx):
        obj_id = self.get_obj_idx(idx)
        return np.asarray(self.obj_mapping[obj_id].faces).astype(np.int32)

    def get_obj_transf(self, idx):
        obj_transf_path = os.path.join(self.root, "anno", "obj_transf", f"{self.info_str_list[idx]}.pkl")
        with open(obj_transf_path, "rb") as f:
            obj_transf = pickle.load(f)
        return obj_transf.astype(np.float32)

    def get_obj_verts_3d(self, idx):
        obj_verts = self.get_obj_verts_can(idx)
        obj_transf = self.get_obj_transf(idx)
        obj_rot = obj_transf[:3, :3]
        obj_tsl = obj_transf[:3, 3]
        obj_verts_transf = (obj_rot @ obj_verts.transpose(1, 0)).transpose(1, 0) + obj_tsl
        return obj_verts_transf

    def get_obj_verts_2d(self, idx):
        obj_verts_3d = self.get_obj_verts_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return persp_project(obj_verts_3d, cam_intr)

    def get_obj_verts_can(self, idx):
        obj_id = self.get_obj_idx(idx)
        obj_verts = np.asarray(self.obj_mapping[obj_id].vertices).astype(np.float32)
        return obj_verts

    def get_corners_3d(self, idx):
        obj_corners = self.get_corners_can(idx)
        obj_transf = self.get_obj_transf(idx)
        obj_rot = obj_transf[:3, :3]
        obj_tsl = obj_transf[:3, 3]
        obj_corners_transf = (obj_rot @ obj_corners.transpose(1, 0)).transpose(1, 0) + obj_tsl
        return obj_corners_transf

    def get_corners_2d(self, idx):
        obj_corners = self.get_corners_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return persp_project(obj_corners, cam_intr)

    def get_corners_can(self, idx):
        obj_id = self.get_obj_idx(idx)
        obj_mesh = self.obj_mapping[obj_id]
        obj_corners = trimesh.bounds.corners(obj_mesh.bounds)
        return obj_corners

    def get_sample_identifier(self, idx):
        res = f"{self.name}__{self.info_str_list[idx]}"
        return res

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

        # for hand mode, early exit
        if not self.include_obj:
            return image, label

        obj_verts_can = self.get_obj_verts_can(idx)
        obj_transf = np.array(packed["obj_transf"])
        obj_rot = obj_transf[:3, :3]
        obj_tsl = obj_transf[:3, 3]
        obj_verts_3d = (obj_rot @ obj_verts_can.transpose(1, 0)).transpose(1, 0) + obj_tsl
        obj_verts_2d = persp_project(obj_verts_3d, cam_intr)
        obj_verts_uvd = np.concatenate((obj_verts_2d, obj_verts_3d[:, 2:]), axis=1)

        obj_corners_can = self.get_corners_can(idx)
        obj_corners_3d = (obj_rot @ obj_corners_can.transpose(1, 0)).transpose(1, 0) + obj_tsl
        obj_corners_2d = persp_project(obj_corners_3d, cam_intr)
        obj_corners_uvd = np.concatenate((obj_corners_2d, obj_corners_3d[:, 2:]), axis=1)

        corners_vis = get_verts_2d_vis(obj_corners_2d, raw_size=raw_size)
        obj_faces = self.get_obj_faces(idx)

        label["obj_verts_3d"] = obj_verts_3d
        label["obj_verts_2d"] = obj_verts_2d
        label["obj_verts_can"] = obj_verts_can
        label["obj_verts_uvd"] = obj_verts_uvd

        label["corners_3d"] = obj_corners_3d
        label["corners_2d"] = obj_corners_2d
        label["corners_can"] = obj_corners_can
        label["corners_uvd"] = obj_corners_uvd
        label["corners_vis"] = corners_vis

        label["obj_transf"] = obj_transf
        label["obj_faces"] = obj_faces

        return image, label
