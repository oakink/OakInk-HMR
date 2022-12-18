import numpy as np
import torch
from torch.utils.data import default_collate
from oib.datasets.hdata import HDataset
from oib.utils.transform import (
    ObjectSimpleTransform3DMANO,
    ObjectSimpleTransformUVD,
    SimpleTransform3DMANO,
    SimpleTransformUVD,
    get_verts_2d_vis,
)


class HODataset(HDataset):
    def __init__(self, cfg):

        # ! If we do not include objects here, the dataset will look like a hand-only dataset
        self.include_obj = cfg.INCLUDE_OBJ
        self.include_sdf = cfg.INCLUDE_SDF

        self.filter_no_contact = cfg.FILTER_NO_CONTACT
        self.filter_thresh = cfg.FILTER_THRESH

        if self.include_sdf:
            self.sdf_n_sample = cfg.SDF_N_SAMPLE

        super().__init__(cfg)
        self.name = "HODataset"

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

        if self.data_mode == "3D":
            self.transform = SimpleTransform3DMANO(center_idx=self.center_idx, **basis_param)
            if self.include_obj:
                self.transform = [
                    self.transform,
                    ObjectSimpleTransform3DMANO(center_idx=self.center_idx, **basis_param),
                ]

        elif self.data_mode == "UVD":
            self.transform = SimpleTransformUVD(center_idx=self.center_idx, **basis_param)
            if self.include_obj:
                self.transform = [self.transform, ObjectSimpleTransformUVD(center_idx=self.center_idx, **basis_param)]
        else:
            raise NotImplementedError("data_mode {} not implemented".format(self.data_mode))

    def getitem_3d(self, idx):
        # * get hand data
        image, label = super().getitem_3d(idx)

        idx = label["idx"]
        # * objects

        if not self.include_obj:
            return image, label

        hand_side = self.get_sides(idx)
        flip_hand = True if hand_side != self.sides else False
        assert not flip_hand, "do not support flipping objects"

        obj_verts_3d = self.get_obj_verts_3d(idx)
        obj_verts_2d = self.get_obj_verts_2d(idx)
        obj_verts_can = self.get_obj_verts_can(idx)

        obj_verts_uvd = self.get_obj_verts_uvd(idx)

        obj_corners_3d = self.get_corners_3d(idx)
        obj_corners_2d = self.get_corners_2d(idx)
        obj_corners_can = self.get_corners_can(idx)

        obj_corners_uvd = self.get_corners_uvd(idx)

        raw_size = [image.shape[1], image.shape[0]]  # (W, H)
        corners_vis = get_verts_2d_vis(obj_corners_2d, raw_size=raw_size)

        obj_faces = self.get_obj_faces(idx)

        obj_transf = self.get_obj_transf(idx)

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

        if self.include_sdf:
            obj_sdf = self.get_obj_sdf(idx)
            label["obj_sdf"] = obj_sdf

        return image, label

    def getitem_uvd(self, idx):
        # * get hand data
        image, label = super().getitem_uvd(idx)

        idx = label["idx"]
        # * objects

        if not self.include_obj:
            return image, label

        hand_side = self.get_sides(idx)
        flip_hand = True if hand_side != self.sides else False
        assert not flip_hand, "do not support flipping objects"

        obj_verts_2d = self.get_obj_verts_2d(idx)
        obj_verts_can = self.get_obj_verts_can(idx)

        obj_verts_uvd = self.get_obj_verts_uvd(idx)

        obj_corners_2d = self.get_corners_2d(idx)
        obj_corners_can = self.get_corners_can(idx)

        obj_corners_uvd = self.get_corners_uvd(idx)

        raw_size = [image.shape[1], image.shape[0]]  # (W, H)
        corners_vis = get_verts_2d_vis(obj_corners_2d, raw_size=raw_size)

        obj_faces = self.get_obj_faces(idx)

        obj_transf = self.get_obj_transf(idx)

        label["obj_verts_2d"] = obj_verts_2d
        label["obj_verts_can"] = obj_verts_can
        label["obj_verts_uvd"] = obj_verts_uvd

        label["corners_2d"] = obj_corners_2d
        label["corners_can"] = obj_corners_can
        label["corners_uvd"] = obj_corners_uvd
        label["corners_vis"] = corners_vis

        label["obj_transf"] = obj_transf
        label["obj_faces"] = obj_faces

        return image, label

    def __getitem__(self, idx):
        if self.data_mode == "3D":
            image, label = self.getitem_3d(idx)
        elif self.data_mode == "UVD":
            image, label = self.getitem_uvd(idx)
        else:
            raise NotImplementedError("data_mode {} not implemented".format(self.data_mode))

        if type(self.transform) != list:
            self.transform = [self.transform]

        results = label
        for trans in self.transform:
            results = trans(image, results)
            results.update(label)
            results.pop("image_mask", None)
        return results

    def collate_fn(self, batch):

        if not self.include_obj:
            return default_collate(batch)

        def collate_field(batch, field):
            max_size = max([sample[field].shape[0] for sample in batch])
            for sample in batch:
                pop_value = sample[field]
                orig_len = pop_value.shape[0]
                # Repeat vertices so all have the same number
                pop_value = np.concatenate([pop_value] * int(max_size / pop_value.shape[0] + 1))[:max_size]
                sample[field] = pop_value

                padding_mask = np.zeros(max_size, dtype=np.float32)
                padding_mask[:orig_len] = 1
                sample[field + "_padding_mask"] = padding_mask

        collate_field(batch, "obj_verts_3d")
        collate_field(batch, "obj_verts_2d")
        collate_field(batch, "obj_verts_can")
        collate_field(batch, "obj_verts_uvd")

        collate_field(batch, "target_obj_verts_uvd")
        collate_field(batch, "target_obj_verts_3d")
        collate_field(batch, "target_obj_verts_3d_rel")

        collate_field(batch, "obj_faces")

        return default_collate(batch)
