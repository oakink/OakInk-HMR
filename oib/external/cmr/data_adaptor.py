import os

import cv2
import numpy as np
import torch
from termcolor import cprint
from yacs.config import CfgNode as CN
from oib.datasets.freihand import FreiHAND_v2_Extra
from oib.datasets.texturedmano import TexturedMano
from oib.utils.builder import DATASET
from oib.utils.logger import logger


def uv2map(uv, size=(224, 224)):
    kernel_size = (size[0] * 13 // size[0] - 1) // 2
    gaussian_map = np.zeros((uv.shape[0], size[0], size[1]))
    size_transpose = np.array(size)
    gaussian_kernel = cv2.getGaussianKernel(2 * kernel_size + 1, (2 * kernel_size + 2) / 4.)
    gaussian_kernel = np.dot(gaussian_kernel, gaussian_kernel.T)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.max()

    for i in range(gaussian_map.shape[0]):
        if (uv[i] >= 0).prod() == 1 and (uv[i][1] <= size_transpose[0]) and (uv[i][0] <= size_transpose[1]):
            s_pt = np.array((uv[i][1], uv[i][0]))
            p_start = s_pt - kernel_size
            p_end = s_pt + kernel_size
            p_start_fix = (p_start >= 0) * p_start + (p_start < 0) * 0
            k_start_fix = (p_start >= 0) * 0 + (p_start < 0) * (-p_start)
            p_end_fix = (p_end <= (size_transpose - 1)) * p_end + (p_end > (size_transpose - 1)) * (size_transpose - 1)
            k_end_fix = (p_end <= (size_transpose - 1)) * kernel_size * 2 + \
                (p_end > (size_transpose - 1)) * (2*kernel_size - (p_end - (size_transpose - 1)))

            gaussian_map[i, p_start_fix[0]: p_end_fix[0] + 1, p_start_fix[1]: p_end_fix[1] + 1] = \
                gaussian_kernel[k_start_fix[0]: k_end_fix[0] + 1, k_start_fix[1]: k_end_fix[1] + 1]

    return gaussian_map


class CMRDataConverter(object):

    def __init__(self, center_idx, is_train=True):

        from .net import Pool
        self.Pool = Pool
        self.center_idx = center_idx
        self.V_STD = 0.2
        self.is_train = is_train

        self.has_spiral_transform = False
        self.spiral_indices_list = []
        self.down_sample_list = []
        self.up_transform_list = []
        self.faces = []

    def convert(self, inputs):
        """
        Convert the data to the format that the CMR can accept.
        """
        if not self.has_spiral_transform:
            # The function: spiral_transform() must be called at runtime to get the down_transform_list.
            # Otherwise, (if you are using torch DDP) you will get ``RuntimeError: sparse tensors do not have storage''
            # This is because sparse tensors on stable pytorch do not support pickling.
            # One workaround at least until pytorch 1.8.2 LTS is simply to construct to SparseTensor within
            # your getitem function rather than to scope it in. For example:
            from .utils import spiral_tramsform
            work_dir = os.path.dirname(os.path.realpath(__file__))
            transf_pth = os.path.join(work_dir, 'template', 'transform.pkl')
            template_pth = os.path.join(work_dir, 'template', 'template.ply')
            spiral_indices_list, down_sample_list, up_transform_list, tmp = spiral_tramsform(transf_pth, template_pth)

            self.spiral_indices_list = spiral_indices_list
            self.down_sample_list = down_sample_list
            self.up_transform_list = up_transform_list
            self.faces = tmp['face']
            self.has_spiral_transform = True

        img = inputs["image"]  # (C, H, W)
        mask = inputs["mask"]  # (C, H, W)
        v0 = inputs["target_verts_3d"][:778]  # ! (778, 3) ignore wrist

        # K = inputs["target_cam_intr"]  # (3, 3)
        xyz = inputs["target_joints_3d"]  # (21, 3)
        uv = inputs["target_joints_2d"]  # (21, 2)

        uv_map = uv2map(uv.astype(np.int), img.shape[1:]).astype(np.float32)  # (21, H, W)
        uv_map = cv2.resize(uv_map.transpose(1, 2, 0), (img.shape[2] // 2, img.shape[1] // 2)).transpose(2, 0, 1)
        mask = cv2.resize(mask, (img.shape[2] // 2, img.shape[1] // 2))

        xyz_root = xyz[self.center_idx]  # (3)
        v0 = (v0 - xyz_root) / self.V_STD
        xyz = (xyz - xyz_root) / self.V_STD

        v0 = torch.from_numpy(v0).float()
        if self.is_train:
            v1 = self.Pool(v0.unsqueeze(0), self.down_sample_list[0])[0]
            v2 = self.Pool(v1.unsqueeze(0), self.down_sample_list[1])[0]
            v3 = self.Pool(v2.unsqueeze(0), self.down_sample_list[2])[0]
            gt = [v0, v1, v2, v3]
        else:
            gt = [v0]

        data = {
            'img': img,
            'mesh_gt': gt,
            # 'K': K,
            'mask_gt': mask,
            'xyz_gt': xyz,
            'uv_point': uv,
            'uv_gt': uv_map,
            'xyz_root': xyz_root
        }

        return data


@DATASET.register_module
class FreiHAND_CMR(FreiHAND_v2_Extra):
    """
    FreiHAND dataset with CMR adaptation
    """

    def __init__(self, cfg: CN):
        super(FreiHAND_CMR, self).__init__(cfg)
        is_train = ("train" in self.data_split)
        self.CMR_DC = CMRDataConverter(self.center_idx, is_train)
        logger.warning(f"Initialized child class: FreiHAND_CMR (FreiHAND_v2_Extra)")

    def getitem_3d(self, index):
        """
        Get 3D data from the dataset
        """
        res = super().getitem_3d(index)
        res = self.CMR_DC.convert(res)
        return res

    def getitem_uvd_ortho(self, idx):
        res = super().getitem_uvd_ortho(idx)
        res = self.CMR_DC.convert(res)
        return res


@DATASET.register_module
class TexturedMano_CMR(TexturedMano):
    """
    TexturedMano dataset with CMR adaptation
    """

    def __init__(self, cfg: CN):
        super(TexturedMano_CMR, self).__init__(cfg)
        is_train = ("train" in self.data_split)
        self.CMR_DC = CMRDataConverter(self.center_idx, is_train)
        logger.warning(f"Initialized child class: TexturedMano_CMR (TexturedMano)")

    def getitem_uvd_ortho(self, idx):
        res = super().getitem_uvd_ortho(idx)
        res = self.CMR_DC.convert(res)
        return res
