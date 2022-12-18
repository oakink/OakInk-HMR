import numpy as np
import torch
from termcolor import colored
from yacs.config import CfgNode as CN

from oib.utils.builder import build_dataset
from oib.utils.logger import logger


class MixDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_cfg_list, preset_cfg, max_len):
        self.datasets = [build_dataset(cfg, preset_cfg) for cfg in dataset_cfg_list]
        logger.warning(f"MixDataset initialized Done! " f"Including {len(self.datasets)} datasets")

        if max_len is None:
            self.length = sum([len(d) for d in self.datasets])
            logger.warning(f"MixDataset uses all datasets")
            self.use_all = True
        else:
            self.ratios = np.array([cfg.MIX_RATIO for cfg in dataset_cfg_list])
            self.partitions = self.ratios.cumsum()

            assert self.partitions[-1] == 1.0, "Mixing ratios must sum to 1.0"

            self.length = max([len(d) for d in self.datasets])
            if max_len != -1:
                self.length = min(self.length, max_len)
            logger.warning(f"MixDataset has {self.length} working length")
            self.use_all = False

        # make sure all the datasets have the same inherent collate_fn
        assert [type(d).collate_fn for d in self.datasets[1:]] == [type(d).collate_fn for d in self.datasets[:-1]]

        # info = colored(" + ", 'blue', attrs=['bold']).join([f"{self.ratios[i]} * {self.datasets[i].name}" for i in range(self.N_DS)])
        # logger.info(f"MixDataset: {info}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Index an element from the dataset.
        This is done by randomly choosing a dataset using the mixing percentages
        and then randomly choosing from the selected dataset.
        Returns:
            Dict: Dictionary containing data and labels for the selected example
        """
        if self.use_all:
            base_length = 0
            for d in self.datasets:
                if idx < base_length + len(d):
                    return d[idx - base_length]
                base_length += len(d)
        else:
            p = np.random.rand()
            for i in range(len(self.datasets)):  # N datasets
                if p <= self.partitions[i]:
                    p = np.random.randint(len(self.datasets[i]))
                    return self.datasets[i][p]

    def collate_fn(self, batch):

        return self.datasets[0].collate_fn(batch)


if __name__ == "__main__":
    import time

    import imageio

    from oib.datasets import create_dataset
    from oib.utils.config import get_config
    from oib.utils.transform import denormalize
    from oib.viztools.draw import plot_image_heatmap_mask

    cfg = get_config("config/train_bihand2d_mix_pl.yml", arg=None, merge=True)
    dataset = create_dataset(cfg.DATASET.TRAIN, cfg.DATA_PRESET)

    for i in range(len(dataset)):
        output = dataset[i]
        image = denormalize(output["image"], [0.5, 0.5, 0.5], [1, 1, 1]).numpy().transpose(1, 2, 0)
        image = (image * 255.0).astype(np.uint8)
        mask = (output["mask"] * 255.0).astype(np.uint8)
        joints_heatmap = output["target_joints_heatmap"]
        comb_img = plot_image_heatmap_mask(image, joints_heatmap, mask)

        imageio.imwrite("img.png", comb_img)
        time.sleep(2)
