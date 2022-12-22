  <p align="center">
    <img src="imgs/oakink_logo.png"" alt="Logo" width="20%">
  </p>

<h1 align="center"> Image Benchmark</h1>

  <p align="center">
    <strong>CVPR, 2022</strong>
    <br />
    <a href="https://lixiny.github.io"><strong>Lixin Yang*</strong></a>
    ·
    <a href="https://kailinli.top"><strong>Kailin Li*</strong></a>
    ·
    <a href=""><strong>Xinyu Zhan*</strong></a>
    ·
    <a href=""><strong>Fei Wu</strong></a>
    ·
    <a href="https://anran-xu.github.io"><strong>Anran Xu</strong></a>
    .
    <a href="https://liuliu66.github.io"><strong>Liu Liu</strong></a>
    ·
    <a href="https://mvig.sjtu.edu.cn"><strong>Cewu Lu</strong></a>
    <br />
    \star = equal contribution
  </p>

  <p align="center">
  <a href='https://openaccess.thecvf.com/content/CVPR2022/html/Yang_OakInk_A_Large-Scale_Knowledge_Repository_for_Understanding_Hand-Object_Interaction_CVPR_2022_paper.html'>
      <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=googlescholar&logoColor=blue' alt='Paper PDF'>
    </a>
    <a href='https://arxiv.org/abs/2203.15709' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/ArXiv-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='ArXiv PDF'>
    </a>
    <a href='https://oakink.net' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    <a href='https://www.youtube.com/watch?v=vNTdeXlLdU8' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Youtube-Video-red?style=flat&logo=youtube&logoColor=red' alt='Youtube Video'>
    </a>
  </p>

This repo contains the official **_benchmark results_** on OakInk-Image dataset.

:warning: This benchmark is based on the **public v2.1** release (Oct,18,2022) of the OakInk-Image dataset.  
To download the dataset, please visit the [official website](https://oakink.net).

## Installation

### Option 1. Create `oakink-bm-img` conda env

```shell
conda env create -f environment.yml
conda activate oakink-bm-img
pip install -r requirements.txt
```

### Option 2. install extra packages in existing `oakink` conda env

If the [`oakink`](https://github.com/oakink/OakInk) env is already installed, you just need to install the extra packages required for image benchmark.

1. comment out `manotorch` and `pytorch3d` in the `requirements.txt` of this repository.
2. `pip install -r requirements.txt`

## Prepare

### OakInk-Image data

Link the `OakInk/image` subset in the `data` directory

```
data
└── OakInk
    └── image
        ├── anno
        ├── obj
        └── stream_release_v2
```

### Assets

- Download the [`postprocess`](), put it under the `assets` directory;
- Download `mano_v1_2.zip` from the [MANO website](https://mano.is.tue.mpg.de), unzip the file and create symlink in the `assets` directory;

```
assets
├── mano_v1_2
│   ├── __init__.py
│   ├── LICENSE.txt
│   ├── models
│   ├── __pycache__
│   └── webuser
└── postprocess
    ├── hand_close.npy
    ├── hand.npy
    └── iknet.pt
```

### Pack annotation

To avoid opening too many small files in pytorch dataloaders which might exceed user limit, we provide a script for packing annotations into a single archive for each sample.

1. Follow instruction: **_import-as-package_** in [OakInk toolkit](https://github.com/oakink/OakInk), install the **oikit** package in current `oakink-bm-img` conda env.
2. Run following script.

   ```bash
   python dev/pack_oakink_image.py --mode_split default --data_split train+val
   ```

   :warning: **mode_split** and **data_split** has following options, based on the experiment settings in the paper.

   - **mode_split**: default, object, subject, handobject
   - **data_split**: all, train+val, test, train, val

## Model Zoo

### Hand Mesh Recovery

| Model        | IKNet              | Split | MPJPE | PCK AUC | MPVPE | PA-MPJPE | PA-MPVPE | Checkpoint |
| ------------ | ------------------ | ----- | ----- | ------- | ----- | -------- | -------- | ---------- |
| IntegralPose | :white_check_mark: | SP0   | 8.14  | 0.838   | 8.75  | 5.22     | 5.60     |            |
| IntegralPose | :white_check_mark: | SP1   | 10.88 | 0.784   | 11.32 | 6.76     | 6.81     |            |
| IntegralPose | :white_check_mark: | SP2   | 8.22  | 0.837   | 8.83  | 5.30     | 5.66     |            |
| RLE          | :white_check_mark: | SP0   | 9.45  | 0.815   | 9.92  | 5.14     | 5.63     |            |
| RLE          | :white_check_mark: | SP1   | 13.22 | 0.739   | 13.34 | 6.60     | 6.79     |            |
| RLE          | :white_check_mark: | SP2   | 9.76  | 0.810   | 10.21 | 5.31     | 5.76     |            |
| I2L-Meshnet  | --                 | SP0   | 9.24  | 0.818   | 9.05  | 5.03     | 5.03     |            |
| I2L-Meshnet  | --                 | SP1   | 12.92 | 0.745   | 12.73 | 6.54     | 6.46     |            |
| I2L-Meshnet  | --                 | SP2   | 9.29  | 0.818   | 9.23  | 5.12     | 5.20     |            |

## Evaluation

### Integral Pose + HandTailor

heatmap-based methods, output 21 keypoints, we use an extra IKNet trained in [HandTailor]() for generating mesh.

```bash
## SP0
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp0_integral_pose_2022_1020_1542_45/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp0_integral_pose_2022_1020_1542_45/checkpoints/checkpoint_100/IntegralPose.pth.tar \
    --gpu_id 1 \
    --exp_id eval_oakink_h_sp0_integral_pose_2022_1020_1542_45

## SP1
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp1_integral_pose_2022_1022_1348_40/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp1_integral_pose_2022_1022_1348_40/checkpoints/checkpoint_100/IntegralPose.pth.tar \
    --gpu_id 1 \
    --exp_id eval_oakink_h_sp1_integral_pose_2022_1022_1348_40

## SP2
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp2_integral_pose_2022_1024_1608_34/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp2_integral_pose_2022_1024_1608_34/checkpoints/checkpoint_100/IntegralPose.pth.tar \
    --gpu_id 1 \
    --exp_id eval_oakink_h_sp2_integral_pose_2022_1024_1608_34
```

### Res-Log-likelihood + HandTailor

regression-based method, output 21 keypoints, we use an extra IKNet trained in [HandTailor]() for generating mesh.

```bash
## SP0
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp0_res_loglike_2022_1031_2130_36/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp0_res_loglike_2022_1031_2130_36/checkpoints/checkpoint_100/RegressFlow3D.pth.tar \
    --gpu_id 1 \
    --exp_id eval_oakink_h_sp0_res_loglike_2022_1031_2130_36

## SP1
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp1_res_loglike_2022_1102_1151_05/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp1_res_loglike_2022_1102_1151_05/checkpoints/checkpoint_100/RegressFlow3D.pth.tar \
    --gpu_id 1 \
    --exp_id eval_oakink_h_sp1_res_loglike_2022_1102_1151_05

## SP2
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp2_res_loglike_2022_1103_1228_03/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp2_res_loglike_2022_1103_1228_03/checkpoints/checkpoint_100/RegressFlow3D.pth.tar \
    --gpu_id 1 \
    --exp_id eval_oakink_h_sp2_res_loglike_2022_1103_1228_03
```

### e.3 I2L-MeshNet

```bash
## SP0
python -m train.test_model_hand_tailor \
    --cfg exp/oakink_h_sp0_i2l_meshnet_2022_1117_0527_10/dump_cfg.yaml \
    --reload exp/oakink_h_sp0_i2l_meshnet_2022_1117_0527_10/checkpoints/checkpoint_100/I2L_MeshNet.pth.tar \
    --gpu_id 1 \
    --workers 32 \
    --batch_size 64 \
    --exp_id eval_oakink_h_sp0_i2l_meshnet_2022_1117_0527_10

## SP1
python -m train.test_model_hand_tailor \
    --cfg exp/oakink_h_sp1_i2l_meshnet_2022_1215_1647_04/dump_cfg.yaml \
    --reload exp/oakink_h_sp1_i2l_meshnet_2022_1215_1647_04/checkpoints/checkpoint_100/I2L_MeshNet.pth.tar \
    --gpu_id 1 \
    --workers 32 \
    --batch_size 64 \
    --exp_id eval_oakink_h_sp1_i2l_meshnet_2022_1215_1647_04

## SP2
python -m train.test_model_hand_tailor \
    --cfg exp/oakink_h_sp2_i2l_meshnet_2022_1121_0414_05/dump_cfg.yaml \
    --reload exp/oakink_h_sp2_i2l_meshnet_2022_1121_0414_05/checkpoints/checkpoint_100/I2L_MeshNet.pth.tar \
    --gpu_id 1 \
    --workers 32 \
    --batch_size 64 \
    --exp_id eval_oakink_h_sp2_i2l_meshnet_2022_1121_0414_05
```

## Training

### Integral Pose

```bash
python -m train.train_model --cfg config/oii/train_integral_pose_oii_sp0.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp0_integral_pose
python -m train.train_model --cfg config/oii/train_integral_pose_oii_sp1.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp1_integral_pose
python -m train.train_model --cfg config/oii/train_integral_pose_oii_sp2.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp2_integral_pose
```

### Res-Log-likelihood

```bash
python -m train.train_model --cfg config/oii/train_res_loglike_oii_sp0.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp0_res_loglike
python -m train.train_model --cfg config/oii/train_res_loglike_oii_sp1.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp1_res_loglike
python -m train.train_model --cfg config/oii/train_res_loglike_oii_sp2.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp2_res_loglike
```

### I2L-MeshNet

```bash
python -m train.train_model --cfg config/oii/train_i2l_meshnet_oii_sp0.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp0_i2l_meshnet
python -m train.train_model --cfg config/oii/train_i2l_meshnet_oii_sp1.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp1_i2l_meshnet
python -m train.train_model --cfg config/oii/train_i2l_meshnet_oii_sp2.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp2_i2l_meshnet
```
