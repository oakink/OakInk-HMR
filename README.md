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

### Option 1. Create `oakink-bm-img` conda env (recommended)

First, create a new conda env and install the required packages.

```bash
conda env create -f environment.yml
conda activate oakink-bm-img
pip install -r requirements.txt
```

Then, download our [**oikit**](https://github.com/oakink/OakInk), and install it in the `oakink-bm-img` conda env (follow the instruction: [**_import-as-package_**](https://github.com/oakink/OakInk#import-as-package-recommended)).

### Option 2. Install extra packages in existing `oakink` conda env

If a **_stand-alone_** `oakink` env is already installed, you need to install the extra packages required for image benchmark.

```bash
conda activate oakink

# comment out `manotorch` and `pytorch3d` in requirements.txt, then run:
pip install -r requirements.txt
```

Also, install our **oikit** in the existing `oakink` conda env (follow the instruction: **_import-as-package_**).

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

- Download the `postprocess.zip` at [here](https://www.dropbox.com/s/gowg1zelicon4f5/postprocess.zip?dl=0), unzip and put it under the `assets` directory;
- Download the `mano_v1_2.zip` from the [MANO website](https://mano.is.tue.mpg.de), unzip the file and create symlink in the `assets` directory;

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

We provide a script for packing annotations into a single archive for each sample. Run following script:

```bash
# mode_split and data_split has following options:

# mode_split:
#    "default"            SP0, view split, one view per sequence as test;
#    "subject"            SP1, subject split, subjects recorded in the test will not appear in the train split;
#    "object"             SP2, objects split, objects recorded in the test will not appear in the train split;
#    ------------
#    "handobject"         view split, similar to SP0, but filter out frames that the min distance between hand and object is greater than 5 mm;

# data_split:
#    all, train+val, test, train, val

python dev/pack_oakink_image.py --mode_split default --data_split train+val
```

## Model Zoo

### Hand Mesh Recovery

| method       | IKNet              | split | MPJPE | PCK AUC | MPVPE | PA-MPJPE | PA-MPVPE | model                                                                                                  | config                                            |
| ------------ | ------------------ | ----- | ----- | ------- | ----- | -------- | -------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------- |
| IntegralPose | :white_check_mark: | SP0   | 8.14  | 0.838   | 8.75  | 5.22     | 5.60     | [zip](https://www.dropbox.com/s/ogm64dfk4qpnyiu/oakink_h_sp0_integral_pose_2022_1020_1542_45.zip?dl=0) | [cfg](config/oii/train_integral_pose_oii_sp0.yml) |
| IntegralPose | :white_check_mark: | SP1   | 10.88 | 0.784   | 11.32 | 6.76     | 6.81     | [zip](https://www.dropbox.com/s/g3hl2ao56fku2tx/oakink_h_sp1_integral_pose_2022_1022_1348_40.zip?dl=0) | [cfg](config/oii/train_integral_pose_oii_sp1.yml) |
| IntegralPose | :white_check_mark: | SP2   | 8.22  | 0.837   | 8.83  | 5.30     | 5.66     | [zip](https://www.dropbox.com/s/u51eamb0vnp8t3m/oakink_h_sp2_integral_pose_2022_1024_1608_34.zip?dl=0) | [cfg](config/oii/train_integral_pose_oii_sp2.yml) |
| RLE          | :white_check_mark: | SP0   | 9.45  | 0.815   | 9.92  | 5.14     | 5.63     | [zip](https://www.dropbox.com/s/cbs73iuqjk27npz/oakink_h_sp0_res_loglike_2022_1031_2130_36.zip?dl=0)   | [cfg](config/oii/train_res_loglike_oii_sp0.yml)   |
| RLE          | :white_check_mark: | SP1   | 13.22 | 0.739   | 13.34 | 6.60     | 6.79     | [zip](https://www.dropbox.com/s/gm71caolzgu079u/oakink_h_sp1_res_loglike_2022_1102_1151_05.zip?dl=0)   | [cfg](config/oii/train_res_loglike_oii_sp1.yml)   |
| RLE          | :white_check_mark: | SP2   | 9.76  | 0.810   | 10.21 | 5.31     | 5.76     | [zip](https://www.dropbox.com/s/3n50fyd1wimk30n/oakink_h_sp2_res_loglike_2022_1103_1228_03.zip?dl=0)   | [cfg](config/oii/train_res_loglike_oii_sp2.yml)   |
| I2L-MeshNet  | --                 | SP0   | 9.24  | 0.818   | 9.05  | 5.03     | 5.03     | [zip](https://www.dropbox.com/s/cbah6mq1n8d50sy/oakink_h_sp0_i2l_meshnet_2022_1117_0527_10.zip?dl=0)   | [cfg](config/oii/train_i2l_meshnet_oii_sp0.yml)   |
| I2L-MeshNet  | --                 | SP1   | 12.92 | 0.745   | 12.73 | 6.54     | 6.46     | [zip](https://www.dropbox.com/s/tyv9k7390vxde76/oakink_h_sp1_i2l_meshnet_2022_1215_1647_04.zip?dl=0)   | [cfg](config/oii/train_i2l_meshnet_oii_sp1.yml)   |
| I2L-MeshNet  | --                 | SP2   | 9.29  | 0.818   | 9.23  | 5.12     | 5.20     | [zip](https://www.dropbox.com/s/ae7sl9xqgk4o8j6/oakink_h_sp2_i2l_meshnet_2022_1121_0414_05.zip?dl=0)   | [cfg](config/oii/train_i2l_meshnet_oii_sp2.yml)   |

## Evaluation

:bulb: Download the corresponding checkpoint from the [Model Zoo](#model-zoo), and put it under the `exp` directory.

### Integral Pose [(link)](https://github.com/JimmySuen/integral-human-pose) + IKNet

heatmap-based methods, output 21 keypoints, we use an extra IKNet trained in [HandTailor](https://github.com/LyuJ1998/HandTailor) for generating mesh.  
The results will be saved at `./exp/{exp_id}_{localtime}/`.

```bash
## SP0
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp0_integral_pose_2022_1020_1542_45/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp0_integral_pose_2022_1020_1542_45/checkpoints/checkpoint_100/IntegralPose.pth.tar \
    --gpu_id 1 \
    --exp_id eval_oakink_h_sp0_integral_pose

## SP1
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp1_integral_pose_2022_1022_1348_40/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp1_integral_pose_2022_1022_1348_40/checkpoints/checkpoint_100/IntegralPose.pth.tar \
    --gpu_id 1 \
    --exp_id eval_oakink_h_sp1_integral_pose

## SP2
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp2_integral_pose_2022_1024_1608_34/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp2_integral_pose_2022_1024_1608_34/checkpoints/checkpoint_100/IntegralPose.pth.tar \
    --gpu_id 1 \
    --exp_id eval_oakink_h_sp2_integral_pose
```

### Res-Loglikelihood-Estimation [(link)](https://github.com/Jeff-sjtu/res-loglikelihood-regression) + HandTailor

regression-based method, output 21 keypoints, we use an extra IKNet trained in [HandTailor](https://github.com/LyuJ1998/HandTailor) for generating mesh.

```bash
## SP0
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp0_res_loglike_2022_1031_2130_36/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp0_res_loglike_2022_1031_2130_36/checkpoints/checkpoint_100/RegressFlow3D.pth.tar \
    --gpu_id 1 \
    --exp_id eval_oakink_h_sp0_res_loglike

## SP1
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp1_res_loglike_2022_1102_1151_05/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp1_res_loglike_2022_1102_1151_05/checkpoints/checkpoint_100/RegressFlow3D.pth.tar \
    --gpu_id 1 \
    --exp_id eval_oakink_h_sp1_res_loglike

## SP2
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp2_res_loglike_2022_1103_1228_03/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp2_res_loglike_2022_1103_1228_03/checkpoints/checkpoint_100/RegressFlow3D.pth.tar \
    --gpu_id 1 \
    --exp_id eval_oakink_h_sp2_res_loglike
```

### I2L-MeshNet [(link)](https://github.com/mks0601/I2L-MeshNet_RELEASE)

```bash
## SP0
python -m train.test_model_hand_tailor \
    --cfg exp/oakink_h_sp0_i2l_meshnet_2022_1117_0527_10/dump_cfg.yaml \
    --reload exp/oakink_h_sp0_i2l_meshnet_2022_1117_0527_10/checkpoints/checkpoint_100/I2L_MeshNet.pth.tar \
    --gpu_id 1 \
    --workers 32 \
    --batch_size 64 \
    --exp_id eval_oakink_h_sp0_i2l_meshnet

## SP1
python -m train.test_model_hand_tailor \
    --cfg exp/oakink_h_sp1_i2l_meshnet_2022_1215_1647_04/dump_cfg.yaml \
    --reload exp/oakink_h_sp1_i2l_meshnet_2022_1215_1647_04/checkpoints/checkpoint_100/I2L_MeshNet.pth.tar \
    --gpu_id 1 \
    --workers 32 \
    --batch_size 64 \
    --exp_id eval_oakink_h_sp1_i2l_meshnet

## SP2
python -m train.test_model_hand_tailor \
    --cfg exp/oakink_h_sp2_i2l_meshnet_2022_1121_0414_05/dump_cfg.yaml \
    --reload exp/oakink_h_sp2_i2l_meshnet_2022_1121_0414_05/checkpoints/checkpoint_100/I2L_MeshNet.pth.tar \
    --gpu_id 1 \
    --workers 32 \
    --batch_size 64 \
    --exp_id eval_oakink_h_sp2_i2l_meshnet
```

## Training

### Integral Pose

```bash
python -m train.train_model --cfg config/oii/train_integral_pose_oii_sp0.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp0_integral_pose
python -m train.train_model --cfg config/oii/train_integral_pose_oii_sp1.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp1_integral_pose
python -m train.train_model --cfg config/oii/train_integral_pose_oii_sp2.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp2_integral_pose
```

### Res-LogLikelihood-Estimation

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

## Citation

:+1: Please consider citing the OakInk dataset and these awesome Pose Estimation approaches

<details><summary>OakInk, Integral-Pose, RLE, I2L-MeshNet</summary>

```

@inproceedings{Yang2022OakInk,
  title={{OakInk}: A Large-scale Knowledge Repository for Understanding Hand-Object Interaction},
  author={Lixin Yang and Kailin Li and Xinyu Zhan and Fei Wu and Anran Xu and Liu Liu and Cewu Lu},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022},
}

@inproceedings{Sun2017IntegralHP,
  title={Integral Human Pose Regression},
  author={Xiao Sun and Bin Xiao and Shuang Liang and Yichen Wei},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2017}
}

@inproceedings{Li2021HumanPR,
  title={Human Pose Regression with Residual Log-likelihood Estimation},
  author={Jiefeng Li and Siyuan Bian and Ailing Zeng and Can Wang and Bo Pang and Wentao Liu and Cewu Lu},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021},
}

@inproceedings{Moon2020I2LMeshNet,
  title = {{I2L-MeshNet}: Image-to-Lixel Prediction Network for Accurate 3D Human Pose and Mesh Estimation from a Single RGB Image},
  author = {Moon, Gyeongsik and Lee, Kyoung Mu},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2020}
}

```

</details>
