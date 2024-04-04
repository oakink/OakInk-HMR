<br />
<p align="center">
  <p align="center">
    <img src="docs/static/oakink_logo.png"" alt="Logo" width="30%">
  </p>
  <h2 align="center">Hand Mesh Recovery models on OakInk-Image dataset</h2>

  <p align="center">
    <a href="https://lixiny.github.io"><strong>Lixin Yang*</strong></a>
    ·
    <a href="https://kailinli.top"><strong>Kailin Li*</strong></a>
    ·
    <a href=""><strong>Xinyu Zhan*</strong></a>
    ·
    <strong>Fei Wu</strong>
    ·
    <a href="https://anran-xu.github.io"><strong>Anran Xu</strong></a>
    .
    <a href="https://scholar.google.com/citations?user=-_aPWUIAAAAJ&hl=en"><strong>Liu Liu</strong></a>
    ·
    <a href="https://mvig.org"><strong>Cewu Lu</strong></a>
  </p>
  <h3 align="center">CVPR 2022</h3>


  <p align="center">
    <a href="https://arxiv.org/abs/2203.15709">
      <img src='https://img.shields.io/badge/Paper-green?style=for-the-badge&logo=adobeacrobatreader&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
    <a href='https://oakink.net'>
      <img src='https://img.shields.io/badge/Project-orange?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=D35400' alt='Project Page'></a>
    <a href="https://www.youtube.com/watch?v=vNTdeXlLdU8"><img alt="youtube views" title="Subscribe to my YouTube channel" src="https://img.shields.io/badge/Video-red?style=for-the-badge&logo=youtube&labelColor=ce4630&logoColor=red"/></a>
  </p>
</p>

This repo contains the training and evaluation of Hand Mesh Recovery (HMR) models on OakInk-Image dataset.

* [**Integral-Pose**](https://github.com/JimmySuen/integral-human-pose) + IKNet
* [**Res-Loglikelihood-Estimation (RLE)**](https://github.com/Jeff-sjtu/res-loglikelihood-regression) + IKNet
* [**I2L-MeshNet**](https://github.com/mks0601/I2L-MeshNet_RELEASE)

[**+IKNet**]((https://github.com/LyuJ1998/HandTailor)): We use a neural inverse kinematics (IKNet) to covert 21 hand joints to MANO parameters, and then recover the mesh. 

:warning: This benchmark is based on the **public v2.1** release (Oct,18,2022) of the OakInk-Image dataset.  
To download the dataset, please visit the [project website](https://oakink.net).


## Table of content
- [Installation](#installation)
- [Prepare annotation](#prepare-annotation)
- [HMR Models](#hmr-models)
- [Evaluation](#evaluation)
- [Training](#training)

## Installation
Create a conda env from `environment.yml`:
```bash
conda env create -f environment.yml
conda activate oiimage_bm
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Install OakInk data toolkit (**oikit**) as package:
```bash
pip install git+https://github.com/oakink/OakInk.git
``` 
Link the [OakInk dataset](https://github.com/oakink/OakInk/blob/main/docs/datasets.md#download-full-oakink):
```bash
ln -s {path_to}/OakInk ./data
```
Get the [MANO hand model](https://github.com/oakink/OakInk/blob/main/docs/install.md#get-mano-asset): 
```bash 
cp -r {path_to}/mano_v1_2 ./assets
```
## Prepare annotation
To accelerate multiprocess reading, we preprocess and package all annotations of each sample into a single file. Run the following script:  
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

## HMR Models

| method       | IKNet                    | split | MPJPE | PCK-AUC | MPVPE | PA-MPJPE | PA-MPVPE | config                                            |
| ------------ | ------------------------ | ----- | ----- | ------- | ----- | -------- | -------- | ------------------------------------------------- |
| IntegralPose | :heavy_check_mark:       | SP0   | 8.14  | 0.838   | 8.75  | 5.22     | 5.60     | [cfg](config/oii/train_integral_pose_oii_sp0.yml) |
| IntegralPose | :heavy_check_mark:       | SP1   | 10.88 | 0.784   | 11.32 | 6.76     | 6.81     | [cfg](config/oii/train_integral_pose_oii_sp1.yml) |
| IntegralPose | :heavy_check_mark:       | SP2   | 8.22  | 0.837   | 8.83  | 5.30     | 5.66     | [cfg](config/oii/train_integral_pose_oii_sp2.yml) |
| RLE          | :heavy_check_mark:       | SP0   | 9.45  | 0.815   | 9.92  | 5.14     | 5.63     | [cfg](config/oii/train_res_loglike_oii_sp0.yml)   |
| RLE          | :heavy_check_mark:       | SP1   | 13.22 | 0.739   | 13.34 | 6.60     | 6.79     | [cfg](config/oii/train_res_loglike_oii_sp1.yml)   |
| RLE          | :heavy_check_mark:       | SP2   | 9.76  | 0.810   | 10.21 | 5.31     | 5.76     | [cfg](config/oii/train_res_loglike_oii_sp2.yml)   |
| I2L-MeshNet  | :heavy_multiplication_x: | SP0   | 9.24  | 0.818   | 9.05  | 5.03     | 5.03     | [cfg](config/oii/train_i2l_meshnet_oii_sp0.yml)   |
| I2L-MeshNet  | :heavy_multiplication_x: | SP1   | 12.92 | 0.745   | 12.73 | 6.54     | 6.46     | [cfg](config/oii/train_i2l_meshnet_oii_sp1.yml)   |
| I2L-MeshNet  | :heavy_multiplication_x: | SP2   | 9.29  | 0.818   | 9.23  | 5.12     | 5.20     | [cfg](config/oii/train_i2l_meshnet_oii_sp2.yml)   |

Download the corresponding checkpoint from the [Hugging Face](https://huggingface.co/oakink/OakInk-v1-HMR/tree/main), and put it under the `./exp` directory.

## Evaluation

### Integral Pose + IKNet
Estimate 21 joints using 3D heatmap, and recover hand mesh using the 21 joints and IKNet. 
```bash 
IMG -(IntegralPose)-> joints -(IKNet)-> MANO params -(MANO)-> mesh
```   
  
The results will be saved at `exp/{exp_id}_{localtime}`.

```bash
# eval on SP0
python -m train.test_model_hand_tailor \
    --workers 32 \
    --cfg ./exp/oakink_h_sp0_integral_pose_2022_1020_1542_45/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp0_integral_pose_2022_1020_1542_45/checkpoints/checkpoint_100/IntegralPose.pth.tar \
    --gpu_id 0 \
    --exp_id eval_oakink_h_sp0_integral_pose

# eval on SP1
python -m train.test_model_hand_tailor \
    --workers 32 \
    --cfg ./exp/oakink_h_sp1_integral_pose_2022_1022_1348_40/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp1_integral_pose_2022_1022_1348_40/checkpoints/checkpoint_100/IntegralPose.pth.tar \
    --gpu_id 0 \
    --exp_id eval_oakink_h_sp1_integral_pose

# eval on SP2
python -m train.test_model_hand_tailor \
    --workers 32 \
    --cfg ./exp/oakink_h_sp2_integral_pose_2022_1024_1608_34/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp2_integral_pose_2022_1024_1608_34/checkpoints/checkpoint_100/IntegralPose.pth.tar \
    --gpu_id 0 \
    --exp_id eval_oakink_h_sp2_integral_pose
```

### RLE + IKNet
Regress 21 joints using Normalizing Flow, and recover head mesh using the 21 joints and IKNet. 
```bash 
IMG -(RLE)-> joints -(IKNet)-> MANO params -(MANO)-> mesh
```   
```bash
# eval on SP0
python -m train.test_model_hand_tailor \
    --workers 32 \
    --cfg ./exp/oakink_h_sp0_res_loglike_2022_1031_2130_36/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp0_res_loglike_2022_1031_2130_36/checkpoints/checkpoint_100/RegressFlow3D.pth.tar \
    --gpu_id 0 \
    --exp_id eval_oakink_h_sp0_res_loglike

# eval on SP1
python -m train.test_model_hand_tailor \
    --workers 32 \
    --cfg ./exp/oakink_h_sp1_res_loglike_2022_1102_1151_05/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp1_res_loglike_2022_1102_1151_05/checkpoints/checkpoint_100/RegressFlow3D.pth.tar \
    --gpu_id 0 \
    --exp_id eval_oakink_h_sp1_res_loglike

# eval on SP2
python -m train.test_model_hand_tailor \
    --workers 32 \
    --cfg ./exp/oakink_h_sp2_res_loglike_2022_1103_1228_03/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp2_res_loglike_2022_1103_1228_03/checkpoints/checkpoint_100/RegressFlow3D.pth.tar \
    --gpu_id 0 \
    --exp_id eval_oakink_h_sp2_res_loglike
```

### I2L-MeshNet
```bash 
IMG -(I2LMeshNet)-> joints & mesh
```  

```bash
# eval on SP0
python -m train.test_model_hand_tailor \
    --cfg exp/oakink_h_sp0_i2l_meshnet_2022_1117_0527_10/dump_cfg.yaml \
    --reload exp/oakink_h_sp0_i2l_meshnet_2022_1117_0527_10/checkpoints/checkpoint_100/I2L_MeshNet.pth.tar \
    --gpu_id 0 \
    --workers 32 \
    --batch_size 64 \
    --exp_id eval_oakink_h_sp0_i2l_meshnet

# eval on SP1
python -m train.test_model_hand_tailor \
    --cfg exp/oakink_h_sp1_i2l_meshnet_2022_1215_1647_04/dump_cfg.yaml \
    --reload exp/oakink_h_sp1_i2l_meshnet_2022_1215_1647_04/checkpoints/checkpoint_100/I2L_MeshNet.pth.tar \
    --gpu_id 0 \
    --workers 32 \
    --batch_size 64 \
    --exp_id eval_oakink_h_sp1_i2l_meshnet

# eval on SP2
python -m train.test_model_hand_tailor \
    --cfg exp/oakink_h_sp2_i2l_meshnet_2022_1121_0414_05/dump_cfg.yaml \
    --reload exp/oakink_h_sp2_i2l_meshnet_2022_1121_0414_05/checkpoints/checkpoint_100/I2L_MeshNet.pth.tar \
    --gpu_id 0 \
    --workers 32 \
    --batch_size 64 \
    --exp_id eval_oakink_h_sp2_i2l_meshnet
```

## Training

### Integral Pose
```bash
# train on SP0
python -m train.train_model --cfg config/oii/train_integral_pose_oii_sp0.yml --gpu_id 0,1,2,3 --workers 32 --exp_id oakink_h_sp0_integral_pose

# train on SP1
python -m train.train_model --cfg config/oii/train_integral_pose_oii_sp1.yml --gpu_id 0,1,2,3 --workers 32 --exp_id oakink_h_sp1_integral_pose

# train on SP2
python -m train.train_model --cfg config/oii/train_integral_pose_oii_sp2.yml --gpu_id 0,1,2,3 --workers 32 --exp_id oakink_h_sp2_integral_pose
```

### RLE

```bash
# train on SP0
python -m train.train_model --cfg config/oii/train_res_loglike_oii_sp0.yml --gpu_id 0,1,2,3 --workers 32 --exp_id oakink_h_sp0_res_loglike

# train on SP1
python -m train.train_model --cfg config/oii/train_res_loglike_oii_sp1.yml --gpu_id 0,1,2,3 --workers 32 --exp_id oakink_h_sp1_res_loglike

# train on SP2
python -m train.train_model --cfg config/oii/train_res_loglike_oii_sp2.yml --gpu_id 0,1,2,3 --workers 32 --exp_id oakink_h_sp2_res_loglike
```

### I2L-MeshNet

```bash
# train on SP0
python -m train.train_model --cfg config/oii/train_i2l_meshnet_oii_sp0.yml --gpu_id 0,1,2,3 --workers 32 --exp_id oakink_h_sp0_i2l_meshnet

# train on SP1
python -m train.train_model --cfg config/oii/train_i2l_meshnet_oii_sp1.yml --gpu_id 0,1,2,3 --workers 32 --exp_id oakink_h_sp1_i2l_meshnet

# train on SP2
python -m train.train_model --cfg config/oii/train_i2l_meshnet_oii_sp2.yml --gpu_id 0,1,2,3 --workers 32 --exp_id oakink_h_sp2_i2l_meshnet
```

## Citation

If you find OakInk-Image dataset useful for your research, please considering cite us:  
```bibtex
@inproceedings{YangCVPR2022OakInk,
  author    = {Yang, Lixin and Li, Kailin and Zhan, Xinyu and Wu, Fei and Xu, Anran and Liu, Liu and Lu, Cewu},
  title     = {{OakInk}: A Large-Scale Knowledge Repository for Understanding Hand-Object Interaction},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022},
}
```

<details><summary>Integral-Pose, RLE, I2L-MeshNet</summary>  

```bibtex
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
