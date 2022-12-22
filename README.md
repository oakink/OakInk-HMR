## A. Install

### a.1 (Option 1) Create oakink-benchmark env

```shell
conda env create -f environment.yml
conda activate oakink-benchmark
pip install -r requirements.txt
```

### a.2 (Option 2) install extra packages in existing `oakink` conda env

If [`OakInk`](https://github.com/oakink/OakInk) env is already installed, you just need to install the extra packages required for benchmark.
1. comment out `manotorch` and `pytorch3d` in `requirements.txt` with `#`
2. ```shell
   pip install -r requirements.txt
   ```
3. restore `requirements.txt`

## B. Link Data

### b.1 data

Create a folder named `data`  
Link datafile from /XXdisk to `data` folder:

### b.2 assets


Now you folder should look like this

```
assets
├── mano_v1_2
└── postprocess

data
└── OakInk
    └── image
```

## b.3 pack annotations

To avoid opening too many small files in pytorch dataloaders which might exceed user limit, annotations need to be packed into a single archive for each sample.

1. Follow instructions in `OakInk` repo to install oikit.
2. Run following instructions (`default` and `train+val` are for illustration, they can be changed to the desired split)
   ```bash
   python dev/pack_oakink_image.py --mode_split default --data_split train+val
   ```


## C. Model Zoo

### c.1 hmr

| Model             | HandTailor | Split | MPJPE | PCK AUC | MPVPE | PA-MPJPE | PA-MPVPE | Checkpoint |
| ----------------- | ---------- | ----- | ----- | ------- | ----- | -------- | -------- | ---------- |
| IntegralPose      | use        | SP0   | 8.14  | 0.838   | 8.75  | 5.22     | 5.60     |            |
| IntegralPose      | use        | SP1   | 10.88 | 0.784   | 11.32 | 6.76     | 6.81     |            |
| IntegralPose      | use        | SP2   | 8.22  | 0.837   | 8.83  | 5.30     | 5.66     |            |
| Res-Loglikelihood | use        | SP0   | 9.45  | 0.815   | 9.92  | 5.14     | 5.63     |            |
| Res-Loglikelihood | use        | SP1   | 13.22 | 0.739   | 13.34 | 6.60     | 6.79     |            |
| Res-Loglikelihood | use        | SP2   | 9.76  | 0.810   | 10.21 | 5.31     | 5.76     |            |
| I2L Meshnet       |            | SP0   | 9.24  | 0.818   | 9.05  | 5.03     | 5.03     |            |
| I2L Meshnet       |            | SP1   | 12.92 | 0.745   | 12.73 | 6.54     | 6.46     |            |
| I2L Meshnet       |            | SP2   | 9.29  | 0.818   | 9.23  | 5.12     | 5.20     |            |


## D. Training

### d.1 IntegralPose

```bash
python -m train.train_model --cfg config/oii/train_integral_pose_oii_sp0.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp0_integral_pose
python -m train.train_model --cfg config/oii/train_integral_pose_oii_sp1.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp1_integral_pose
python -m train.train_model --cfg config/oii/train_integral_pose_oii_sp2.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp2_integral_pose
```

### d.2 RLE

```bash
python -m train.train_model --cfg config/oii/train_res_loglike_oii_sp0.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp0_res_loglike
python -m train.train_model --cfg config/oii/train_res_loglike_oii_sp1.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp1_res_loglike
python -m train.train_model --cfg config/oii/train_res_loglike_oii_sp2.yml --gpu_id 0,1,2,3 --workers 56 --exp_id oakink_h_sp2_res_loglike
```

### d.3 i2lmeshnet

```bash
python -m train.train_model --cfg config/oii/train_i2l_meshnet_oii_sp0.yml --gpu_id 4,5,6,7 --workers 56 --exp_id oakink_h_sp0_i2l_meshnet
python -m train.train_model --cfg config/oii/train_i2l_meshnet_oii_sp1.yml --gpu_id 4,5,6,7 --workers 56 --exp_id oakink_h_sp1_i2l_meshnet
python -m train.train_model --cfg config/oii/train_i2l_meshnet_oii_sp2.yml --gpu_id 4,5,6,7 --workers 56 --exp_id oakink_h_sp2_i2l_meshnet
```

## E. Testing

### e.1 Integral Pose

```bash
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp0_integral_pose_2022_1020_1542_45/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp0_integral_pose_2022_1020_1542_45/checkpoints/checkpoint_100/IntegralPose.pth.tar \
    --gpu_id 1 \
    --exp_id eval_oakink_h_sp0_integral_pose_2022_1020_1542_45
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp1_integral_pose_2022_1022_1348_40/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp1_integral_pose_2022_1022_1348_40/checkpoints/checkpoint_100/IntegralPose.pth.tar \
    --gpu_id 1 \
    --exp_id eval_oakink_h_sp1_integral_pose_2022_1022_1348_40
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp2_integral_pose_2022_1024_1608_34/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp2_integral_pose_2022_1024_1608_34/checkpoints/checkpoint_100/IntegralPose.pth.tar \
    --gpu_id 1 \
    --exp_id eval_oakink_h_sp2_integral_pose_2022_1024_1608_34
```

### e.2 rle

```bash
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp0_res_loglike_2022_1031_2130_36/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp0_res_loglike_2022_1031_2130_36/checkpoints/checkpoint_100/RegressFlow3D.pth.tar \
    --gpu_id 3 \
    --exp_id eval_oakink_h_sp0_res_loglike_2022_1031_2130_36
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp1_res_loglike_2022_1102_1151_05/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp1_res_loglike_2022_1102_1151_05/checkpoints/checkpoint_100/RegressFlow3D.pth.tar \
    --gpu_id 3 \
    --exp_id eval_oakink_h_sp1_res_loglike_2022_1102_1151_05
python -m train.test_model_hand_tailor \
    --workers 56 \
    --cfg ./exp/oakink_h_sp2_res_loglike_2022_1103_1228_03/dump_cfg.yaml \
    --reload ./exp/oakink_h_sp2_res_loglike_2022_1103_1228_03/checkpoints/checkpoint_100/RegressFlow3D.pth.tar \
    --gpu_id 3 \
    --exp_id eval_oakink_h_sp2_res_loglike_2022_1103_1228_03
```

### e.3 i2lmeshnet

```bash
python -m train.test_model_hand_tailor \
    --cfg exp/oakink_h_sp0_i2l_meshnet_2022_1117_0527_10/dump_cfg.yaml \
    --reload exp/oakink_h_sp0_i2l_meshnet_2022_1117_0527_10/checkpoints/checkpoint_100/I2L_MeshNet.pth.tar \
    --gpu_id 4 \
    --workers 32 \
    --batch_size 64 \
    --exp_id eval_oakink_h_sp0_i2l_meshnet_2022_1117_0527_10
python -m train.test_model_hand_tailor \
    --cfg exp/oakink_h_sp1_i2l_meshnet_2022_1215_1647_04/dump_cfg.yaml \
    --reload exp/oakink_h_sp1_i2l_meshnet_2022_1215_1647_04/checkpoints/checkpoint_100/I2L_MeshNet.pth.tar \
    --gpu_id 4 \
    --workers 32 \
    --batch_size 64 \
    --exp_id eval_oakink_h_sp1_i2l_meshnet_2022_1215_1647_04
python -m train.test_model_hand_tailor \
    --cfg exp/oakink_h_sp2_i2l_meshnet_2022_1121_0414_05/dump_cfg.yaml \
    --reload exp/oakink_h_sp2_i2l_meshnet_2022_1121_0414_05/checkpoints/checkpoint_100/I2L_MeshNet.pth.tar \
    --gpu_id 4 \
    --workers 32 \
    --batch_size 64 \
    --exp_id eval_oakink_h_sp2_i2l_meshnet_2022_1121_0414_05
```
