# Multi-stage-Spatial-Spectral-Fusion-Network-for-Spectral-Super-Resolution
The code implementation of paper "Multi-stage Spatial-Spectral Fusion Network for Spectral Super-Resolution".

# Environment
```
Python=3.8.18
opencv-python==4.4.0.46
einops
torchvision==0.8.2
torch==1.7.1
scipy==1.0.0
h5py
hdf5storage
tqdm
torchinfo
```

# Data Preparation
You can find NTIRE2022 dataset from ([here](https://github.com/caiyuanhao1998/MST-plus-plus)). Make sure you place the dataset as the following form (similar for NTIRE2020 dataset, too):

```
|--MSFN
    |--model_pth
    |--test_develop_code
    |--train_code  
    |--dataset 
        |--NTIRE2022
            |--RGB
                |--ARAD_1K_0001.jpg
                |--ARAD_1K_0002.jpg
                ： 
                |--ARAD_1K_0950.jpg
            |--Spec
                |--ARAD_1K_0001.mat
                |--ARAD_1K_0002.mat
                ： 
                |--ARAD_1K_0950.mat
            |--split_txt
                |--train_list.txt
                |--valid_list.txt
                |--test_list.txt
```

Not uploading NTIRE2020 dataset for no sufficient space in my cloud drive.

# Train

【NTIRE2022】train.py
```
--method MSFN --batch_size 20 --end_epoch 3 --init_lr 4e-4 --outf ./exp/MSFN/ --data_root ../dataset/NTIRE2022 --patch_size 128 --stride 8 --gpu_id 0
```

【NTIRE2020】train2020.py
```
--method MSFN --batch_size 20 --end_epoch 5 --init_lr 4e-4 --outf ./exp/MSFN/ --data_root ../dataset/NTIRE2020 --patch_size 64 --stride 8 --gpu_id 0
```

# Test
Download the pretrained model for NTIRE2022: ([here](https://drive.google.com/file/d/1cCmo_NPgwcP1R6wvGD9uDcP7IKdU0Ue8/view?usp=drive_link))

Download the pretrained model for NTIRE2020: ([here](https://drive.google.com/file/d/1DqafMHGSMTJvs2dz1Z2c6oGHd-VJotSA/view?usp=drive_link))

Make sure that you place them in the right place.

【NTIRE2022】test.py
```
--data_root ../dataset/NTIRE2022/  --method MSFN --pretrained_model_path ../model_pth/MSFN/NTIRE2022/MSFN_2022.pth --outf ./exp/2022/MSFN/ --gpu_id 0
```

【NTIRE2020】test2020.py
```
--data_root ../dataset/NTIRE2020/  --method MSFN --pretrained_model_path ../model_pth/MSFN/NTIRE2020/MSFN_2020.pth --outf ./exp/2020/MSFN/ --gpu_id 0
```

# Citation
If you find this repo useful, please consider citing our works.
```
@ARTICLE{msfn,
  author={Wu, Yaohang and Dian, Renwei and Li, Shutao},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Multistage Spatial-Spectral Fusion Network for Spectral Super-Resolution}, 
  year={2025},
  volume={36},
  number={7},
  pages={12736-12746},
  keywords={Image reconstruction;Correlation;Superresolution;Hyperspectral imaging;Spatial resolution;Convolutional neural networks;Attention mechanisms;Accuracy;Transformers;Tensors;Attention mechanism;hyperspectral image (HSI);spectral super-resolution (SSR)},
  doi={10.1109/TNNLS.2024.3460190}}
```

# Acknowledgement
Thanks a lot for the outstanding work and dedication from [https://github.com/caiyuanhao1998/MST-plus-plus](https://github.com/caiyuanhao1998/MST-plus-plus). The code structure and datasets are borrowed from MST++. We sincerely appreciate their contributions.
