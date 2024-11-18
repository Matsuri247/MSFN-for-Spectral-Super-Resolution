# Multi-stage-Spatial-Spectral-Fusion-Network-for-Spectral-Super-Resolution
The code implementation of paper "Multi-stage Spatial-Spectral Fusion Network for Spectral Super-Resolution".  

# Train 
train.py # NTIRE2022
--method MSFN --batch_size 20 --end_epoch 3 --init_lr 4e-4 --outf ./exp/MSFN/ --data_root ../dataset/NTIRE2022 --patch_size 128 --stride 8 --gpu_id 0

train2020.py # NTIRE2020
--method MSFN --batch_size 20 --end_epoch 5 --init_lr 4e-4 --outf ./exp/MSFN/ --data_root ../dataset/NTIRE2020 --patch_size 64 --stride 8 --gpu_id 0

# Test
test.py # NTIRE2022
--data_root ../dataset/NTIRE2022/  --method MSFN --pretrained_model_path ../model_pth/MSFN/NTIRE2022/MSFN.pth --outf ./exp/2022/MSFN/ --gpu_id 0

test2020.py # NTIRE2020
--data_root ../dataset/NTIRE2020/  --method MSFN --pretrained_model_path ../model_pth/MSFN/NTIRE2020/MSFN.pth --outf ./exp/2020/MSFN/ --gpu_id 0

# Acknowledgement
Thanks a lot for the outstanding work and dedication from ([https://github.com/caiyuanhao1998/MST](https://github.com/caiyuanhao1998/MST-plus-plus)). The code structure and datasets are borrowed from MST++. We sincerely appreciate their contributions.
