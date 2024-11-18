# Multi-stage-Spatial-Spectral-Fusion-Network-for-Spectral-Super-Resolution
The code implementation of paper "Multi-stage Spatial-Spectral Fusion Network for Spectral Super-Resolution".  

# Train 

【NTIRE2022】train.py

`--method MSFN --batch_size 20 --end_epoch 3 --init_lr 4e-4 --outf ./exp/MSFN/ --data_root ../dataset/NTIRE2022 --patch_size 128 --stride 8 --gpu_id 0`

【NTIRE2020】train2020.py

`--method MSFN --batch_size 20 --end_epoch 5 --init_lr 4e-4 --outf ./exp/MSFN/ --data_root ../dataset/NTIRE2020 --patch_size 64 --stride 8 --gpu_id 0`

# Test

【NTIRE2022】test.py

`--data_root ../dataset/NTIRE2022/  --method MSFN --pretrained_model_path ../model_pth/MSFN/NTIRE2022/MSFN.pth --outf ./exp/2022/MSFN/ --gpu_id 0`

【NTIRE2020】test2020.py

`--data_root ../dataset/NTIRE2020/  --method MSFN --pretrained_model_path ../model_pth/MSFN/NTIRE2020/MSFN.pth --outf ./exp/2020/MSFN/ --gpu_id 0`

# Citation
@ARTICLE{msfn,
  author={Wu, Yaohang and Dian, Renwei and Li, Shutao},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Multistage Spatial–Spectral Fusion Network for Spectral Super-Resolution}, 
  year={2024},
  volume={},
  number={},
  pages={1-11},
  keywords={Image reconstruction;Correlation;Superresolution;Hyperspectral imaging;Spatial resolution;Convolutional neural networks;Attention mechanisms;Accuracy;Transformers;Tensors;Attention mechanism;hyperspectral image (HSI);spectral super-resolution (SSR)},
  doi={10.1109/TNNLS.2024.3460190}}

# Acknowledgement
Thanks a lot for the outstanding work and dedication from ([https://github.com/caiyuanhao1998/MST](https://github.com/caiyuanhao1998/MST-plus-plus)). The code structure and datasets are borrowed from MST++. We sincerely appreciate their contributions.
