from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py

class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8):
        self.crop_size = crop_size
        self.hypers_path = []
        self.bgrs_path = []
        self.arg = arg
        h,w = 482,512  # img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum
        self.bgr2rgb = bgr2rgb

        hyper_data_path = f'{data_root}/Spec/'
        bgr_data_path = f'{data_root}/RGB/'

        with open(f'{data_root}/split_txt/train_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n','.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper) of NTIRE2022 Train dataset:{len(hyper_list)}')
        print(f'len(bgr) of NTIRE2022 Train dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            bgr_path = bgr_data_path + bgr_list[i]
            self.hypers_path.append(hyper_path)
            self.bgrs_path.append(bgr_path)
        self.img_num = len(self.hypers_path)
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # [c,h,w]
        # Random rotation
        for j in range(rotTimes):
            img_ = np.rot90(img.copy(), axes=(1, 2))
            img = img_.copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, :, ::-1].copy()
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, ::-1, :].copy()
        return img

    #fetching a data sample for a given key.
    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line

        with h5py.File(self.hypers_path[img_idx],'r') as mat:
            hyper = np.float32(np.array(mat['cube']))
        mat.close()
        hyper = np.transpose(hyper,[0,2,1])
        bgr = cv2.imread(self.bgrs_path[img_idx])
        if self.bgr2rgb:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bgr = np.float32(bgr)
        bgr = (bgr-bgr.min()) / (bgr.max()-bgr.min())
        bgr = np.transpose(bgr, [2,0,1]) # [3,482,512]

        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size, w_idx * stride:w_idx * stride + crop_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img*self.img_num

class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.hypers_path = []
        self.bgrs_path = []
        self.bgr2rgb = bgr2rgb

        hyper_data_path = f'{data_root}/Spec/'
        bgr_data_path = f'{data_root}/RGB/'

        with open(f'{data_root}/split_txt/valid_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper_valid) of NTIRE2022 Validate dataset:{len(hyper_list)}')
        print(f'len(bgr_valid) of NTIRE2022 Validate dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            bgr_path = bgr_data_path + bgr_list[i]
            self.hypers_path.append(hyper_path)
            self.bgrs_path.append(bgr_path)

    def __getitem__(self, idx):
        with h5py.File(self.hypers_path[idx], 'r') as mat:
            hyper = np.float32(np.array(mat['cube']))
        mat.close()
        hyper = np.transpose(hyper, [0, 2, 1])
        bgr = cv2.imread(self.bgrs_path[idx])
        if self.bgr2rgb:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bgr = np.float32(bgr)
        bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
        bgr = np.transpose(bgr, [2, 0, 1])
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers_path)

class TestDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.hypers_path = []
        self.bgrs_path = []
        self.bgr2rgb = bgr2rgb

        hyper_data_path = f'{data_root}/Spec/'
        bgr_data_path = f'{data_root}/RGB/'

        with open(f'{data_root}/split_txt/test_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat', 'jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper_test) of NTIRE2022 Test dataset:{len(hyper_list)}')
        print(f'len(bgr_test) of NTIRE2022 Test dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            bgr_path = bgr_data_path + bgr_list[i]
            self.hypers_path.append(hyper_path)
            self.bgrs_path.append(bgr_path)

    def __getitem__(self, idx):
        with h5py.File(self.hypers_path[idx],'r') as mat:
            hyper = np.float32(np.array(mat['cube']))
        mat.close()
        hyper = np.transpose(hyper,[0,2,1])
        bgr = cv2.imread(self.bgrs_path[idx])
        if self.bgr2rgb:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bgr = np.float32(bgr)
        bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
        bgr = np.transpose(bgr, [2, 0, 1])
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers_path)