import hdf5storage
import torch
import torch.nn as nn
import numpy as np
import math
from ssim_torch import ssim

def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)

def reconstruct(model, input, target, crop_size, stride):
    abundance_matrix = torch.zeros_like(target).cuda()
    index_matrix = torch.zeros_like(target).cuda()
    h_idx = []
    for j in range(0, input.shape[2]-crop_size+1, stride):
        h_idx.append(j)
    h_idx.append(input.shape[2]-crop_size)

    w_idx = []
    for j in range(0, input.shape[3]-crop_size+1, stride):
        w_idx.append(j)
    w_idx.append(input.shape[3]-crop_size)

    # patch-wise reconstruction to avoid out of memory(especially AWAN)
    for h in h_idx:
        for w in w_idx:
            patch_input = input[:, :, h:h+crop_size, w:w+crop_size]
            with torch.no_grad():
                # model output
                patch_output = model(patch_input) # [b,c,h,w]
                patch_output = torch.clamp(patch_output,0,1)
                # from patch to whole img
                abundance_matrix[:, :, h:h+crop_size, w:w+crop_size] = patch_output + abundance_matrix[:, :, h:h+crop_size, w:w+crop_size]
                # deal with overlapping
                index_matrix[:, :, h:h+crop_size, w:w+crop_size] = 1 + index_matrix[:, :, h:h+crop_size, w:w+crop_size]
    output = abundance_matrix / index_matrix
    return output

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # 求平均
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 按[0,1]归一化像素值算的
class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / (label + 1e-6)
        # mrae = torch.mean(error.view(-1))
        mrae = torch.mean(error.reshape(-1))
        return mrae

# 按[0,1]归一化像素值算的
class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        # rmse = torch.sqrt(torch.mean(sqrt_error.view(-1)))
        rmse = torch.sqrt(torch.mean(sqrt_error.reshape(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        # Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        # Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Itrue = im_true.clamp(0., 1.).mul_(data_range).reshape(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).reshape(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)

# degree
class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()

    def forward(self, im_true, im_fake):
        # print(im_fake.size())
        im_true = im_true.squeeze(0)
        im_fake = im_fake.squeeze(0)
        # print(im_true.shape)
        C = im_true.size()[0]
        H = im_true.size()[1]
        W = im_true.size()[2]
        im_fake.reshape(C, H * W)
        im_true.reshape(C, H * W)
        esp = 1e-12
        Itrue = im_true.clone()  # .resize_(C, H*W)
        Ifake = im_fake.clone()  # .resize_(C, H*W)
        nom = torch.mul(Itrue, Ifake).sum(dim=0)  # .resize_(H*W)
        denominator = Itrue.norm(p=2, dim=0, keepdim=True).clamp(min=esp) * \
                      Ifake.norm(p=2, dim=0, keepdim=True).clamp(min=esp)
        denominator = denominator.squeeze()
        sam = torch.div(nom, denominator).acos()
        sam[sam != sam] = 0
        sam_sum = torch.sum(sam) / (H * W) / np.pi * 180
        return sam_sum

class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()

    def forward(self, im_true, im_fake):
        return ssim(im_true, im_fake)


def calc_psnr(img1, img2, data_range=255):
    img1 = img1.clamp(0., 1.).mul_(data_range).cpu().numpy()
    img2 = img2.clamp(0., 1.).mul_(data_range).cpu().numpy()
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
