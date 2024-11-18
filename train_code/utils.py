from __future__ import division
import torch
import torch.nn as nn
import logging
import numpy as np
import os
from ssim_torch import ssim

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

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch_%diter.pth' % (epoch, iteration)))

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / (label + 1e-6)
        # mrae = torch.mean(error.view(-1))
        mrae = torch.mean(error.reshape(-1))
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        # rmse = torch.sqrt(torch.mean(sqrt_error.view(-1))) # NTIRE2022
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
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.) # torch.log和np.log都是以e为底；log换底公式
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

def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iteraion is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr*(1 - iteraion/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close