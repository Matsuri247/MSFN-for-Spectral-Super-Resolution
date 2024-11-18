import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from NTIRE2022_dataset import TrainDataset, ValidDataset
from architecture import *
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR, SAM, SSIM
import datetime

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='MSFN')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--end_epoch", type=int, default=3, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='./exp/MSFN/', help='path log files')
parser.add_argument("--data_root", type=str, default='../dataset/NTIRE2022')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
opt = parser.parse_args()
# 获取环境变量
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# load dataset
print(f"Start training: {opt.method}")
print("\nloading dataset ...")
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=opt.stride)
per_epoch_iteration = len(train_data) // opt.batch_size
print(f"Iteration per epoch: {per_epoch_iteration}")
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
print("Validation set samples: ", len(val_data))

# iterations
total_iteration = per_epoch_iteration * opt.end_epoch

# criterion
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_sam = SAM()
criterion_ssim = SSIM()

# model
pretrained_model_path = opt.pretrained_model_path
method = opt.method
model = model_generator(method, pretrained_model_path).cuda()
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

# output path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = opt.outf + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.is_available():
    model.cuda()
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()
    criterion_sam.cuda()
    criterion_ssim.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# ADAM优化器
optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
# 初始学习率4e-4，余弦退火策略
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

# logging
log_dir = os.path.join(opt.outf, 'train.log')
logger = initialize_logger(log_dir)

# Resume
resume_file = opt.pretrained_model_path
if resume_file is not None:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

def main():
    cudnn.benchmark = True
    iteration = 0
    record_mrae_loss = 1000
    while iteration<total_iteration:
        model.train()
        losses = AverageMeter()
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=10,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)
        for i, (images, labels) in enumerate(train_loader):
            labels = labels.cuda()
            images = images.cuda()
            images = Variable(images) # 使用这种数据相当于将数据加入到一个节点,这种数据有grad属性
            labels = Variable(labels)
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            output = model(images)
            loss = criterion_mrae(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.data)
            iteration = iteration+1
            if iteration % 20 == 0:
                print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
                      % (iteration, total_iteration, lr, losses.avg))
            if iteration % 1000 == 0:
                mrae_loss, rmse_loss, psnr_loss, sam_loss, ssim_loss = validate(val_loader, model)
                print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PSNR:{psnr_loss}, SAM:{sam_loss}, SSIM:{ssim_loss}.')
                # Save model
                if iteration >= 180000:
                    print(f'Saving to {opt.outf}')
                    save_checkpoint(opt.outf, (iteration // per_epoch_iteration + 1), iteration, model, optimizer)
                    if mrae_loss < record_mrae_loss:
                        record_mrae_loss = mrae_loss
                # print loss
                print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Validate MRAE: %.9f, "
                      "Validate RMSE: %.9f, Validate PSNR: %.9f, Validate SAM: %.9f, Validate SSIM: %.9f " %
                      (iteration, (iteration // per_epoch_iteration + 1), lr, losses.avg, mrae_loss, rmse_loss, psnr_loss, sam_loss, ssim_loss))
                logger.info(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train Loss: %.9f, Validate MRAE: %.9f, "
                            "Validate RMSE: %.9f, Validate PSNR: %.9f, Validate SAM: %.9f, Validate SSIM: %.9f " %
                            (iteration, (iteration // per_epoch_iteration + 1), lr, losses.avg, mrae_loss, rmse_loss, psnr_loss, sam_loss, ssim_loss))
    return 0

# Validate
def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_sam = AverageMeter()
    losses_ssim = AverageMeter()

    for i, (input, target) in enumerate(val_loader):
        # for NTIRE2022 dataset, validate on NTIRE2022 competition area
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            # NTIRE2022 Competition
            loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_sam = criterion_sam(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_ssim = criterion_ssim(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])

        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
        losses_sam.update(loss_sam.data)
        losses_ssim.update(loss_ssim.data)

    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_ssim.avg

if __name__ == '__main__':
    main()
    print(torch.__version__)