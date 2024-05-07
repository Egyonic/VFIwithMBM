import os
import sys

sys.path.append('.')
import cv2
import math
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from model.pytorch_msssim import ssim_matlab
from model.RIFE import Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
在测试前需要先把BiformerBlock的 n_win从7变为更大的值才行，如14。不然注意力计算时的单个块尺寸太大，内存不足
"""


divisor = 32
class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
    def __init__(self, dims, divisor=divisor):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

model_name = 'IFNet_bf_resnet_cbam_HM'
model = Model(model_name=model_name)
model.load_model('train_log/IFNet_bf_resnet_cbam_HM')
model.eval()
model.device()

path = r'E:\Workspace\Datasets\SNU-FILM'
f = open(path + '\\test-extreme.txt', 'r')
psnr_list = []
ssim_list = []
count = 0

for line in f:
    file_paths = line.strip().split()
    file_path_1, file_path_2, file_path_3 = file_paths

    img0 = os.path.join(r'E:\Workspace\Datasets', file_path_1[5:].replace('/', '\\'))
    img1 = os.path.join(r'E:\Workspace\Datasets', file_path_3[5:].replace('/', '\\'))
    gt = os.path.join(r'E:\Workspace\Datasets', file_path_2[5:].replace('/', '\\'))
    print(f"File {count}:", img0)

    img0 = (torch.tensor(cv2.imread(img0).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
    img1 = (torch.tensor(cv2.imread(img1).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
    gt = (torch.tensor(cv2.imread(gt).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)

    padder = InputPadder(img0.shape)
    img0, img1 = padder.pad(img0, img1)

    # if img0.shape[3] < 750 and img0.shape[2] < 750:
    #     pred = model.inference(img0, img1)[0]
    # elif img0.shape[3] > img0.shape[2]:  # 宽度大于高度，按照宽度切分
    if img0.shape[3] > img0.shape[2]:  # 宽度大于高度，按照宽度切分
        w = img0.shape[3] // 2
        img0_left = img0[:, :, :, :w]
        img1_left = img1[:, :, :, :w]
        img0_right = img0[:, :, :, w:]
        img1_right = img1[:, :, :, w:]

        pred_left = model.inference(img0_left, img1_left)[0]
        pred_right = model.inference(img0_right, img1_right)[0]
        pred = torch.cat((pred_left, pred_right), dim=2)
    else:  # 宽度小于等于高度，按照高度切分
        h = img0.shape[2] // 2
        img0_up = img0[:, :, :h, :]
        img1_up = img1[:, :, :h, :]
        img0_down = img0[:, :,  h:, :]
        img1_down = img1[:, :, h:, :]

        pred_up = model.inference(img0_up, img1_up)[0]
        pred_down = model.inference(img0_down, img1_down)[0]
        pred = torch.cat((pred_up, pred_down), dim=1)

    pred = padder.unpad(pred)
    ssim = ssim_matlab(gt,
                       torch.round(pred * 255).unsqueeze(0) / 255.).detach().cpu().numpy()

    gt = gt.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    pred = pred.detach().cpu().numpy().transpose(1, 2, 0)
    psnr = -10 * math.log10(((gt - pred) * (gt - pred)).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    print("Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))
    count = count + 1
    del img0, img1, gt, pred