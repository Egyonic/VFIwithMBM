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
from model.RIFE_with_mask import Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = 'IFNet_bf_resnet_cbam_L'
model = Model(model_name=model_name)
model.load_model('train_log/IFNet_bf_resnet_cbam_L')
model.eval()
model.device()

path = r'E:\Workspace\Datasets\SNU-FILM'
f = open(path + '\\test-easy.txt', 'r')
psnr_list = []
ssim_list = []

for line in f:
    file_paths = line.strip().split()
    file_path_1, file_path_2, file_path_3 = file_paths

    img0 = os.path.join(r'E:\Workspace\Datasets', file_path_1[5:].replace('/', '\\'))
    img1 = os.path.join(r'E:\Workspace\Datasets', file_path_3[5:].replace('/', '\\'))
    gt = os.path.join(r'E:\Workspace\Datasets', file_path_2[5:].replace('/', '\\'))
    print("File 1:", img0)

    img0 = (torch.tensor(cv2.imread(img0).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
    img1 = (torch.tensor(cv2.imread(img1).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
    gt = (torch.tensor(cv2.imread(gt).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
    pred = model.inference(img0, img1)[0]
    ssim = ssim_matlab(gt, torch.round(pred * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
    out = pred.detach().cpu().numpy().transpose(1, 2, 0)
    out = np.round(out * 255) / 255.
    gt = gt[0].cpu().numpy().transpose(1, 2, 0)
    psnr = -10 * math.log10(((gt - out) * (gt - out)).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    print("Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))
