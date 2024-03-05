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
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_result = False
save_path = 'inference_result'  # 保存预测结果
timestruct = time.localtime(time.time())
t_s = time.strftime('%Y-%m-%d_%H_%M_%S', timestruct)
model_name = 'IFNet_bf_resnet_cbam_L'

result_dir = os.path.join(save_path, 'vimeo90k', model_name + '_' + t_s)
if save_result:
    os.mkdir(result_dir)

model = Model(model_name=model_name)
model.load_model('train_log/IFNet_bf_resnet_cbam_L')
model.eval()
model.device()

path = r'E:\Workspace\Datasets\ucf101_train_test_split\ucf101_interp_ours'
dirs = os.listdir(path)
psnr_list = []
ssim_list = []
print(len(dirs))
for d in dirs:
    img0 = (path + '\\' + d + '/frame_00.png').replace("/", "\\")
    img1 = (path + '\\' + d + '/frame_02.png').replace("/", "\\")
    gt = (path + '\\' + d + '/frame_01_gt.png').replace("/", "\\")
    print(img0)
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
