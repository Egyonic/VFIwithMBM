import os
import sys
import time
sys.path.append('.')
import cv2
import math
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from model.pytorch_msssim import ssim_matlab
from model.RIFE import Model


def crop(img0, gt, img1, h, w):
    ih, iw, _ = img0.shape
    x = np.random.randint(0, ih - h + 1)
    y = np.random.randint(0, iw - w + 1)
    img0 = img0[x:x + h, y:y + w, :]
    img1 = img1[x:x + h, y:y + w, :]
    gt = gt[x:x + h, y:y + w, :]
    return img0, gt, img1

def center_crop(img0, gt, img1, h, w):
    ih, iw, _ = img0.shape
    x = int((ih - h) / 2)
    y = int((iw - w) / 2)
    img0 = img0[x:x + h, y:y + w, :]
    img1 = img1[x:x + h, y:y + w, :]
    gt = gt[x:x + h, y:y + w, :]
    return img0, gt, img1


save_result = True
path = '/home/usst/egy/data/vimeo90k/'
f = open(path + 'tri_testlist.txt', 'r')
save_path = 'inference_result'  # 保存预测结果
timestruct = time.localtime(time.time())
t_s = time.strftime('%Y-%m-%d_%H_%M_%S', timestruct)
result_dir = os.path.join(save_path, 'vimeo90k',t_s)
if save_result:
    os.mkdir(result_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(model_name='IFNet_bf')
model.load_model('train_log/IFNet_bf')
model.eval()
model.device()
psnr_list = []
ssim_list = []

count = 0
save_interval = 200

for i in f:
    name = str(i).strip()
    if(len(name) <= 1):
        continue
    print(path + 'target/' + name + '/im1.png')
    I0 = cv2.imread(path + 'target/' + name + '/im1.png')
    I1 = cv2.imread(path + 'target/' + name + '/im2.png')
    I2 = cv2.imread(path + 'target/' + name + '/im3.png')
    I0, I1, I2 = center_crop(I0, I1, I2, 224, 448)
    # I0, I1, I2 = center_crop(I0, I1, I2, 224, 336)
    if count % save_interval == 0 and save_result:
        cv2.imwrite(os.path.join(result_dir, f'im_{count}_I0.png'), I0)
        cv2.imwrite(os.path.join(result_dir, f'im_{count}_I1.png'), I1)
        cv2.imwrite(os.path.join(result_dir, f'im_{count}_I2.png'), I2)

    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    mid = model.inference(I0, I2)[0]
    ssim = ssim_matlab(torch.tensor(I1.transpose(2, 0, 1)).to(device).unsqueeze(0) / 255., torch.round(mid * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
    # mid = np.round((mid * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.
    if count % save_interval == 0 and save_result:
        mid_tmp = np.round((mid * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0)
        cv2.imwrite(os.path.join(result_dir, f'im_{count}_predict.png'), mid_tmp)
    mid = np.round((mid * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.
    I1 = I1 / 255.
    psnr = -10 * math.log10(((I1 - mid) * (I1 - mid)).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    print("Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))
    count = count + 1
