import os
import sys
sys.path.append('.')
import time
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(model_name='IFNet_bf')
model.load_model('train_log/IFNet_bf')
model.eval()
model.device()

save_result = False
save_path = 'inference_result'  # 保存预测结果
timestruct = time.localtime(time.time())
t_s = time.strftime('%Y-%m-%d_%H_%M_%S', timestruct)
result_dir = os.path.join(save_path, 'ATD12k', t_s)
if save_result:
    os.mkdir(result_dir)
count = 0
save_interval = 100

path = '/home/usst/egyonic/data/atd12/datasets/test_2k_540p/'
dirs = os.listdir(path)
psnr_list = []
ssim_list = []
print(len(dirs))
for d in dirs:
    img0 = (path + d + '/frame1.png')
    img1 = (path + d + '/frame3.png')
    gt = (path + d + '/frame2.png')
    img0 = cv2.imread(img0)
    gt = cv2.imread(gt)
    img1 = cv2.imread(img1)
    img0, gt, img1 = center_crop(img0, gt, img1, 448, 896)
    #img0, gt, img1 = center_crop(img0, gt, img1, 224, 448)
    if count % save_interval == 0 and save_result:
        cv2.imwrite(os.path.join(result_dir, f'im_{count}_I0.png'), img0)
        cv2.imwrite(os.path.join(result_dir, f'im_{count}_I1.png'), gt)
        cv2.imwrite(os.path.join(result_dir, f'im_{count}_I2.png'), img1)
    img0 = (torch.tensor(img0.transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
    gt = (torch.tensor(gt.transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
    pred = model.inference(img0, img1)[0]
    ssim = ssim_matlab(gt, torch.round(pred * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
    out = pred.detach().cpu().numpy().transpose(1, 2, 0)
    if count % save_interval == 0 and save_result:
        mid_tmp = np.round(out * 255).astype('uint8')
        cv2.imwrite(os.path.join(result_dir, f'im_{count}_predict.png'), mid_tmp)
    out = np.round(out * 255) / 255.
    gt = gt[0].cpu().numpy().transpose(1, 2, 0)
    psnr = -10 * math.log10(((gt - out) * (gt - out)).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    print("Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))
    count = count + 1
