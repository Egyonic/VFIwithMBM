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
from lpips import LPIPS
from PIL import Image
from torchvision.transforms.functional import resize


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


def resize_to_multiple_of_112(img, multiple=112):
    h, w, _ = img.shape
    new_h = ((h - 1) // multiple + 1) * multiple
    new_w = ((w - 1) // multiple + 1) * multiple
    resized_img = cv2.resize(img, (new_w, new_h))
    return resized_img  # 返回调整后的图像


save_result = True
save_original = False
path = '/home/usst/egy/data/vimeo90k/'
f = open(path + 'tri_testlist.txt', 'r')
save_path = 'inference_result'  # 保存预测结果
timestruct = time.localtime(time.time())
t_s = time.strftime('%Y-%m-%d_%H_%M_%S', timestruct)
result_dir = os.path.join(save_path, 'vimeo90k',t_s)
if save_result:
    os.mkdir(result_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(model_name='IFNet_bf_resnet')
model.load_model('train_log/bf_4b_resnte_bi_rrrb')
model.eval()
model.device()
psnr_list = []
ssim_list = []
lpips_list = []
psnr_ori_list = []
ssim_ori_list = []
lpips_ori_list = []

# 加载LPIPS模型
lpips_net = LPIPS(net='vgg').to('cuda')  # 使用GPU加速

count = 0
save_interval = 50

for i in f:
    name = str(i).strip()
    if(len(name) <= 1):
        continue
    print(path + 'target/' + name + '/im1.png')
    I0 = cv2.imread(path + 'target/' + name + '/im1.png')
    I1 = cv2.imread(path + 'target/' + name + '/im2.png')
    I2 = cv2.imread(path + 'target/' + name + '/im3.png')

    # 保存原始图像用于对比
    if save_original and count % save_interval == 0:
        cv2.imwrite(os.path.join(result_dir, f'im_{count}_raw_I0.png'), I0)
        cv2.imwrite(os.path.join(result_dir, f'im_{count}_raw_tg_I1.png'), I1)
        cv2.imwrite(os.path.join(result_dir, f'im_{count}_raw_I2.png'), I2)

    original_h, original_w, _ = I0.shape  # 记录原始尺寸
    # I0, I1, I2 = center_crop(I0, I1, I2, 224, 448) # 通过裁剪将图像大小调整为适合Transformer的输入大小
    # 通过resize将图像大小调整为适合Transformer的输入大小
    I0_resize = resize_to_multiple_of_112(I0)
    I1_resize = resize_to_multiple_of_112(I1)
    I2_resize = resize_to_multiple_of_112(I2)

    # 保存需要在结果中查看的图像
    if save_result and count % save_interval == 0:
        cv2.imwrite(os.path.join(result_dir, f'im_{count}_resize_I0.png'), I0_resize)
        cv2.imwrite(os.path.join(result_dir, f'im_{count}_resize_tg_I1.png'), I1_resize)
        cv2.imwrite(os.path.join(result_dir, f'im_{count}_resize_I2.png'), I2_resize)

    I0_resize = (torch.tensor(I0_resize.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    I2_resize = (torch.tensor(I2_resize.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    mid = model.inference(I0_resize, I2_resize)[0]
    # 将 mid 的尺寸转换回原始的输入尺寸
    mid_resize = resize(mid, (original_h, original_w))

    ssim = ssim_matlab(torch.tensor(I1_resize.transpose(2, 0, 1)).to(device).unsqueeze(0) / 255., torch.round(mid * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
    ssim_re = ssim_matlab(torch.tensor(I1.transpose(2, 0, 1)).to(device).unsqueeze(0) / 255., torch.round(mid_resize * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
    # 计算LPIPS相似性分数
    lpips_score = lpips_net((torch.tensor(I1_resize.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0), mid)
    lpips_score2 = lpips_net((torch.tensor(I1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0), mid_resize)
    lpips_list.append(lpips_score.item())  # 将分数添加到列表中
    lpips_ori_list.append(lpips_score2.item())  # 将分数添加到列表中
    # mid = np.round((mid * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.
    if count % save_interval == 0 and save_result:
        mid_tmp = np.round((mid * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0)
        cv2.imwrite(os.path.join(result_dir, f'im_{count}_resize_I1_predict.png'), mid_tmp)  # 保存预测图

        mid_resize_tmp = np.round((mid_resize * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0)
        cv2.imwrite(os.path.join(result_dir, f'im_{count}_raw_I1_predict.png'), mid_resize_tmp)
    mid = np.round((mid * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.
    mid_resize = np.round((mid_resize * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.
    I1_resize = I1_resize / 255.
    I1 = I1 / 255.
    psnr = -10 * math.log10(((I1_resize - mid) * (I1_resize - mid)).mean())
    psnr2 = -10 * math.log10(((I1 - mid_resize) * (I1 - mid_resize)).mean())

    psnr_list.append(psnr)
    ssim_list.append(ssim)
    psnr_ori_list.append(psnr2)
    ssim_ori_list.append(ssim_re)
    print("同尺寸 Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))
    print("转换尺寸 Avg PSNR: {} SSIM: {}".format(np.mean(psnr_ori_list), np.mean(ssim_ori_list)))
    count = count + 1

print('预测结果尺寸不变')
print("平均 PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))
print(f'验证集平均LPIPS相似性分数为: {np.mean(lpips_list)}')
print('\n resize预测结果回原尺寸')
print("平均 PSNR: {} SSIM: {}".format(np.mean(psnr_ori_list), np.mean(ssim_ori_list)))
print(f'验证集平均LPIPS相似性分数为: {np.mean(lpips_ori_list)}')
