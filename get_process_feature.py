import os
import sys
import time
sys.path.append('.')
import cv2
import torch
import argparse
import numpy as np
from model.RIFE import Model


def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


save_path = 'inference_result/process/00072_0675'  # 保存预测结果


if not os.path.exists(save_path):
    os.mkdir(save_path)

# 准备模型
model_name = 'IFNet_bf_resnet_cbam_M'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(model_name=model_name)
model.load_model('train_log/IFNet_bf_resnet_cbam_M')
model.eval()
model.device()

# 加载输入帧图像
im0_path = 'E:\\Workspace\\Datasets\\vimeo_triplet\\sequences\\00072\\0675\\im1.png'
im1_path = 'E:\\Workspace\\Datasets\\vimeo_triplet\\sequences\\00072\\0675\\im3.png'
I0 = cv2.imread(im0_path)
I1 = cv2.imread(im1_path)
cv2.imwrite(os.path.join(save_path, f'I0.png'), I0)
cv2.imwrite(os.path.join(save_path, f'I1.png'), I1)

I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
I1 = (torch.tensor(I1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

merged, mask, before_res, res, mask_guide, flow = model.inference_with_visibility(I0, I1)
mid = merged[0]
mask = mask[0]
before_res = before_res[0]
res = res[0]
# 保存mask
mask_img = np.round((mask * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0)
cv2.imwrite(os.path.join(save_path, f'mask.png'), mask_img)


count = 0
for p_mask in mask_guide:
    p_mask = p_mask[0]
    mask_guide_tmp = np.round((p_mask * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0)
    cv2.imwrite(os.path.join(save_path, f'mask_guide_{count}.png'), mask_guide_tmp)
    count = count + 1

# 保存预测帧
mid_tmp = np.round((mid * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0)
cv2.imwrite(os.path.join(save_path, f'pred.png'), mid_tmp)

before_res_tmp = np.round((before_res * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0)
cv2.imwrite(os.path.join(save_path, f'merged_warped.png'), before_res_tmp)

res_tmp = np.round((res * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0)
cv2.imwrite(os.path.join(save_path, f'residual.png'), res_tmp)

# 处理光流并保存为RGB图像
flow = flow[2]
flow = flow.permute(0, 2, 3, 1).detach().cpu().numpy()
flow = flow[0]
flow_1 = flow[:, :, :2]
flow_2 = flow[:, :, 2:]
rgb_flow1 = flow2rgb(flow_1)  # 将光流转换为RGB图像
rgb_flow2 = flow2rgb(flow_2)  # 将光流转换为RGB图像
cv2.imwrite(os.path.join(save_path, f'flow_1.png'), (rgb_flow1 * 255).astype(np.uint8))  # 保存RGB图像
cv2.imwrite(os.path.join(save_path, f'flow_2.png'), (rgb_flow2 * 255).astype(np.uint8))  # 保存RGB图像
