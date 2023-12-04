import os

import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import itertools
from PIL import Image
import matplotlib.pyplot as plt

from model.lomar.models_lomar import MaskedAutoencoderViT, mae_vit_base_patch16
from model.lomar.models_vit import vit_base_patch16_decoder
from model.mae.models_mae import mae_vit_spe_base_patch16_dec512d8b, mae_vit_spe_base_patch8_dec512d8b, \
    mae_vit_spe_base_patch8_tiny

from model.warplayer import warp
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1,
                                 bias=True),
        nn.PReLU(out_planes)
    )


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


c = 16


class Contextnet(nn.Module):
    def __init__(self):
        super(Contextnet, self).__init__()
        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2 * c)
        self.conv3 = Conv2(2 * c, 4 * c)
        self.conv4 = Conv2(4 * c, 8 * c)

    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False,
                             recompute_scale_factor=False) * 0.5
        f1 = warp(x, flow)
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False,
                             recompute_scale_factor=False) * 0.5
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False,
                             recompute_scale_factor=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False,
                             recompute_scale_factor=False) * 0.5
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down0 = Conv2(17, 2 * c)
        self.down1 = Conv2(4 * c, 4 * c)
        self.down2 = Conv2(8 * c, 8 * c)
        self.down3 = Conv2(16 * c, 16 * c)
        self.up0 = deconv(32 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return torch.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = x * out
        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        out = x * out
        return out


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()

        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out


class UnetCBAM(nn.Module):
    def __init__(self):
        super(UnetCBAM, self).__init__()
        self.down0 = Conv2(17, 2 * c)
        self.down1 = Conv2(4 * c, 4 * c)
        self.down2 = Conv2(8 * c, 8 * c)
        self.down3 = Conv2(16 * c, 16 * c)

        self.cbam0 = CBAM(channels=2 * c)
        self.cbam1 = CBAM(channels=4 * c)
        self.cbam2 = CBAM(channels=8 * c)
        self.cbam3 = CBAM(channels=16 * c)

        self.up0 = deconv(32 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
        s0 = self.cbam0(s0) + s0
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s1 = self.cbam1(s1) + s1
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s2 = self.cbam2(s2) + s2
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        s3 = self.cbam3(s3) + s3
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return torch.sigmoid(x)


class UNetMAEViT(nn.Module):
    def __init__(self):
        super(UNetMAEViT, self).__init__()
        self.unet_cbam = UnetCBAM()
        self.mae_vit = mae_vit_base_patch16()
        self.decoder = vit_base_patch16_decoder()

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        # U-Net 阶段
        unet_output = self.unet_cbam(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)

        # MAE ViT 编码阶段
        mae_vit_loss, mae_vit_output, mask_indices_encoder = self.mae_vit(unet_output)

        # 解码阶段
        decode_out = self.decoder.forward_features(mae_vit_output)
        mae_img_pred = self.mae_vit.unpatchify(decode_out)  # 解码出完整图片
        mae_img_pred = torch.sigmoid(mae_img_pred)  # 变正数符合像素要求

        # 可以根据需要返回 U-Net 输出、MAE ViT 输出等
        return unet_output, mae_img_pred, mae_vit_output, mae_vit_loss, mask_indices_encoder


def get_rec_region(tensor, low, high):
    # 创建一个和输入张量相同大小的零张量
    rec_region = torch.zeros_like(tensor)

    # 使用逻辑运算设置满足条件的像素点为1，表示需要重建的区域，其他部分为0
    rec_region[(tensor < low) | (tensor > high)] = 1

    return rec_region


def get_rec_patches(mask, patch_size=8, threshold=16):
    """
    根据 mask 产生 patch mask
    :param mask: 遮挡区域，1表示遮挡
    :param patch_size: patch_size大小
    :param threshold: 表示一个patch_size内最少有多少个需重建的像素点才重建
    :return: window_mark
    """
    B, C, H, W = mask.shape
    _, _, mask_H, mask_W = mask.shape
    num_patch = int((H / patch_size) * (W / patch_size))

    # 使用平均池化对 mask 进行窗口划分
    pool = F.avg_pool2d(mask.float(), patch_size, stride=patch_size, padding=0)
    # window_mark = torch.zeros_like(pool)
    # 将窗口内大于等于 threshold 的部分标记为 1
    window_mark = (pool >= threshold / (patch_size ** 2)).float()
    window_region = F.interpolate(window_mark, scale_factor=patch_size, mode='nearest')
    window_mark = window_mark.view(B, num_patch)
    return window_mark, window_region


def is_rec_window(img_patch, min_pix=512):
    return torch.sum(img_patch) > min_pix


class LocalMae(nn.Module):
    def __init__(self, patch_size=8, mask_min=0.15, mask_max=0.85, rec_threshold=16):
        super(LocalMae, self).__init__()
        self.mae_vit = mae_vit_spe_base_patch8_tiny()
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.patch_size = patch_size
        self.window_size = 56
        self.rec_threshold = rec_threshold  # 最少n个像素就重建

    def sliding_window(self, imgs, target, rec_region, window_size):
        batch_size, _, height, width = imgs.size()
        window_h = height // window_size
        window_w = width // window_size
        loss_sum = 0.

        for b in range(batch_size):
            for i in range(window_h):
                for j in range(window_w):
                    # Extract window from rec_region
                    window_rec_region = rec_region[b:b + 1, :, i * window_size:(i + 1) * window_size,
                                        j * window_size:(j + 1) * window_size]

                    # 判断是否需要重建
                    is_reconstruct = is_rec_window(window_rec_region)
                    if is_reconstruct:
                        img_window = imgs[b, :, i * window_size:(i + 1) * window_size,
                                     j * window_size:(j + 1) * window_size]
                        img_window = img_window.unsqueeze(0)
                        target_window = target[b, :, i * window_size:(i + 1) * window_size,
                                     j * window_size:(j + 1) * window_size]
                        target_window = target_window.unsqueeze(0)
                        patch_mask, _ = get_rec_patches(window_rec_region, patch_size=self.patch_size)
                        pred_window, window_loss = self.mae_vit(img_window, patch_mask, target_window)
                        # Replace corresponding window in imgs with processed window
                        imgs[b, :, i * window_size:(i + 1) * window_size,
                        j * window_size:(j + 1) * window_size] = pred_window
                        loss_sum = loss_sum + window_loss

        return imgs, loss_sum

    def forward(self, imgs, mask, target):
        B, C, H, W = imgs.shape

        # 预处理阶段，根据mask确定重建的范围
        mask_region = get_rec_region(mask, self.mask_min, self.mask_max)
        mask_patch, window_region = get_rec_patches(mask_region, patch_size=self.patch_size)

        imgs_reconstructed, loss_sum = self.sliding_window(imgs, target, window_region, self.window_size)

        return imgs_reconstructed, loss_sum


def get_local_mae_patch_8():
    return LocalMae(patch_size=8, mask_min=0.15, mask_max=0.85, rec_threshold=16)


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip(image * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


if __name__ == "__main__":
    """ 测试lomar 模型的图像输入和重建 """
    # 准备模型和图像数据
    # chkpt_dir = '../lomar_base.pth'
    # model = mae_vit_base_patch16()
    # decoder = vit_base_patch16_decoder()
    # # checkpoint = torch.load(chkpt_dir, map_location='cpu')
    # # msg = model.load_state_dict(checkpoint['model'], strict=False)
    # # print(msg)
    #
    # img = Image.open('../demo/im_1350_I1_predict.png')
    # img = img.resize((224, 224))
    # img = np.array(img) / 255.
    # img = img.astype('float32')
    # img2 = Image.open('../demo/im_0_I1_predict.png').resize((224, 224))
    # img2 = np.array(img2) / 255.
    # img2 = img2.astype('float32')
    # x = torch.tensor(img)
    # x = x.unsqueeze(dim=0)
    # x2 = torch.tensor(img2)
    # x2 = x2.unsqueeze(dim=0)
    #
    # x = torch.cat([x, x2], dim=0)
    # x = torch.einsum('nhwc->nchw', x)
    #
    # # input = torch.rand(1, 9, 224, 224)
    # loss, pred, mask_indices = model(x)
    #
    # mae_pred = pred.reshape(-1, 196, 768)
    # dec = decoder.forward_features(mae_pred)
    # mae_img_pred = model.unpatchify(dec)
    #
    # plt.subplot(1, 2, 1)
    # show_image(x.permute(0, 2, 3, 1)[0], "original")
    #
    # plt.subplot(1, 2, 2)
    # show_image(mae_img_pred.permute(0, 2, 3, 1)[0], "reconstruction")
    # # show_image(mae_img_pred[0].permute(1, 2, 0), "reconstruction")
    #
    # print('ok')

    device = 'cpu'
    img_root = r'E:\WorkSpace\python\DeepLearning\RIFE-biformer\inference_result\vimeo90k\IFNet_bf_resnet_cbam_2023-11-30_19_15_44'
    mask_root = r'E:\WorkSpace\python\DeepLearning\RIFE-biformer\inference_result\mask'
    img_name = 'im_3400_I1_pred.png'
    target_name = 'im_3400_target_I1.png'
    mask_name = 'im_3400_I1_mask.png'
    I0 = cv2.imread(os.path.join(img_root, img_name))
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    I0 = I0[:, :, :, :224]
    target = cv2.imread(os.path.join(img_root, img_name))
    target = (torch.tensor(target.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    target = target[:, :, :, :224]
    mask_img = cv2.imread(os.path.join(img_root, mask_name))
    mask_img = (torch.tensor(mask_img.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    mask_img = (mask_img[:, 0, :, :]).unsqueeze(0)
    mask_img = mask_img[:, :, :, :224]

    model = LocalMae(patch_size=8)
    out = model(I0, mask_img, target)
    print('ok')

