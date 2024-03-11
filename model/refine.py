import os
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import itertools
from PIL import Image
import matplotlib.pyplot as plt
import functools

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


def sobel_edge_detection(image_tensor):
    # 使用Sobel算子对每个通道进行边缘检测
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
    sobel_y = sobel_x.T

    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).to(device)
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).to(device)

    edge_r_x = F.conv2d(image_tensor[:, 0:1, :, :], sobel_x, padding=1)
    edge_r_y = F.conv2d(image_tensor[:, 0:1, :, :], sobel_y, padding=1)

    edge_g_x = F.conv2d(image_tensor[:, 1:2, :, :], sobel_x, padding=1)
    edge_g_y = F.conv2d(image_tensor[:, 1:2, :, :], sobel_y, padding=1)

    edge_b_x = F.conv2d(image_tensor[:, 2:3, :, :], sobel_x, padding=1)
    edge_b_y = F.conv2d(image_tensor[:, 2:3, :, :], sobel_y, padding=1)

    edge_result = torch.cat([edge_r_x + edge_r_y, edge_g_x + edge_g_y, edge_b_x + edge_b_y], dim=1)

    return edge_result


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


class UnetCBAM_L(nn.Module):
    def __init__(self):
        super(UnetCBAM_L, self).__init__()
        self.down0 = Conv2(17, 2 * c, 1)
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
        self.conv = nn.Conv2d(c, 3, 3, 2, 1)

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


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.activation = nn.PReLU()
        # upsample layers
        self.upsample = functools.partial(F.interpolate, scale_factor=2)
        # Conv layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.learnable_sc = in_channels != out_channels or self.upsample
        if self.learnable_sc:
            self.conv_sc = nn.Conv2d(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        h = self.activation(self.bn1(x))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=nn.Conv2d, wide=True,
                 preactivation=False, down=True):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.down = down  # if downsample
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = nn.PReLU()
        self.downsample = nn.AvgPool2d(2)

        # Conv layers
        self.conv1 = nn.Conv2d(self.in_channels, self.hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_channels, self.out_channels,3, padding=1)
        self.learnable_sc = True if (in_channels != out_channels) or self.downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.down:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it
            #              will negatively affect the shortcut connection.
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.down:
            h = self.downsample(h)

        return h + self.shortcut(x)


class UnetCBAM_L_Res(nn.Module):
    """
    用于Larger模型，使用了CBAM, 使用Residual模块替代普通的卷积块
    """
    def __init__(self):
        super(UnetCBAM_L_Res, self).__init__()
        self.down0 = DBlock(17, 2 * c, down=False)
        self.down1 = DBlock(4 * c, 4 * c)
        self.down2 = DBlock(8 * c, 8 * c)
        self.down3 = DBlock(16 * c, 16 * c)

        self.cbam0 = CBAM(channels=2 * c)
        self.cbam1 = CBAM(channels=4 * c)
        self.cbam2 = CBAM(channels=8 * c)
        self.cbam3 = CBAM(channels=16 * c)

        self.up0 = GBlock(32 * c, 8 * c)
        self.up1 = GBlock(16 * c, 4 * c)
        self.up2 = GBlock(8 * c, 2 * c)
        self.up3 = GBlock(4 * c, c)
        self.conv = nn.Conv2d(c, 3, 3, 2, 1)

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


class UnetCBAM_L_M_pmask(nn.Module):
    """
    用于Large版本的模型，使用了CBAM， patch mask， 运动信息
    """
    def __init__(self):
        super(UnetCBAM_L, self).__init__()
        self.down0 = Conv2(17, 2 * c, 1)
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
        self.conv = nn.Conv2d(c, 3, 3, 2, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, edge, mo, guide_list):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
        s0 = self.cbam0(s0) + s0
        s1 = self.down1(torch.cat((guide_list[0], s0, c0[0], c1[0], ), 1))
        s1 = self.cbam1(s1) + s1
        s2 = self.down2(torch.cat((guide_list[0], s1, c0[1], c1[1]), 1))
        s2 = self.cbam2(s2) + s2
        s3 = self.down3(torch.cat((guide_list[0], s2, c0[2], c1[2]), 1))
        s3 = self.cbam3(s3) + s3
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return torch.sigmoid(x)


class Unet_FF(nn.Module):
    def __init__(self):
        super(Unet_FF, self).__init__()
        self.CatChannels = 4 * c
        self.CatBlocks = 4
        self.UpChannels = self.CatChannels * self.CatBlocks

        self.down0 = Conv2(17 + 6, 2 * c, 1)
        self.down1 = Conv2(4 * c, 4 * c)
        self.down2 = Conv2(8 * c + self.CatChannels, 8 * c)
        self.down3 = Conv2(16 * c + self.CatChannels, 16 * c)

        self.cbam0 = CBAM(channels=2 * c)
        self.cbam1 = CBAM(channels=4 * c)
        self.cbam2 = CBAM(channels=8 * c)
        self.cbam3 = CBAM(channels=16 * c)

        """全尺度融合部分"""
        self.conv = nn.Conv2d(self.UpChannels, 3, 3, 1, 1)
        self.h3_f = nn.Conv2d(256, self.CatChannels, 3, padding=1)
        self.m0_f = nn.Conv2d(360, self.CatChannels, 3, padding=1)
        self.m1_f = nn.Conv2d(225, self.CatChannels, 3, padding=1)
        self.m2_f = nn.Conv2d(135, self.CatChannels, 3, padding=1)
        # h0
        self.h0_3 = nn.Sequential(
            nn.MaxPool2d(8, 8, ceil_mode=True),
            nn.Conv2d(2 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h1_3 = nn.Sequential(
            nn.MaxPool2d(4, 4, ceil_mode=True),
            nn.Conv2d(4 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h2_3 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(8 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h3_3 = nn.Sequential(
            nn.Conv2d(16 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        # fusion 3
        self.fusion_3 = nn.Sequential(
            nn.Conv2d(self.UpChannels + self.CatChannels * 2, self.UpChannels, 3, padding=1),
            nn.BatchNorm2d(self.UpChannels),
            nn.PReLU(self.UpChannels)
        )

        # h1
        self.h0_2 = nn.Sequential(
            nn.MaxPool2d(4, 4, ceil_mode=True),
            nn.Conv2d(2 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h1_2 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(4 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h2_2 = nn.Sequential(
            nn.Conv2d(8 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h3_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.fusion_2 = nn.Sequential(
            nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1),
            nn.BatchNorm2d(self.UpChannels),
            nn.PReLU(self.UpChannels)
        )

        # h2
        self.h0_1 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(2 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h1_1 = nn.Sequential(
            nn.Conv2d(4 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h2_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h3_1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.fusion_1 = nn.Sequential(
            nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1),
            nn.BatchNorm2d(self.UpChannels),
            nn.PReLU(self.UpChannels)
        )

        # h3
        self.h0_0 = nn.Sequential(
            nn.Conv2d(2 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h1_0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h2_0 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h3_0 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.fusion_0 = nn.Sequential(
            nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1),
            nn.BatchNorm2d(self.UpChannels),
            nn.PReLU(self.UpChannels)
        )

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, edge, mo):
        m0 = self.m0_f(mo[0])
        m1 = self.m1_f(mo[1])
        m2 = self.m2_f(mo[2])
        f3 = self.h3_f(torch.cat((c0[3], c1[3]), 1))
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow, edge), 1))
        s0 = s0 + self.cbam0(s0)
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))  # W:256
        s1 = s1 + self.cbam1(s1)
        s2 = self.down2(torch.cat((s1, c0[1], c1[1], m2), 1))  # W:128
        s2 = s2 + self.cbam2(s2)
        s3 = self.down3(torch.cat((s2, c0[2], c1[2], m1), 1))  # W:64
        s3 = s3 + self.cbam3(s3)

        fa3 = self.fusion_3(torch.cat((self.h0_3(s0), self.h1_3(s1), self.h2_3(s2), self.h3_3(s3), f3, m0), 1))
        fa2 = self.fusion_2(torch.cat((self.h0_2(s0), self.h1_2(s1), self.h2_2(s2), self.h3_2(fa3)), 1))
        fa1 = self.fusion_1(torch.cat((self.h0_1(s0), self.h1_1(s1), self.h2_1(fa2), self.h3_1(fa3)), 1))
        fa0 = self.fusion_0(torch.cat((self.h0_0(s0), self.h1_0(fa1), self.h2_0(fa2), self.h3_0(fa3)), 1))

        x = self.conv(fa0)
        return torch.sigmoid(x)


class Unet_FF_M(nn.Module):
    def __init__(self):
        super(Unet_FF_M, self).__init__()
        self.CatChannels = 4 * c
        self.CatBlocks = 4
        self.UpChannels = self.CatChannels * self.CatBlocks

        self.down0 = Conv2(17 + 6, 2 * c, 1)
        self.down1 = Conv2(4 * c + 1, 4 * c)
        self.down2 = Conv2(8 * c + 1 + self.CatChannels, 8 * c)
        self.down3 = Conv2(16 * c + 1 + self.CatChannels, 16 * c)

        self.cbam0 = CBAM(channels=2 * c)
        self.cbam1 = CBAM(channels=4 * c)
        self.cbam2 = CBAM(channels=8 * c)
        self.cbam3 = CBAM(channels=16 * c)

        """全尺度融合部分"""
        self.conv = nn.Conv2d(self.UpChannels, 3, 3, 1, 1)
        self.h3_f = nn.Conv2d(256, self.CatChannels, 3, padding=1)
        self.m0_f = nn.Conv2d(360, self.CatChannels, 3, padding=1)
        self.m1_f = nn.Conv2d(225, self.CatChannels, 3, padding=1)
        self.m2_f = nn.Conv2d(135, self.CatChannels, 3, padding=1)
        # h0
        self.h0_3 = nn.Sequential(
            nn.MaxPool2d(8, 8, ceil_mode=True),
            nn.Conv2d(2 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h1_3 = nn.Sequential(
            nn.MaxPool2d(4, 4, ceil_mode=True),
            nn.Conv2d(4 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h2_3 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(8 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h3_3 = nn.Sequential(
            nn.Conv2d(16 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        # fusion 3
        self.fusion_3 = nn.Sequential(
            nn.Conv2d(self.UpChannels + self.CatChannels * 2, self.UpChannels, 3, padding=1),
            nn.BatchNorm2d(self.UpChannels),
            nn.PReLU(self.UpChannels)
        )

        # h1
        self.h0_2 = nn.Sequential(
            nn.MaxPool2d(4, 4, ceil_mode=True),
            nn.Conv2d(2 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h1_2 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(4 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h2_2 = nn.Sequential(
            nn.Conv2d(8 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h3_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.fusion_2 = nn.Sequential(
            nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1),
            nn.BatchNorm2d(self.UpChannels),
            nn.PReLU(self.UpChannels)
        )

        # h2
        self.h0_1 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(2 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h1_1 = nn.Sequential(
            nn.Conv2d(4 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h2_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h3_1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.fusion_1 = nn.Sequential(
            nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1),
            nn.BatchNorm2d(self.UpChannels),
            nn.PReLU(self.UpChannels)
        )

        # h3
        self.h0_0 = nn.Sequential(
            nn.Conv2d(2 * c, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h1_0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h2_0 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.h3_0 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1),
            nn.BatchNorm2d(self.CatChannels),
            nn.PReLU(self.CatChannels)
        )
        self.fusion_0 = nn.Sequential(
            nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1),
            nn.BatchNorm2d(self.UpChannels),
            nn.PReLU(self.UpChannels)
        )

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, edge, mo, guide_list):
        m0 = self.m0_f(mo[0])
        m1 = self.m1_f(mo[1])
        m2 = self.m2_f(mo[2])
        f3 = self.h3_f(torch.cat((c0[3], c1[3]), 1))
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow, edge), 1))
        s0 = s0 + self.cbam0(s0)
        s1 = self.down1(torch.cat((s0, guide_list[0], c0[0], c1[0]), 1))  # W:256
        s1 = s1 + self.cbam1(s1)
        s2 = self.down2(torch.cat((s1, guide_list[1], c0[1], c1[1], m2), 1))  # W:128
        s2 = s2 + self.cbam2(s2)
        s3 = self.down3(torch.cat((s2, guide_list[2], c0[2], c1[2], m1), 1))  # W:64
        s3 = s3 + self.cbam3(s3)

        fa3 = self.fusion_3(torch.cat((self.h0_3(s0), self.h1_3(s1), self.h2_3(s2), self.h3_3(s3), f3, m0), 1))
        fa2 = self.fusion_2(torch.cat((self.h0_2(s0), self.h1_2(s1), self.h2_2(s2), self.h3_2(fa3)), 1))
        fa1 = self.fusion_1(torch.cat((self.h0_1(s0), self.h1_1(s1), self.h2_1(fa2), self.h3_1(fa3)), 1))
        fa0 = self.fusion_0(torch.cat((self.h0_0(s0), self.h1_0(fa1), self.h2_0(fa2), self.h3_0(fa3)), 1))

        x = self.conv(fa0)
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
    def __init__(self, patch_size=8, mask_min=0.15, mask_max=0.85, rec_threshold=7, patch_thr=16, img_size=56):
        super(LocalMae, self).__init__()
        if img_size == 56:
            self.mae_vit = mae_vit_spe_base_patch8_tiny()
        elif img_size == 112:
            self.mae_vit = mae_vit_spe_base_patch16_dec512d8b()
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.patch_size = patch_size
        self.window_size = img_size
        self.patch_thr = patch_thr  # 一个patch中有多少像素符合条件才重建
        self.rec_threshold = rec_threshold  # 最少n个块就重建

    def sliding_window(self, imgs, target, rec_region, window_size):
        batch_size, _, height, width = imgs.size()
        window_h = height // window_size
        window_w = width // window_size
        loss_sum = 0.
        window_rec_num = 0

        for b in range(batch_size):
            for i in range(window_h):
                for j in range(window_w):
                    # Extract window from rec_region
                    window_rec_region = rec_region[b:b + 1, :, i * window_size:(i + 1) * window_size,
                                        j * window_size:(j + 1) * window_size]

                    # 判断是否需要重建
                    is_reconstruct = is_rec_window(window_rec_region, self.patch_size ** 2 * self.rec_threshold)
                    if is_reconstruct:
                        img_window = imgs[b, :, i * window_size:(i + 1) * window_size,
                                     j * window_size:(j + 1) * window_size]
                        img_window = img_window.unsqueeze(0)
                        target_window = target[b, :, i * window_size:(i + 1) * window_size,
                                        j * window_size:(j + 1) * window_size]
                        target_window = target_window.unsqueeze(0)
                        patch_mask, _ = get_rec_patches(window_rec_region, patch_size=self.patch_size,
                                                        threshold=self.patch_thr)
                        pred_window, window_loss = self.mae_vit(img_window, patch_mask, target_window)
                        window_rec_num = window_rec_num + 1
                        # Replace corresponding window in imgs with processed window
                        imgs[b, :, i * window_size:(i + 1) * window_size,
                        j * window_size:(j + 1) * window_size] = img_window * (
                                1 - window_rec_region) + pred_window * window_rec_region
                        loss_sum = loss_sum + window_loss
        print(f"reconstruct {window_rec_num} window in 1 batch")
        if window_rec_num == 0:
            loss_avg = 0.0
        else:
            loss_avg = loss_sum / window_rec_num
        return imgs, loss_avg

    def forward(self, imgs, mask, target):
        B, C, H, W = imgs.shape

        # 预处理阶段，根据mask确定重建的范围
        mask_region = get_rec_region(mask, self.mask_min, self.mask_max)
        # count_ones = torch.sum(mask_region[0:1,:,:,:] == 1).item()
        # print(f"mask[0]中值为1的像素点: {count_ones} {count_ones/(H*W)}")
        mask_patch, window_region = get_rec_patches(mask_region, patch_size=self.patch_size,
                                                    threshold=self.rec_threshold)
        mask_patch, window_region = get_rec_patches(mask_region, patch_size=self.patch_size, threshold=self.patch_thr)

        imgs_reconstructed, loss_sum = self.sliding_window(imgs, target, window_region, self.window_size)

        return imgs_reconstructed, loss_sum


def get_local_mae_patch_8():
    return LocalMae(patch_size=8, mask_min=0.15, mask_max=0.85, rec_threshold=7, patch_thr=16, img_size=56)


def get_local_mae_patch_16():
    return LocalMae(patch_size=16, mask_min=0.15, mask_max=0.85, rec_threshold=4, patch_thr=16, img_size=112)


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

    result_dir = 'test_result'

    model = get_local_mae_patch_16()
    output_tensor, loss = model(I0, mask_img, target)
    for i in range(1):
        reconstructed_img = np.round((output_tensor * 255)[i].detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0)
        # mask_img = mask_img[:, :224, :]
        cv2.imwrite(os.path.join(result_dir, f'reconstructed_img_{i}.png'), reconstructed_img)
    print('ok')
