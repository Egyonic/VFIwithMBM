import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.refine import *
from model.refine_tf import Restormer
from model.biformer_models.biformer import Block as BiformerBlock
from model.resnet import resnet50_feature, resnet50_feature_L

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.tran_conv = nn.Sequential(
            conv(in_planes, 64, 3, 1, 1),
        )
        self.tf_conv_revert = nn.Sequential(
            conv(64, in_planes, 3, 1, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)
        self.tf_block = BiformerBlock(
            dim=64,
            n_win=7,
            num_heads=4,
            kv_downsample_mode='identity',
            kv_per_win=-1,
            topk=4,
            mlp_ratio=3,
            side_dwconv=5,
            before_attn_dwconv=3,
            layer_scale_init_value=-1,
            qk_dim=32,
            param_routing=False, diff_routing=False, soft_routing=False,
            pre_norm=True, auto_pad=True)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.tran_conv(x)
        x = self.tf_block(x)
        x = self.tf_conv_revert(x)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask


class IFBlock_bf(nn.Module):
    """
    使用了 BiFormer Block 的IFBlock， 采用金字塔结构提取三层特征叠加
    """

    def __init__(self, in_planes, c=64, tf_dim=64, n_win=14, n_block=4, kv_downsample_mode='identity', topk=4,
                 mlp_ratio=3, num_heads=4, kv_per_win=2):
        super(IFBlock_bf, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.tf_conv = nn.Sequential(
            conv(c, tf_dim, 3, 1, 1),
        )
        self.tf_conv_revert = nn.Sequential(
            conv(tf_dim, c, 3, 1, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)
        biformer_blocks = []
        for i in range(n_block):
            biformer_blocks.append(BiformerBlock(
                dim=tf_dim, n_win=n_win, num_heads=num_heads, kv_downsample_mode=kv_downsample_mode, kv_per_win=kv_per_win,
                topk=topk, mlp_ratio=mlp_ratio, side_dwconv=5, before_attn_dwconv=3, layer_scale_init_value=-1,
                qk_dim=tf_dim, param_routing=False, diff_routing=False, soft_routing=False, pre_norm=True,
                auto_pad=True))
        self.tf_block = nn.Sequential(*biformer_blocks)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        y = self.tf_conv(x)
        y = self.tf_block(y)
        y = self.tf_conv_revert(y)
        tmp = self.lastconv(x) + self.lastconv(y)
        # 上采样
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask


class IFBlock_bf_L(nn.Module):
    def __init__(self, in_planes, c=64, tf_dim=64, n_win=7, n_block=4):
        super(IFBlock_bf_L, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 1, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.tf_conv = nn.Sequential(
            conv(c, tf_dim, 3, 2, 1),
        )
        self.tf_conv_revert = nn.Sequential(
            deconv(tf_dim, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)
        biformer_blocks = []
        for i in range(n_block):
            biformer_blocks.append(BiformerBlock(
                dim=tf_dim, n_win=n_win, num_heads=4, kv_downsample_mode='identity', kv_per_win=-1,
                topk=4, mlp_ratio=3, side_dwconv=5, before_attn_dwconv=3, layer_scale_init_value=-1,
                qk_dim=tf_dim, param_routing=False, diff_routing=False, soft_routing=False, pre_norm=True,
                auto_pad=True))
        self.tf_block = nn.Sequential(*biformer_blocks)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        y = self.tf_conv(x)
        y = self.tf_block(y)
        y = self.tf_conv_revert(y)
        tmp = self.lastconv(x) + self.lastconv(y)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask


class IFBlock_bf_H(nn.Module):
    def __init__(self, in_planes, c=64, tf_dim=64, n_win=7, n_block=4):
        super(IFBlock_bf_H, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.tf_conv = nn.Sequential(
            conv(c, tf_dim, 3, 1, 1),
        )
        self.tf_conv_revert = nn.Sequential(
            conv(tf_dim, c, 3, 1, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)
        biformer_blocks = []
        for i in range(n_block):
            biformer_blocks.append(BiformerBlock(
                dim=tf_dim, n_win=n_win, num_heads=4, kv_downsample_mode='ada_avgpool', kv_per_win=4,
                topk=4, mlp_ratio=3, side_dwconv=5, before_attn_dwconv=3, layer_scale_init_value=-1,
                qk_dim=tf_dim, param_routing=False, diff_routing=False, soft_routing=False, pre_norm=True,
                auto_pad=True))
        self.tf_block = nn.Sequential(*biformer_blocks)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        y = self.tf_conv(x)
        y = self.tf_block(y)
        y = self.tf_conv_revert(y)
        tmp = self.lastconv(x) + self.lastconv(y)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask, y


class IFBlock_bf_H_v2(nn.Module):
    def __init__(self, in_planes, c=64, tf_dim=64, n_win=7, n_block=4, kv_downsample_mode='identity', topk=4,
                 mlp_ratio=3, num_heads=4, kv_per_win=2):
        super(IFBlock_bf_H_v2, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.tf_conv = nn.Sequential(
            conv(c, tf_dim, 3, 1, 1),
        )
        self.tf_conv_revert = nn.Sequential(
            conv(tf_dim, c, 3, 1, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)
        biformer_blocks = []
        for i in range(n_block):
            biformer_blocks.append(BiformerBlock(
                dim=tf_dim, n_win=n_win, num_heads=num_heads, kv_downsample_mode=kv_downsample_mode, kv_per_win=kv_per_win,
                topk=topk, mlp_ratio=mlp_ratio, side_dwconv=5, before_attn_dwconv=3, layer_scale_init_value=-1,
                qk_dim=tf_dim, param_routing=False, diff_routing=False, soft_routing=False, pre_norm=True,
                auto_pad=True))
        self.tf_block = nn.Sequential(*biformer_blocks)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        y = self.tf_conv(x)
        y = self.tf_block(y)
        y = self.tf_conv_revert(y)
        tmp = self.lastconv(x) + self.lastconv(y)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask, y


class IFBlock_bf_H_L(nn.Module):
    def __init__(self, in_planes, c=64, tf_dim=64, n_win=7, n_block=4, topk=4):
        super(IFBlock_bf_H_L, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 1, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.tf_conv = nn.Sequential(
            conv(c, tf_dim, 3, 2, 1),
        )
        self.tf_conv_revert = nn.Sequential(
            deconv(tf_dim, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)
        biformer_blocks = []
        for i in range(n_block):
            biformer_blocks.append(BiformerBlock(
                dim=tf_dim, n_win=n_win, num_heads=4, kv_downsample_mode='identity', kv_per_win=-1,
                topk=topk, mlp_ratio=3, side_dwconv=5, before_attn_dwconv=3, layer_scale_init_value=-1,
                qk_dim=tf_dim, param_routing=False, diff_routing=False, soft_routing=False, pre_norm=True,
                auto_pad=True))
        self.tf_block = nn.Sequential(*biformer_blocks)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        y = self.tf_conv(x)
        y = self.tf_block(y)
        y = self.tf_conv_revert(y)
        if y.shape[2] > x.shape[2]:
            y = y[:, :, :x.shape[2], :]
        tmp = self.lastconv(x) + self.lastconv(y)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask, y


class IFBlock_bf_H_L_v2(nn.Module):
    def __init__(self, in_planes, c=64, tf_dim=64, n_win=7, n_block=4, kv_downsample_mode='identity', topk=4,
                 mlp_ratio=3, num_heads=4, kv_per_win=2):
        super(IFBlock_bf_H_L_v2, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 1, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.tf_conv = nn.Sequential(
            conv(c, tf_dim, 3, 2, 1),
        )
        self.tf_conv_revert = nn.Sequential(
            deconv(tf_dim, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)
        biformer_blocks = []
        for i in range(n_block):
            biformer_blocks.append(BiformerBlock(
                dim=tf_dim, n_win=n_win, num_heads=num_heads, kv_downsample_mode=kv_downsample_mode,
                kv_per_win=kv_per_win,
                topk=topk, mlp_ratio=mlp_ratio, side_dwconv=5, before_attn_dwconv=3, layer_scale_init_value=-1,
                qk_dim=tf_dim, param_routing=False, diff_routing=False, soft_routing=False, pre_norm=True,
                auto_pad=True))
        self.tf_block = nn.Sequential(*biformer_blocks)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        y = self.tf_conv(x)
        y = self.tf_block(y)
        y = self.tf_conv_revert(y)
        if y.shape[2] > x.shape[2]:
            y = y[:, :, :x.shape[2], :]
        tmp = self.lastconv(x) + self.lastconv(y)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask, y


class IFBlock_bf_L_M(nn.Module):
    def __init__(self, in_planes, c=64, tf_dim=64, n_win=7):
        super(IFBlock_bf_L_M, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 1, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.tf_conv = nn.Sequential(
            conv(c, tf_dim, 3, 2, 1),
        )
        self.tf_conv_revert = nn.Sequential(
            deconv(tf_dim, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)
        biformer_blocks = []
        for i in range(4):
            biformer_blocks.append(BiformerBlock(
                dim=tf_dim, n_win=n_win, num_heads=4, kv_downsample_mode='identity', kv_per_win=-1,
                topk=4, mlp_ratio=2, side_dwconv=5, before_attn_dwconv=3, layer_scale_init_value=-1,
                qk_dim=tf_dim, param_routing=False, diff_routing=False, soft_routing=False, pre_norm=True,
                auto_pad=True))
        self.tf_block = nn.Sequential(*biformer_blocks)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        y = self.tf_conv(x)
        y = self.tf_block(y)
        y = self.tf_conv_revert(y)
        tmp = self.lastconv(x) + self.lastconv(y)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask, y


class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(13 + 4, c=150)
        self.block2 = IFBlock(13 + 4, c=90)
        self.block_tea = IFBlock(16 + 4, c=90)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                        scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        # 在训练时，有参考帧，教师网络使用参考帧进行光流计算
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                            scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        # 融合多个信息
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill


class IFNet_bf(nn.Module):
    def __init__(self):
        super(IFNet_bf, self).__init__()
        self.block0 = IFBlock_bf(6, c=240, tf_dim=64, n_win=14)
        self.block1 = IFBlock_bf(13 + 4, c=150, tf_dim=64, n_win=14)
        self.block2 = IFBlock_bf(13 + 4, c=90, tf_dim=128, n_win=14)
        self.block_tea = IFBlock_bf(16 + 4, c=90, tf_dim=128)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        # print(img0.shape)
        # 学生网络
        stu = [self.block0, self.block1, self.block2]
        # 三层IFBlock计算累加
        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                        scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        # 在训练时，有参考帧，教师网络使用参考帧进行光流计算
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                            scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        # 融合多个信息
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill


class IFNet_bf_resnet_cbam(nn.Module):
    def __init__(self):
        super(IFNet_bf_resnet_cbam, self).__init__()
        self.block0 = IFBlock_bf(6, c=240, tf_dim=64)
        self.block1 = IFBlock_bf(13 + 4, c=150, tf_dim=64)
        self.block2 = IFBlock_bf(13 + 4, c=90, tf_dim=128)
        self.block_tea = IFBlock_bf(16 + 4, c=90, tf_dim=128)
        # self.contextnet = Contextnet()
        # 使用 resnet结构提取特征
        self.contextnet = resnet50_feature()
        # 使用添加了CBAM的Unet
        self.unet = UnetCBAM()

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        # print(img0.shape)
        # 学生网络
        stu = [self.block0, self.block1, self.block2]
        # 三层IFBlock计算累加
        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                        scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        # 在训练时，有参考帧，教师网络使用参考帧进行光流计算
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                            scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        # 融合多个信息
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill


class IFNet_bf_resnet(nn.Module):
    def __init__(self):
        super(IFNet_bf_resnet, self).__init__()
        self.block0 = IFBlock_bf(6, c=240, tf_dim=64)
        self.block1 = IFBlock_bf(13 + 4, c=150, tf_dim=64)
        self.block2 = IFBlock_bf(13 + 4, c=90, tf_dim=128)
        self.block_tea = IFBlock_bf(16 + 4, c=90, tf_dim=128)
        # self.contextnet = Contextnet()
        # 使用 resnet结构提取特征
        self.contextnet = resnet50_feature()
        self.unet = Unet()

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        # print(img0.shape)
        # 学生网络
        stu = [self.block0, self.block1, self.block2]
        # 三层IFBlock计算累加
        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                        scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        # 在训练时，有参考帧，教师网络使用参考帧进行光流计算
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                            scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        # 融合多个信息
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill



class IFNet_bf_resnet_local_mae(nn.Module):
    """通过参数来选择不同的组件"""

    def __init__(self):
        super(IFNet_bf_resnet_local_mae, self).__init__()
        self.block0 = IFBlock_bf(6, c=240, tf_dim=64)
        self.block1 = IFBlock_bf(13 + 4, c=150, tf_dim=64)
        self.block2 = IFBlock_bf(13 + 4, c=90, tf_dim=128)
        self.block_tea = IFBlock_bf(16 + 4, c=90, tf_dim=128)
        self.contextnet = resnet50_feature()
        self.unet = UnetCBAM()
        self.mae = get_local_mae_patch_8()

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        # 学生网络
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                        scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                            scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])

        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)

        # 重建阶段
        fusion_imgs = merged[2]
        if hasattr(self, 'mae'):
            if gt.shape[1] == 3:
                # 训练时使用 gt 做目标，推测时使用fusion_imgs
                # start_time = time.time()
                imgs_reconstructed, loss_reconstruct = self.mae(fusion_imgs, mask, gt)
                # end_time = time.time()
                # print(f"imgs_reconstruction took {end_time - start_time} seconds.")
            else:
                imgs_reconstructed, loss_reconstruct = self.mae(fusion_imgs, mask, fusion_imgs)
            return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill, \
                imgs_reconstructed, loss_reconstruct
        else:
            return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill, None, 0


class IFNet_bf_resnet_cbam_L(nn.Module):
    def __init__(self):
        super(IFNet_bf_resnet_cbam_L, self).__init__()
        self.block0 = IFBlock_bf_L(6, c=360, tf_dim=128)
        self.block1 = IFBlock_bf_L(13 + 4, c=225, tf_dim=128)
        self.block2 = IFBlock_bf_L(13 + 4, c=135, tf_dim=256)
        self.block_tea = IFBlock_bf_L(16 + 4, c=135, tf_dim=256)
        self.contextnet = resnet50_feature_L()
        self.unet = UnetCBAM_L()

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                        scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                            scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill


class IFNet_bf_resnet_cbam_res(nn.Module):
    def __init__(self):
        super(IFNet_bf_resnet_cbam_res, self).__init__()
        self.block0 = IFBlock_bf(6, c=240, tf_dim=64)
        self.block1 = IFBlock_bf(13 + 4, c=150, tf_dim=64)
        self.block2 = IFBlock_bf(13 + 4, c=90, tf_dim=128)
        self.block_tea = IFBlock_bf(16 + 4, c=90, tf_dim=128)
        self.contextnet = resnet50_feature()
        self.unet = UnetCBAM_Res()

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                        scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                            scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        # 融合多个信息
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill


class IFNet_bf_resnet_cbam_L_res(nn.Module):
    def __init__(self):
        super(IFNet_bf_resnet_cbam_L_res, self).__init__()
        self.block0 = IFBlock_bf_L(6, c=360, tf_dim=128)
        self.block1 = IFBlock_bf_L(13 + 4, c=225, tf_dim=128)
        self.block2 = IFBlock_bf_L(13 + 4, c=135, tf_dim=256)
        self.block_tea = IFBlock_bf_L(16 + 4, c=135, tf_dim=256)
        self.contextnet = resnet50_feature_L()
        self.unet = UnetCBAM_L_Res()

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                        scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                            scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill


class IFNet_bf_resnet_cbam_M(nn.Module):
    def __init__(self):
        super(IFNet_bf_resnet_cbam_M, self).__init__()
        self.block0 = IFBlock_bf(6, c=240, tf_dim=64)
        self.block1 = IFBlock_bf(13 + 4, c=150, tf_dim=64)
        self.block2 = IFBlock_bf(13 + 4, c=90, tf_dim=128)
        self.block_tea = IFBlock_bf(16 + 4, c=90, tf_dim=128)
        self.contextnet = resnet50_feature()
        self.unet = UnetCBAM_M()

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                        scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                            scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])

        mask_guide = []
        # 获得多尺度patch mask
        m = get_rec_region(mask_list[2], 0.15, 0.85)
        _, m = get_rec_patches(m, patch_size=8, threshold=2)
        for i in range(3):
            scale_factor = 2
            m = F.avg_pool2d(m, kernel_size=scale_factor, stride=scale_factor)
            mask_guide.append(m)
        # 融合多个信息
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, mask_guide)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill

    def visualize(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                        scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                            scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

        c0, c0_x = self.contextnet.visilize(img0, flow[:, :2])
        c1, c1_x = self.contextnet.visilize(img1, flow[:, 2:4])
        mask_guide = []
        # 获得多尺度patch mask
        m = get_rec_region(mask_list[2], 0.2, 0.8)
        _, m = get_rec_patches(m, patch_size=8, threshold=2)
        for i in range(3):
            scale_factor = 2
            m = F.avg_pool2d(m, kernel_size=scale_factor, stride=scale_factor)
            mask_guide.append(m)
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, mask_guide)
        res = tmp[:, :3] * 2 - 1
        before_res = merged[2].clone()
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, tmp, before_res, res, mask_guide, warped_img0, warped_img1, c0, c0_x


class IFNet_bf_resnet_cbam_M_Res(nn.Module):
    def __init__(self):
        super(IFNet_bf_resnet_cbam_M_Res, self).__init__()
        self.block0 = IFBlock_bf(6, c=240, tf_dim=64)
        self.block1 = IFBlock_bf(13 + 4, c=150, tf_dim=64)
        self.block2 = IFBlock_bf(13 + 4, c=90, tf_dim=128)
        self.block_tea = IFBlock_bf(16 + 4, c=90, tf_dim=128)
        self.contextnet = resnet50_feature()
        self.unet = UnetCBAM_M_Res()

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                        scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                            scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])

        mask_guide = []
        # 获得多尺度patch mask
        m = get_rec_region(mask_list[2], 0.15, 0.85)
        _, m = get_rec_patches(m, patch_size=4, threshold=2)
        for i in range(3):
            scale_factor = 2
            m = F.avg_pool2d(m, kernel_size=scale_factor, stride=scale_factor)
            mask_guide.append(m)
        # 融合多个信息
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, mask_guide)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill


class IFNet_bf_resnet_cbam_M_Res_L(nn.Module):
    def __init__(self):
        super(IFNet_bf_resnet_cbam_M_Res_L, self).__init__()
        self.block0 = IFBlock_bf_L(6, c=360, tf_dim=192, n_block=6)
        self.block1 = IFBlock_bf_L(13 + 4, c=225, tf_dim=128, n_block=6)
        self.block2 = IFBlock_bf_L(13 + 4, c=135, tf_dim=128, n_block=6)
        self.block_tea = IFBlock_bf_L(16 + 4, c=135, tf_dim=128, n_block=4)
        self.contextnet = resnet50_feature()
        self.unet = UnetCBAM_M_Res()

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                        scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                            scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])

        mask_guide = []
        # 获得多尺度patch mask
        m = get_rec_region(mask_list[2], 0.15, 0.85)
        _, m = get_rec_patches(m, patch_size=4, threshold=2)
        for i in range(3):
            scale_factor = 2
            m = F.avg_pool2d(m, kernel_size=scale_factor, stride=scale_factor)
            mask_guide.append(m)
        # 融合多个信息
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, mask_guide)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill

class IFNet_bf_resnet_cbam_HM(nn.Module):
    def __init__(self):
        super(IFNet_bf_resnet_cbam_HM, self).__init__()
        self.block0 = IFBlock_bf_H_v2(6, c=240, tf_dim=192, n_win=7, n_block=4, kv_downsample_mode='ada_avgpool',
                                      topk=4, mlp_ratio=3, kv_per_win=1)
        self.block1 = IFBlock_bf_H_v2(13 + 4, c=150, tf_dim=128, n_win=7, n_block=6, kv_downsample_mode='ada_avgpool',
                                      topk=6, mlp_ratio=3, kv_per_win=1)
        self.block2 = IFBlock_bf_H_v2(13 + 4, c=90, tf_dim=64, n_win=14, n_block=6, kv_downsample_mode='ada_avgpool',
                                      topk=8, mlp_ratio=3, kv_per_win=2)
        self.block_tea = IFBlock_bf_H_v2(16 + 4, c=90, tf_dim=64, n_win=14, n_block=2, kv_downsample_mode='ada_avgpool',
                                         topk=4, mlp_ratio=3, kv_per_win=1)
        self.contextnet = resnet50_feature()
        self.unet = UnetCBAM_MH()
        self.hc0 = conv(240, 64, 3, 1, 1)
        self.hc1 = conv(150, 32, 3, 1, 1)
        self.hc2 = conv(90, 16, 3, 1, 1)
        self.hc = [self.hc0, self.hc1, self.hc2]

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        hybrid_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow is not None:
                flow_d, mask_d, hybrid = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                                scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask, hybrid = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            hybrid_list.append(hybrid)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d, _ = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                               scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        hybs = []
        for i in range(3):
            hyb = self.hc[i](hybrid_list[i])
            hybs.append(hyb)
        mask_guide = []
        # 获得多尺度patch mask
        m = get_rec_region(mask_list[2], 0.2, 0.8)
        _, m = get_rec_patches(m, patch_size=8, threshold=2)
        for i in range(3):
            scale_factor = 2
            m = F.avg_pool2d(m, kernel_size=scale_factor, stride=scale_factor)
            mask_guide.append(m)
        # 融合多个信息
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, hybs, mask_guide)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill


class IFNet_bf_resnet_cbam_HM_Res(nn.Module):
    def __init__(self):
        super(IFNet_bf_resnet_cbam_HM_Res, self).__init__()
        self.block0 = IFBlock_bf_H_v2(6, c=240, tf_dim=192, n_win=7, n_block=4, kv_downsample_mode='ada_avgpool',
                                      topk=4, mlp_ratio=3, kv_per_win=1)
        self.block1 = IFBlock_bf_H_v2(13 + 4, c=150, tf_dim=128, n_win=7, n_block=6, kv_downsample_mode='ada_avgpool',
                                      topk=6, mlp_ratio=3, kv_per_win=1)
        self.block2 = IFBlock_bf_H_v2(13 + 4, c=90, tf_dim=64, n_win=14, n_block=6, kv_downsample_mode='ada_avgpool',
                                      topk=8, mlp_ratio=3, kv_per_win=2)
        self.block_tea = IFBlock_bf_H_v2(16 + 4, c=90, tf_dim=64, n_win=14, n_block=2, kv_downsample_mode='ada_avgpool',
                                         topk=4, mlp_ratio=3, kv_per_win=1)
        self.contextnet = resnet50_feature()
        self.unet = UnetCBAM_MH_Res()
        self.hc0 = conv(240, 64, 3, 1, 1)
        self.hc1 = conv(150, 32, 3, 1, 1)
        self.hc2 = conv(90, 16, 3, 1, 1)
        self.hc = [self.hc0, self.hc1, self.hc2]

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        hybrid_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d, hybrid = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                                scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask, hybrid = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            hybrid_list.append(hybrid)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d, _ = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                               scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        hybs = []
        for i in range(3):
            hyb = self.hc[i](hybrid_list[i])
            hybs.append(hyb)
        mask_guide = []
        # 获得多尺度patch mask
        m = get_rec_region(mask_list[2], 0.2, 0.8)
        _, m = get_rec_patches(m, patch_size=8, threshold=2)
        for i in range(3):
            scale_factor = 2
            m = F.avg_pool2d(m, kernel_size=scale_factor, stride=scale_factor)
            mask_guide.append(m)
        # 融合多个信息
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, hybs, mask_guide)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill


class IFNet_bf_resnet_cbam_HM_L(nn.Module):
    def __init__(self):
        super(IFNet_bf_resnet_cbam_HM_L, self).__init__()
        self.block0 = IFBlock_bf_H_L(6, c=360, tf_dim=64, n_block=6, topk=4)
        self.block1 = IFBlock_bf_H_L(13 + 4, c=225, tf_dim=128, n_block=6, topk=6)
        self.block2 = IFBlock_bf_H_L(13 + 4, c=135, tf_dim=192, n_block=6, topk=8, n_win=7)
        self.block_tea = IFBlock_bf_H_L(16 + 4, c=135, tf_dim=128, n_block=6)
        self.contextnet = resnet50_feature_L()
        self.unet = UnetCBAM_MH_L()
        self.hc0 = conv(360, 64, 3, 1, 1)
        self.hc1 = conv(225, 32, 3, 1, 1)
        self.hc2 = conv(135, 16, 3, 1, 1)
        self.hc = [self.hc0, self.hc1, self.hc2]

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        hybrid_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d, hybrid = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                                scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask, hybrid = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            hybrid_list.append(hybrid)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d, _ = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                               scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        hybs = []
        for i in range(3):
            hyb = self.hc[i](hybrid_list[i])
            # hyb = F.interpolate(hyb, scale_factor=2, mode="bilinear", align_corners=False)
            hybs.append(hyb)
        mask_guide = []
        # 获得多尺度patch mask
        m = get_rec_region(mask_list[2], 0.15, 0.85)
        _, m = get_rec_patches(m, patch_size=4, threshold=2)
        mask_guide.append(m)
        for i in range(2):
            scale_factor = 2
            m = F.avg_pool2d(m, kernel_size=scale_factor, stride=scale_factor)
            mask_guide.append(m)
        # 融合多个信息
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, hybs, mask_guide)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill


class IFNet_bf_resnet_cbam_HM_Res_L(nn.Module):
    def __init__(self):
        super(IFNet_bf_resnet_cbam_HM_Res_L, self).__init__()
        self.block0 = IFBlock_bf_H_L(6, c=360, tf_dim=64, n_block=6, topk=4)
        self.block1 = IFBlock_bf_H_L(13 + 4, c=225, tf_dim=128, n_block=6, topk=4, n_win=7)
        self.block2 = IFBlock_bf_H_L(13 + 4, c=135, tf_dim=192, n_block=6, topk=4, n_win=14)
        self.block_tea = IFBlock_bf_H_L(16 + 4, c=135, tf_dim=128, n_block=6)
        self.contextnet = resnet50_feature_L()
        self.unet = UnetCBAM_MH_Res_L()
        self.hc0 = conv(360, 64, 3, 1, 1)
        self.hc1 = conv(225, 32, 3, 1, 1)
        self.hc2 = conv(135, 16, 3, 1, 1)
        self.hc = [self.hc0, self.hc1, self.hc2]

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        hybrid_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d, hybrid = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                                scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask, hybrid = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            hybrid_list.append(hybrid)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d, _ = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                               scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        hybs = []
        for i in range(3):
            hyb = self.hc[i](hybrid_list[i])
            # hyb = F.interpolate(hyb, scale_factor=2, mode="bilinear", align_corners=False)
            hybs.append(hyb)
        mask_guide = []
        # 获得多尺度patch mask
        m = get_rec_region(mask_list[2], 0.2, 0.8)
        _, m = get_rec_patches(m, patch_size=8, threshold=2)
        mask_guide.append(m)
        for i in range(2):
            scale_factor = 2
            m = F.avg_pool2d(m, kernel_size=scale_factor, stride=scale_factor)
            mask_guide.append(m)
        # 融合多个信息
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, hybs, mask_guide)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill


class IFNet_bf_resnet_cbam_HM_Res_L_v2(nn.Module):
    def __init__(self):
        super(IFNet_bf_resnet_cbam_HM_Res_L_v2, self).__init__()

        self.block0 = IFBlock_bf_H_L_v2(6, c=360, tf_dim=192, n_win=7, n_block=4, kv_downsample_mode='ada_avgpool',
                                      topk=4, mlp_ratio=3, kv_per_win=1)
        self.block1 = IFBlock_bf_H_L_v2(13 + 4, c=225, tf_dim=128, n_win=7, n_block=6, kv_downsample_mode='ada_avgpool',
                                      topk=6, mlp_ratio=3, kv_per_win=1)
        self.block2 = IFBlock_bf_H_L_v2(13 + 4, c=135, tf_dim=64, n_win=14, n_block=6, kv_downsample_mode='ada_avgpool',
                                      topk=8, mlp_ratio=3, kv_per_win=1)
        self.block_tea = IFBlock_bf_H_L_v2(16 + 4, c=135, tf_dim=64, n_win=14, n_block=2, kv_downsample_mode='ada_avgpool',
                                         topk=4, mlp_ratio=3, kv_per_win=1)

        self.contextnet = resnet50_feature_L()
        self.unet = UnetCBAM_MH_Res_L()
        self.hc0 = conv(360, 64, 3, 1, 1)
        self.hc1 = conv(225, 32, 3, 1, 1)
        self.hc2 = conv(135, 16, 3, 1, 1)
        self.hc = [self.hc0, self.hc1, self.hc2]

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        hybrid_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d, hybrid = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                                scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask, hybrid = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            hybrid_list.append(hybrid)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d, _ = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                               scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        hybs = []
        for i in range(3):
            hyb = self.hc[i](hybrid_list[i])
            # hyb = F.interpolate(hyb, scale_factor=2, mode="bilinear", align_corners=False)
            hybs.append(hyb)
        mask_guide = []
        # 获得多尺度patch mask
        m = get_rec_region(mask_list[2], 0.2, 0.8)
        _, m = get_rec_patches(m, patch_size=8, threshold=2)
        mask_guide.append(m)
        for i in range(2):
            scale_factor = 2
            m = F.avg_pool2d(m, kernel_size=scale_factor, stride=scale_factor)
            mask_guide.append(m)
        # 融合多个信息
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, hybs, mask_guide)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill



class IFNet_bf_resnet_RF_M(nn.Module):
    def __init__(self):
        super(IFNet_bf_resnet_RF_M, self).__init__()
        self.block0 = IFBlock_bf(6, c=240, tf_dim=192, n_win=7, n_block=4, kv_downsample_mode='ada_avgpool',
                                      topk=4, mlp_ratio=2, kv_per_win=1)
        self.block1 = IFBlock_bf(13 + 4, c=150, tf_dim=128, n_win=7, n_block=6, kv_downsample_mode='ada_avgpool',
                                      topk=4, mlp_ratio=2, kv_per_win=1)
        self.block2 = IFBlock_bf(13 + 4, c=90, tf_dim=64, n_win=14, n_block=6, kv_downsample_mode='ada_avgpool',
                                      topk=6, mlp_ratio=2, kv_per_win=1)
        self.block_tea = IFBlock_bf(16 + 4, c=90, tf_dim=64, n_win=14, n_block=2, kv_downsample_mode='ada_avgpool',
                                         topk=2, mlp_ratio=2, kv_per_win=1)
        self.contextnet = resnet50_feature_L()
        self.unet = Restormer(inp_channels=50, out_channels=3, dim=32)

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                        scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow,
                                            scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1,
                                                                                                     True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])

        mask_guide = []
        # 获得多尺度patch mask
        m = get_rec_region(mask_list[2], 0.15, 0.85)
        _, m = get_rec_patches(m, patch_size=8, threshold=2)
        mask_guide.append(m)
        for i in range(2):
            scale_factor = 2
            m = F.avg_pool2d(m, kernel_size=scale_factor, stride=scale_factor)
            mask_guide.append(m)
        # 融合多个信息
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, mask_guide)
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill
# test net

if __name__ == "__main__":
    # flownet = IFNet_bf_resnet_local_mae()
    flownet = IFNet_bf_resnet_RF_M()
    input = torch.rand(1, 9, 224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flownet.to(device)
    input_cuda = input.to(device)
    output = flownet(input_cuda, scale=[4, 2, 1])
    print('finish')
