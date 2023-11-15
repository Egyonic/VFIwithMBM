import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.refine import *
from model.biformer_models.biformer import Block as BiformerBlock
from model.biformer_models.biformer import get_4_scale_multi_feature_extractor
from model.resnet import resnet50_feature, resnet50_feature_tws, ResNet_feature_bi, get_ResNet_feature_bi


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
            pre_norm=True)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        # print(x.shape)
        # breakpoint()
        x = self.tran_conv(x)
        # print(f'shape before TF block: {x.shape}')
        # breakpoint()
        x = self.tf_block(x)
        # print(f'shape after TF block: {x.shape}')
        x = self.tf_conv_revert(x)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        # 上采样
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask


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


class IFBlock_bf(nn.Module):
    """
    使用了 BiFormer Block 的IFBlock， 采用金字塔结构提取三层特征叠加
    """

    def __init__(self, in_planes, c=64, tf_dim=64):
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
        for i in range(4):
            biformer_blocks.append(BiformerBlock(
                dim=tf_dim, n_win=7, num_heads=4, kv_downsample_mode='identity', kv_per_win=-1,
                topk=4, mlp_ratio=3, side_dwconv=5, before_attn_dwconv=3, layer_scale_init_value=-1,
                qk_dim=tf_dim, param_routing=False, diff_routing=False, soft_routing=False, pre_norm=True))
        self.tf_block = nn.Sequential(*biformer_blocks)
        # self.tf_block = BiformerBlock(
        #     dim=tf_dim, n_win=7, num_heads=4, kv_downsample_mode='identity', kv_per_win=-1,
        #     topk=4, mlp_ratio=3, side_dwconv=5, before_attn_dwconv=3, layer_scale_init_value=-1,
        #     qk_dim=tf_dim, param_routing=False, diff_routing=False, soft_routing=False, pre_norm=True)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        # breakpoint()
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


class IFNet_bf(nn.Module):
    def __init__(self):
        super(IFNet_bf, self).__init__()
        self.block0 = IFBlock_bf(6, c=240, tf_dim=64)
        self.block1 = IFBlock_bf(13 + 4, c=150, tf_dim=64)
        self.block2 = IFBlock_bf(13 + 4, c=90, tf_dim=128)
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


class IFNet_bf_resnet_tws(nn.Module):
    def __init__(self):
        super(IFNet_bf_resnet_tws, self).__init__()
        self.block0 = IFBlock_bf(6, c=240, tf_dim=64)
        self.block1 = IFBlock_bf(13 + 4, c=150, tf_dim=64)
        self.block2 = IFBlock_bf(13 + 4, c=90, tf_dim=128)
        self.block_tea = IFBlock_bf(16 + 4, c=90, tf_dim=128)
        # self.contextnet = Contextnet()
        # 使用 resnet结构提取特征
        self.contextnet = get_ResNet_feature_bi()
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


class IFNet_bf_cbam_resnet_bi(nn.Module):
    """使用IFNet + BiFormer + CBAM + Resnet with BiFormer"""

    def __init__(self):
        super(IFNet_bf_cbam_resnet_bi, self).__init__()
        self.block0 = IFBlock_bf(6, c=240, tf_dim=64)
        self.block1 = IFBlock_bf(13 + 4, c=150, tf_dim=64)
        self.block2 = IFBlock_bf(13 + 4, c=90, tf_dim=128)
        self.block_tea = IFBlock_bf(16 + 4, c=90, tf_dim=128)
        # self.contextnet = Contextnet()
        # 使用 resnet结构提取特征
        self.contextnet = get_ResNet_feature_bi()
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


class IFNet_bf_cbam_mulExt(nn.Module):
    """使用IFNet + BiFormer +多尺度的基于BiFormer的特征提取器"""

    def __init__(self):
        super(IFNet_bf_cbam_mulExt, self).__init__()
        self.block0 = IFBlock_bf(6, c=240, tf_dim=64)
        self.block1 = IFBlock_bf(13 + 4, c=150, tf_dim=64)
        self.block2 = IFBlock_bf(13 + 4, c=90, tf_dim=128)
        self.block_tea = IFBlock_bf(16 + 4, c=90, tf_dim=128)
        self.contextnet = get_4_scale_multi_feature_extractor()
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
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill


class IFNet_bf_resnet_maeVit(nn.Module):
    """通过参数来选择不同的组件"""

    def __init__(self):
        super(IFNet_bf_resnet_maeVit, self).__init__()
        self.block0 = IFBlock_bf(6, c=240, tf_dim=64)
        self.block1 = IFBlock_bf(13 + 4, c=150, tf_dim=64)
        self.block2 = IFBlock_bf(13 + 4, c=90, tf_dim=128)
        self.block_tea = IFBlock_bf(16 + 4, c=90, tf_dim=128)

        self.contextnet = resnet50_feature()
        self.unet = UNetMAEViT()

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
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill


# test net

if __name__ == "__main__":
    flownet = IFNet_bf_resnet_maeVit()
    input = torch.rand(1, 9, 224, 224)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_cuda = input.to("cpu")
    output = flownet(input, scale=[4, 2, 1])
