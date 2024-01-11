import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet import *
from model.IFNet_m import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model:
    def __init__(self, local_rank=-1, arbitrary=False, model_name='IFNet'):
        if arbitrary == True:
            self.flownet = IFNet_m()
        elif model_name == 'IFNet_bf':
            self.flownet = IFNet_bf()
        elif model_name == 'IFNet_bf_resnet_cbam':
            self.flownet = IFNet_bf_resnet_cbam()
        elif model_name == 'IFNet_bf_resnet':
            self.flownet = IFNet_bf_resnet()
        elif model_name == 'IFNet_bf_resnet_tws':
            self.flownet = IFNet_bf_resnet_tws()
        elif model_name == 'IFNet_bf_cbam_resnet_bi':
            self.flownet = IFNet_bf_cbam_resnet_bi()
        elif model_name == 'IFNet_bf_cbam_mulExt':
            self.flownet = IFNet_bf_cbam_mulExt()
        elif model_name == 'IFNet_bf_resnet_local_mae':
            self.flownet = IFNet_bf_resnet_local_mae()
        elif model_name == 'IFNet_bf_resnet_spe_mae':
            self.flownet = IFNet_bf_resnet_cbam()
            self.mae = get_local_mae_patch_8()
        elif model_name == 'IFNet_bf_resnet_spe_mae_p16':
            self.flownet = IFNet_bf_resnet_cbam()
            self.mae = get_local_mae_patch_16()

        else:
            self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
        if hasattr(self, 'mae'):
            self.optimG_mae = AdamW(self.mae.parameters(), lr=1e-6,
                                weight_decay=1e-3)
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()
        if hasattr(self, 'mae'):
            self.mae.train()

    def eval(self):
        self.flownet.eval()
        if hasattr(self, 'mae'):
            self.mae.eval()

    def device(self):
        self.flownet.to(device)
        if hasattr(self, 'mae'):
            self.mae.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if rank <= 0:
            self.flownet.load_state_dict(torch.load('{}/flownet.pkl'.format(path)))
            #self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))
            checkpoint_path = '{}/mae.checkpoint'.format(path)
            if hasattr(self, 'mae') and os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path)
                self.mae.load_state_dict(convert(state_dict), strict=False)

    def save_model(self, path, epoch, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(), '{}/flownet_{}.pkl'.format(path, epoch))
            if hasattr(self, 'mae'):
                torch.save(self.mae.state_dict(), '{}/mae_{}.checkpoint'.format(path, epoch))

    def inference(self, img0, img1, scale=1, scale_list=[4, 2, 1], TTA=False, timestep=0.5):
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        if TTA == False:
            if hasattr(self, 'mae'):
                imgs_reconstructed, loss_reconstruct = self.mae(merged[2], mask, merged[2])
                pred = imgs_reconstructed
            return pred
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            if hasattr(self, 'mae'):
                pred = (merged[2] + merged2[2].flip(2).flip(3)) / 2
                imgs_reconstructed, loss_reconstruct = self.mae(pred, mask, pred)
                pred = imgs_reconstructed
            return pred

    def inference_with_mask(self, img0, img1, scale=1, scale_list=[4, 2, 1], TTA=False, timestep=0.5):
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        if TTA == False:
            return merged[2], mask
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        # 两个被cat起来的img
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        #start_time = time.time()
        with torch.no_grad():
            flow, mask, merged, flow_teacher, merged_teacher, loss_distill \
                = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1])
        #end_time = time.time()
        #print(f"flownet took {end_time - start_time} seconds.")
        loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean()

        # 重建阶段
        if hasattr(self, 'mae'):
            imgs_reconstructed, loss_reconstruct = self.mae(merged[2], mask, gt)
            if training:
                self.optimG_mae.zero_grad()
                loss_reconstruct.backward()
                self.optimG_mae.step()
        else:
            imgs_reconstructed = merged[2]
            loss_reconstruct = 0.0

        return merged[2], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
            'imgs_reconstructed': imgs_reconstructed,
            'loss_reconstruct': loss_reconstruct,
            }
