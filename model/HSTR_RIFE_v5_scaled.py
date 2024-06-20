import torch
import torch.nn as nn

import torch.nn.functional as F


from model.RIFE_v5.warplayer import warp
from model.RIFE_v5.IFNet import IFNet
from model.RIFE_v5.RIFE import ContextNet, FusionNet
from model.RIFE_v5.swin_v2 import Swin_V2


class HSTRNet(nn.Module):
    def __init__(self, device):
        super(HSTRNet, self).__init__()
        self.device = device
        
        self.ifnet = IFNet(self.device)
        self.contextnet = ContextNet()
        self.attention = Swin_V2(3)
        self.unet = FusionNet()
        
        self.ifnet.to(self.device)
        self.contextnet.to(self.device)
        self.attention.to(self.device)
        self.unet.to(self.device)
    
    def forward(self, imgs, gt=None):
        ref = imgs[:, :3]
        lr = imgs[:, 3:6]
        
        _, flow = self.ifnet(torch.cat((lr, ref), 1))
     
        f_0_1 = flow[:, 2:4] * 2
        f_0_1 = F.interpolate(f_0_1, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_gt = warp(ref, f_0_1, self.device)
        
        c0 = self.contextnet(ref, f_0_1)

        attn = self.attention(lr, ref)

        refine_output = self.unet(warped_gt, lr, attn, c0)
        
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        #mask = torch.sigmoid(refine_output[:, 3:4])
        #merged_img = warped_gt * mask + lr * (1 - mask)
        pred = warped_gt + res
        pred = torch.clamp(pred, 0, 1)
        return pred
