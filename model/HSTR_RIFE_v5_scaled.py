import torch
import torch.nn as nn
import cv2
import time
import math
import numpy as np
import sys
import argparse
import io
import torch.nn.functional as F



from torch.nn.parallel import DistributedDataParallel as DDP
from model.RIFE_v5.warplayer import warp
from model.RIFE_v5.IFNet import IFNet
from model.RIFE_v5.RIFE import ContextNet, FusionNet

def image_show(result):
    result = result.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    cv2.imshow("win", result)
    cv2.waitKey(2000)
    return result

class HSTRNet(nn.Module):
    def __init__(self, device):
        super(HSTRNet, self).__init__()
        self.device = device
        
        self.ifnet = IFNet(self.device)
        self.contextnet = ContextNet()
        self.unet = FusionNet()
        
        self.ifnet.to(self.device)
        self.contextnet.to(self.device)
        self.unet.to(self.device)
        
        
    def return_parameters(self):
        param_list = list(self.contextnet.parameters())
        for param in self.unet.parameters():
            param_list.append(param)
        return param_list
    
    def convert(self, param):
        return {
        k.replace("module.", ""): v
            for k, v in param.items()
            if "module." 
        }
    
    def convert_to_numpy(self, img):
        result = img.cpu().detach().numpy()
        result = result[0, :]
        result = np.transpose(result, (1, 2, 0))
        return result * 255

    #-------------------------------------------------------------------
    def homography(self, img):
        img = self.convert_to_numpy(img)
        
        homography = np.zeros((3,3))
        homography[0][0] = 1
        homography[1][1] = 1
        homography[2][2] = 1

        #homography = homography + (0.00000000001 ** 0.5) * np.random.randn(homography.shape[0], homography.shape[1])
        homography = homography + (0.0000000005 ** 0.5) * np.random.randn(homography.shape[0], homography.shape[1])
        homography[2][2] = 1

        homography_img = cv2.warpPerspective(img, homography, (img.shape[1], img.shape[0]))

        homography_img = torch.from_numpy(np.transpose(homography_img, (2, 0, 1))).to(
            self.device, non_blocking=True).unsqueeze(0).float() / 255.

        return homography_img
    #-------------------------------------------------------------------

    def forward(self, imgs, gt=None):
        ref = imgs[:, :3]
        lr = imgs[:, 3:6]
        # image_show(ref)
        # image_show(lr)
        
        
        # ref-->t+1, lr-->t
        _, _, flow = self.ifnet(torch.cat((lr, ref), 1))
        # 0.5 --> lr, 0.5 -->ref
        # what we need --> gt --> ref (0-->1)
        # (0.5 --> ref) * 2 = 0-->ref (gt-->ref)

        
        f_0_1 = flow[:, 2:4] * 2
        f_0_1 = F.interpolate(f_0_1, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_gt = warp(ref, f_0_1, self.device)
        # Warped gt and warped gt2 gives same result, using warped gt2 directly might give better performance
        
        
        c0 = self.contextnet(ref, f_0_1)


        refine_output = self.unet(warped_gt, lr, c0)
        
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = warped_gt * mask + lr * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)
        
        
        return pred
