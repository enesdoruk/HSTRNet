import argparse
import sys
sys.path.append('.')
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
import math
import logging
import os

from model.HSTR_RIFE_v5_scaled import HSTRNet
from model.pytorch_msssim import ssim_matlab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_show(result):
    result = result.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    cv2.imshow("win", result)
    cv2.waitKey(2000)
    return result


def convert(param):
    return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

def validate(model):
    model.ifnet.eval()
    model.contextnet.eval()
    model.unet.eval()

    for k, v in model.ifnet.named_parameters():
        v.requires_grad = False
    for k, v in model.contextnet.named_parameters():
        v.requires_grad = False
    for k, v in model.unet.named_parameters():
        v.requires_grad = False

    psnr_list = []
    ssim_list = []
    total_times = []
    total_rife_time = []
    total_context_time = []
    total_fusion_time = []
    warp_psnr_list = []
    
    video_idx = 0
    for root, dirs, files in os.walk("/home/ortak/mughees/datasets/ERA/Videos/Test/", topdown=True):
        for name in sorted(files):
            hr_frames = []
            lr_frames = []
            r_path = os.path.join(root, name)
            w_path = os.path.join("/home/ortak/mughees/datasets/ERA/HR/Test", r_path[2:-5])
            cap = cv2.VideoCapture(r_path)
            
            if(r_path[-7:-6] != "0"):
                continue
            
            s, frame = cap.read()
            count = 0
            while s:
                hr_frame = torch.from_numpy(frame.copy()).permute(2, 0, 1).to(device)
                hr_frame = hr_frame.float() / 255.0
                hr_frame = torch.unsqueeze(hr_frame, 0)
                hr_frames.append(hr_frame)
                lr_frame = F.interpolate(hr_frame, scale_factor=0.25, mode="bicubic", align_corners=False)
                lr_frame = F.interpolate(lr_frame, scale_factor=4, mode="bicubic", align_corners=False)
                lr_frames.append(lr_frame)
                s, frame = cap.read()
                # image_show(hr_frame)
                # image_show(lr_frame)
            
            
            for idx in range(len(hr_frames)-1):
                lr = lr_frames[idx]
                gt = hr_frames[idx]
                ref = hr_frames[idx+1]
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                imgs = torch.cat((ref, lr), 1)
                start.record()
                pred,  rife_time,  context_time, fusion_time, warp_psnr = model(imgs, gt)
                end.record()
                torch.cuda.synchronize()
                total_times.append(start.elapsed_time(end))
                
                psnr = -10 * math.log10(((gt - pred) * (gt - pred)).mean())
                ssim_ = float(ssim_matlab(pred, gt))
                psnr_list.append(psnr)
                ssim_list.append(ssim_)
            print("Video " + str(video_idx) + " has finished processing. Current average PSNR:" + str(np.mean(psnr_list)))
            video_idx += 1
       
    print("Total average time:" + str(np.mean(total_times)))
    return np.mean(psnr_list), np.mean(ssim_list)


if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


    model = HSTRNet(device)
    model.ifnet.load_state_dict(convert(torch.load('./model/RIFE_v5/train_log/flownet.pkl', map_location=device)))
    model.ifnet.eval()
    model.contextnet.load_state_dict(torch.load('./model_dict/HSTR_contextnet_30.pkl', map_location=device))
    model.contextnet.eval()
    model.unet.load_state_dict(torch.load("./model_dict/HSTR_unet_30.pkl", map_location=device))
    model.unet.eval()

    psnr,  ssim_ = validate(model) 
    print("Average PSNR:" + str(psnr))
    print("Average SSIM:" + str(ssim_))
