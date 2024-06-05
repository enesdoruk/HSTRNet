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

from model.HSTR_RIFE_v5_scaled import HSTRNet
from dataset import CameraFusionDataset, DataLoader
from model.pytorch_msssim import ssim_matlab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_show(result):
    result = result.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    cv2.imshow("win", result)
    cv2.waitKey(2000)
    return result

def padding(img, dim_2, dim_3):
        padding1_mult = math.floor(img.shape[2] / 32) + 2
        padding2_mult = math.floor(img.shape[3] / 32) + 1
        if img.shape[2] < img.shape[3]:
            pad1 = dim_2 - img.shape[2]
            pad2 = dim_3 - img.shape[3]
        else:
            pad1 = dim_3 - img.shape[2]
            pad2 = dim_2 - img.shape[3]
        
        img = torch.unsqueeze(img, 0)
        if pad1 % 2 == 0 and pad2 % 2 == 0:
            padding = (int(pad2/2), int(pad2/2), int(pad1/2), int(pad1/2))
        elif pad1 % 2 == 1 and pad2 % 2 == 0:
            padding = (int(pad2/2), int(pad2/2), int(pad1/2), int(pad1/2) + 1)
        elif pad1 % 2 == 0 and pad2 % 2 == 1:
            padding = (int(pad2/2), int(pad2/2) + 1, int(pad1/2), int(pad1/2))
        else:
            padding = (int(pad2/2), int(pad2/2) + 1, int(pad1/2), int(pad1/2) + 1)
        
        # img = img.float()
        # img = padding(img)
        img = F.pad(img, padding, "constant", 0)
        img = torch.squeeze(img, 0)
        
        return img

def crop(pred, w, h):
        _, _ , iw, ih = pred.shape
        x = int((iw - w) / 2)
        y = int((ih - h) / 2)
        if w % 2 == 0 and h % 2 == 0:
            pred = pred[:, :, x:iw-x, y:ih-y]
        elif w % 2 == 1 and h % 2 == 0:
            pred = pred[:, :, x:iw-x-1, y:ih-y]
        elif w % 2 == 0 and h % 2 == 1:
            pred = pred[:, :, x:iw-x, y:ih-y-1]
        else:
            pred = pred[:, :, x:iw-x-1, y:ih-y-1]
        return pred

def convert(param):
    return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

def validate(model, val_data, len_val):
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

    for valIndex, data in enumerate(val_data):
        
        if(valIndex % 100 == 0):
            print("Validation index " + str(valIndex))
        
        with torch.no_grad():
            data = data.to(device, non_blocking=True) / 255.0
            
            gt = data[:, :3] # ground truth
            ref = data[:, 3:6]
            lr = data[:, 6:9]
                        
            scale = 4
            for i in range(scale):
                index_2_0 = int((gt.shape[2] / scale) * i)
                index_2_1 = int((gt.shape[2] / scale) * i + (gt.shape[2] / scale))
                index_3_0 = int((gt.shape[3] / scale) * i)
                index_3_1 = int((gt.shape[3] / scale) * i + (gt.shape[3] / scale))
                gt_temp = gt[:, :, index_2_0 : index_2_1, index_3_0 : index_3_1]
                ref_temp = ref[:, :, index_2_0 : index_2_1, index_3_0 : index_3_1]
                lr_temp = lr[:, :, index_2_0 : index_2_1, index_3_0 : index_3_1]
                
                
                ref_temp = padding(ref_temp, 768, 1024)
                lr_temp = padding(lr_temp, 768, 1024)
                
                imgs = torch.cat((ref_temp, lr_temp), 1)
            
                start_time = time.time()
                pred, _ ,_ ,_ ,_ = model(imgs)
                total_times.append(time.time()-start_time)
                pred = crop(pred, gt_temp.shape[2], gt_temp.shape[3])
                
                psnr = -10 * math.log10(((gt_temp - pred) * (gt_temp - pred)).mean())
                ssim_ = float(ssim_matlab(pred, gt_temp))
                psnr_list.append(psnr)
                print("Index:" + str(i) + " --> PSNR:"+ str(psnr))
                ssim_list.append(ssim_)
            
           
            """image_show(gt)
            image_show(ref)
            image_show(lr)"""
    print("Total average:")
    print(np.mean(total_times))
    return np.mean(psnr_list), np.mean(ssim_list)


if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    model = HSTRNet(device)

    dataset_val = CameraFusionDataset("validation", "/home/ortak/mughees/datasets/CameraFusion/test/", device)
    val_data_last = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=False)
    len_val = dataset_val.__len__()

    model.ifnet.load_state_dict(
        convert(torch.load('./model/RIFE_v5/train_log/flownet.pkl', map_location=device)))
    model.ifnet.eval()
    
    model.contextnet.load_state_dict(torch.load('/home/mughees/Projects/HSTRNet_RefSR/model_dict/HSTR_contextnet_30.pkl', map_location=device), strict=False)
    model.contextnet.eval()
    
    model.unet.load_state_dict(torch.load("/home/mughees/Projects/HSTRNet_RefSR/model_dict/HSTR_unet_30.pkl", map_location=device))
    model.unet.eval()

    psnr,  ssim_ = validate(model, val_data_last, len_val) 

    print(psnr)
    print(ssim_)
