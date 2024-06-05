d import argparse
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
from dataset import VimeoTripletDataset, DataLoader
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
    total_rife_time = []
    total_context_time = []
    total_fusion_time = []
    warp_psnr_list = []
    

    for valIndex, data in enumerate(val_data):
        
        if(valIndex % 100 == 0):
            print("Validation index " + str(valIndex))
        
        with torch.no_grad():
            data = data.to(device, non_blocking=True) / 255.0
            
            gt = data[:, :3] # ground truth
            ref = data[:, 3:6]
            lr = data[:, 6:9]
            
            """image_show(gt)
            image_show(ref)
            image_show(lr)"""
            
            imgs = torch.cat((ref, lr), 1)
            
            start_time = time.time()
            pred,  rife_time,  context_time, fusion_time, warp_psnr = model(imgs, gt)
            # pred = model(imgs)
            total_times.append(time.time()-start_time)
            total_rife_time.append(rife_time)
            # total_warp_time.append(warp_time)
            total_context_time.append(context_time)
            total_fusion_time.append(fusion_time)
    	
            psnr = -10 * math.log10(((gt - pred) * (gt - pred)).mean())
            ssim_ = float(ssim_matlab(pred, gt))
            psnr_list.append(psnr)
            ssim_list.append(ssim_)
            warp_psnr_list.append(warp_psnr)
            
            

    print("Total average:")
    print(np.mean(total_times))
    print("RIFE average")
    print(np.mean(total_rife_time))
    # print("Warp average")
    # print(np.mean(total_warp_time))
    print("ContextNet average")
    print(np.mean(total_context_time))
    print("FusionNet average")
    print(np.mean(total_fusion_time))
    print("Warp PSNR")
    print(np.mean(warp_psnr_list))

    return np.mean(psnr_list), np.mean(ssim_list)


if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    

    model = HSTRNet(device)

    dataset_val = VimeoTripletDataset("validation", "/home/ortak/mughees/datasets/vimeo_triplet/", device)
    val_data_last = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=False)
    len_val = dataset_val.__len__()

    model.ifnet.load_state_dict(
        convert(torch.load('./model/RIFE_v5/train_log/flownet.pkl', map_location=device)))
    model.ifnet.eval()
    
    model.contextnet.load_state_dict(torch.load('./model_dict/HSTR_contextnet_30.pkl', map_location=device))
    model.contextnet.eval()
    
    model.unet.load_state_dict(torch.load("./model_dict/HSTR_unet_30.pkl", map_location=device))
    model.unet.eval()

    psnr,  ssim_ = validate(model, val_data_last, len_val) 

    print(psnr)
    print(ssim_)
