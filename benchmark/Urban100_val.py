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
from model.pytorch_msssim import ssim_matlab
from dataset import Urban100Dataset
from dataset import DataLoader

device = "cuda"

def convert(param):
    return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

def image_show(result):
    result = result.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    cv2.imshow("win", result)
    cv2.waitKey(2000)
    return result

def padding(img):
        padding1_mult = math.floor(img.shape[2] / 32) + 1
        if padding1_mult % 2 == 1:
            padding1_mult +=1
        padding2_mult = math.floor(img.shape[3] / 32) + 1
        if padding2_mult % 2 == 1:
            padding2_mult +=1
        
        pad1 = (32 * padding1_mult) - img.shape[2]
        pad2 = (32 * padding2_mult) - img.shape[3]
        
        
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

def validate(model, val_data, len_val):
    model.ifnet.eval()
    model.contextnet.eval()
    model.unet.eval()

    psnr_list = []
    ssim_list = []
    total_times = []

    count = 0
    for valIndex, data in enumerate(val_data):
        
        if(valIndex % 100 == 0):
            print("Validation index " + str(valIndex))
        with torch.no_grad():
            gt = data[0].to(device, non_blocking=True) / 255.0
            ref = data[1].to(device, non_blocking=True) / 255.0
            lr = data[2].to(device, non_blocking=True) / 255.0
            
            initial_shape = gt.shape
            ref = padding(ref)
            lr = padding(lr)
            if(initial_shape[2] > 1000 and initial_shape[3] > 1000):
                continue
            if(initial_shape[2] > 900 and initial_shape[3] > 1100):                                                                                        
                continue
            # image_show(gt)
            # image_show(ref)
            # image_show(lr)
            
            imgs = torch.cat((ref, lr), 1)
            start_time = time.time()
            pred = model(imgs)
            total_times.append(time.time() - start_time)

            pred = crop(pred, initial_shape[2], initial_shape[3])
            psnr = float(-10 * math.log10(((gt - pred) * (gt - pred)).mean()))
            ssim_ = float(ssim_matlab(pred, gt))
            psnr_list.append(psnr)
            ssim_list.append(ssim_)
            count += 1
            torch.cuda.empty_cache()    

    print(count)
    
    print("Total time average")
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
   
    dataset_val = Urban100Dataset("validation", "/home/ortak/mughees/datasets/Urban100/", device)
    val_data = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=False)

    len_val = dataset_val.__len__()
    
    model.ifnet.load_state_dict(
        convert(torch.load('./model/RIFE_v5/train_log/flownet.pkl', map_location=device)))
    model.ifnet.eval()
    
    model.contextnet.load_state_dict(torch.load('./model_dict/HSTR_contextnet_30.pkl', map_location=device))
    model.contextnet.eval()
    
    model.unet.load_state_dict(torch.load("./model_dict/HSTR_unet_30.pkl", map_location=device))
    model.unet.eval()
    
    print("Validation is starting")
    psnr, ssim_ = validate(model, val_data, len_val)
    print(psnr)
    print(ssim_)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
