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
from dataset import VizdroneDataset
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

def crop(pred, gt, w, h):
        _, _ , iw, ih = pred.shape
        x = int((iw - w) / 2)
        y = int((ih - h) / 2)
        pred = pred[:, :, x:iw-x, y:ih-y]
        gt = gt[:, :, x:iw-x, y:ih-y]
        return pred, gt

def validate(model, val_data, len_val):
    model.ifnet.eval()
    model.contextnet.eval()
    model.unet.eval()

    psnr_list = []
    ssim_list = []
    total_times = []
    total_rife_time = []
    total_warp_time = []
    total_context_time = []
    total_fusion_time = []

    for valIndex, data in enumerate(val_data):
        #if(valIndex != 2519):
        #    continue
        if(valIndex % 100 == 0):
            print("Validation index " + str(valIndex))
        with torch.no_grad():
            data = data.to(device, non_blocking=True) / 255.0
            gt = data[:, :3]
            ref = data[:, 3:6]
            lr = data[:, 6:9]

            # gt, ref, lr = data
            # gt = gt.to(device, non_blocking=True) / 255.0
            # ref = ref.to(device, non_blocking=True) / 255.0
            # lr = lr.to(device, non_blocking=True) / 255.0

            imgs = torch.cat((ref, lr), 1)
        
            # image_show(gt)
            # image_show(ref)
            # image_show(lr)
     
            # exit()   
     
            start_time = time.time()
            pred, _, _, _, _ = model(imgs)
            total_times.append(time.time() - start_time)
            
            """total_times.append(time.time()-start_time)
        total_rife_time.append(rife_time)
        total_warp_time.append(warp_time)
        total_context_time.append(context_time)
        total_fusion_time.append(fusion_time)"""

            """image_show(pred)
            cv2.waitKey(2000)
            image_show(gt)
            cv2.waitKey(2000)"""
        
            pred, gt = crop(pred, gt, 380, 672)
            #result = image_show(pred * 255)
            #cv2.imwrite("hstr-ref_rec_vis.png", result)
            # result = pred.cpu().detach().numpy()
            # result = result[0, :]
            # result = np.transpose(result, (1, 2, 0))
            # pred = cv2.resize(result, (gt.shape[3], gt.shape[2]))
            # pred = torch.from_numpy(pred).permute(2,0,1)
            # pred = pred.unsqueeze(0).to(device)
            # image_show(pred)
            # image_show(gt)
            # print(pred.shape)
            # print(gt.shape)
            # exit()
            psnr = -10 * math.log10(((gt - pred) * (gt - pred)).mean())
            
            # print(psnr)
            ssim_ = float(ssim_matlab(pred, gt))
            psnr_list.append(psnr)
            ssim_list.append(ssim_)
            # print(psnr)
    
    print("Total time average")
    print(np.mean(total_times))
    """print("RIFE average")
    print(np.mean(total_rife_time))
    print("Warp average")
    print(np.mean(total_warp_time))
    print("ContextNet average")
    print(np.mean(total_context_time))
    print("FusionNet average")
    print(np.mean(total_fusion_time))"""
    return np.mean(psnr_list), np.mean(ssim_list)

if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    """torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True"""

    model = HSTRNet(device)
   
    dataset_val = VizdroneDataset("validation", "/home/mughees/thinclient_drives/VIZDRONE/upsampled/original/", device)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
