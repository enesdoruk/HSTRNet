import argparse
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
import datetime
import os

from tqdm import tqdm
from model.HSTR_RIFE_v5_scaled import HSTRNet
from dataset import VimeoDataset, DataLoader, HSTR_Dataset
from model.pytorch_msssim import ssim_matlab
from utils.utils import image_show, convert_module


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model):

    logger.info("Device: %s", device)

    dataset_train = VimeoDataset("train", "/home/other/Umut/Vimeo90K/vimeo_septuplet/", device)
    train_data = DataLoader(
        dataset_train, batch_size=32, num_workers=0, drop_last=True, shuffle=True
    )

    logger.info("Training dataset is loaded")

    dataset_val = HSTR_Dataset("validation", "/home/other/Umut/HSTR_Data", device)
    val_data = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=False)

    logger.info("Validation dataset is loaded")

    len_val = dataset_val.__len__()

    L1_lossFn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    print("Training...")
    print("Train data set size:" + str(train_data.__len__()))
    print("Val data set size:" + str(val_data.__len__()))
    logger.info("Training is starting")

    model.ifnet.load_state_dict(torch.load('./model_dict/pretrained/HSTR_ifnet_26.pkl', map_location=device))
   
    """model.contextnet.load_state_dict(torch.load('./model_dict/vimeo/HSTR_contextnet_62.pkl', map_location=device))

    model.attention.load_state_dict(torch.load('./model_dict/vimeo/HSTR_attention_62.pkl', map_location=device))

    model.unet.load_state_dict(torch.load('./model_dict/vimeo/HSTR_unet_62.pkl', map_location=device))"""
    
    #optimizer.load_state_dict(torch.load('./model_dict/vimeo/HSTR_optimizer_62.pkl', map_location=device))
    
    # Freezing models that are not going to be trained
    """for k, v in model.ifnet.named_parameters():
        v.requires_grad = False"""

    # Below code is a test to check if validation works as expected.
    print("Validation is starting")
    psnr, ssim = validate(model, val_data, len_val, 1)
    print(psnr)
    print(ssim)

    start = time.time()

    loss = 0
    psnr_list = []
    ssim_list = []
    for epoch in range(100):
        model.ifnet.train()
        model.contextnet.train()
        model.attention.train()
        model.unet.train()
        loss = 0

        print("Epoch: ", epoch)
        logger.info("---------------------------------------------")
        logger.info("Epoch:" + str(epoch))
        logger.info("---------------------------------------------")

        for trainIndex, data in enumerate(tqdm(train_data)):
            model.ifnet.train()
            model.contextnet.train()
            model.attention.train()
            model.unet.train()            

            if trainIndex % 100 == 0:
                logger.info("Train Index:" + str(trainIndex))

            data = data.to(device, non_blocking=True) / 255.0
            data = data.float()
            
            gt_batch = data[:, :18]
            ref_batch = data[:, 18:36]
            lr_batch = data[:, 36:54]
            
            for i in range(6):
                gt = gt_batch[:, i * 3: i * 3 + 3]
                ref = ref_batch[:, i * 3: i * 3 + 3]
                lr = lr_batch[:, i * 3: i * 3 + 3]
            
                """image_show(gt)
                image_show(ref)
                cv2.imwrite("augmented_lr.png", image_show(lr))
                exit()"""

                

                imgs = torch.cat((ref, lr), 1)
                optimizer.zero_grad()
                pred = model(imgs)
                L1_loss = L1_lossFn(pred, gt)
                L1_loss.backward()
                optimizer.step()
                loss += float(L1_loss.item())
                end = time.time()

            if trainIndex == (train_data.__len__() - 1):
                torch.save(model.ifnet.state_dict(), "model_dict/hstr_data/HSTR_ifnet_" + str(epoch) + ".pkl")
                torch.save(model.contextnet.state_dict(), "model_dict/hstr_data/HSTR_contextnet_" + str(epoch) + ".pkl")
                torch.save(model.attention.state_dict(), "model_dict/hstr_data/HSTR_attention_" + str(epoch) + ".pkl")
                torch.save(model.unet.state_dict(), "model_dict/hstr_data/HSTR_unet_" + str(epoch) + ".pkl")
                torch.save(optimizer.state_dict(), "model_dict/hstr_data/HSTR_optimizer_" + str(epoch) + ".pkl")
                
                print("Validating, Train Index: " + str(trainIndex))
                logger.info("Validating, Train Index: " + str(trainIndex))

                with torch.no_grad():
                    psnr, ssim = validate(model, val_data, len_val, 1)

                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    endVal = time.time()

                print(
                    " Loss: %0.6f  TrainExecTime: %0.1f  ValPSNR: %0.4f  ValEvalTime: %0.2f  SSIM: %0.4f "
                    % (loss / trainIndex, end - start, psnr, endVal - end, ssim)
                )
                logger.info(
                    "Train index: "
                    + str(trainIndex)
                    + " Loss: "
                    + str(round(loss / trainIndex, 6))
                    + " TrainExecTime: "
                    + str(round(end - start, 1))
                    + " ValPSNR: "
                    + str(round(psnr, 4))
                    + " ValEvalTime: "
                    + str(round(endVal - end, 2))
                    + " SSIM: "
                    + str(round(ssim, 4))
                )
                start = time.time()

    val_data_last = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=False)

    psnr, ssim = validate(model, val_data_last, len_val, 1)
    logger.info("------------------------------------------")
    logger.info(
        "Last evaluation --> PSNR:"
        + str(psnr)
        + " SSIM:"
        + str(ssim)
    )
    print("When batch size=1 --> PSNR:" + str(psnr) + " SSIM:" + str(ssim))

def validate(model, val_data, len_val, batch_size):
    model.ifnet.eval()
    model.contextnet.eval()
    model.attention.eval()
    model.unet.eval()

    psnr_list = []
    ssim_list = []

    for valIndex, data in enumerate(tqdm(val_data)):
        with torch.no_grad():
            data = data.to(device, non_blocking=True) / 255.0
    
            gt = data[:, :3]
            ref = data[:, 3:6]
            lr = data[:, 6:9]
            
            """image_show(gt)
            image_show(ref)
            image_show(lr)"""
        
            imgs = torch.cat((ref, lr), 1)

            

            pred = model(imgs)
            for i in range(int(pred.shape[0])):
                
                if(((gt[i: i+1,:] - pred[i: i+1,:]) * (gt[i: i+1,:] - pred[i: i+1,:])).mean()) <= 0:
                    psnr = 0
                else:
                    psnr = -10 * math.log10(((gt[i: i+1,:] - pred[i: i+1,:]) * (gt[i: i+1,:] - pred[i: i+1,:])).mean())
                ssim_ = float(ssim_matlab(pred[i: i+1,:], gt[i: i+1,:]))
                psnr_list.append(psnr)
                ssim_list.append(ssim_)
    return np.mean(psnr_list), np.mean(ssim_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--epoch", default=100, type=int)
    args = parser.parse_args()


    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    #torch.autograd.set_detect_anomaly(True)

    log_file_path = "logs/training.log" + str(datetime.datetime.today().strftime('_%d-%m-%H'))
    if os.path.isfile(log_file_path):
        os.remove(log_file_path)

    debug = 0
    logger = logging.getLogger('train')
    logger.propagate = False
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    fh = logging.FileHandler("logs/training.log" + str(datetime.datetime.today().strftime('_%d-%m-%H')))
    if debug:
        fh.setLevel(logging.DEBUG)
    else:
        fh.setLevel(logging.INFO)
    #ch = logging.StreamHandler()
    #ch.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    #ch.setFormatter(formatter)
    logger.addHandler(fh)
    #logger.addHandler(ch)    model = nn.DataParallel(HSTRNet(device))

    model = HSTRNet(device)

    try:
        train(model)
    except Exception as e:
        logging.exception("Unexpected exception! %s", e)
