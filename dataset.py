import cv2
import torch
import numpy as np
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
import os
import torchvision.transforms.functional as FF
from PIL import Image
import time
import math
from model.pytorch_msssim import ssim
import torch.nn as nn

from utils.utils import image_show, padding_airforce, im2tensor
from utils.aug import homography_d, gaussian_noise, contrast, horizontal_flip, rotate, apply_augment

device = "cuda"
img_index = 1 


def aug_vis(gt, ref, lr, h, w, mode):
    if mode == "train_cufed":
        print("Cropping is skipped")
    else:    
        ih, iw, _ = gt.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        gt = gt[x:x+h, y:y+w, :]
        ref = ref[x:x+h, y:y+w, :]
        lr = lr[x:x+h, y:y+w, :]

    if mode == "train":
        gt = Image.fromarray(gt.astype(np.uint8))
        ref = Image.fromarray(ref.astype(np.uint8))
        lr = Image.fromarray(lr.astype(np.uint8))
        

        #Applying horizontal flip with %20 probability 
        #-------------------------------------------------------------------
        p = random.uniform(0.0, 1.0)
        if(p < 0.2):
            gt, ref, lr = horizontal_flip(gt, ref, lr)
        #-------------------------------------------------------------------
    
        #Applying rotation
        #-------------------------------------------------------------------
        gt, ref, lr = rotate(gt, ref, lr)
        #-------------------------------------------------------------------

        gt = np.array(gt)
        ref = np.array(ref)
        lr = np.array(lr)
    return gt, ref, lr


def aug(gt, ref, lr, h, w, mode):
    ih, iw, _ = gt.shape
    x = np.random.randint(0, ih - h + 1)
    y = np.random.randint(0, iw - w + 1)
    gt = gt[x:x+h, y:y+w, :]
    ref = ref[x:x+h, y:y+w, :]
    lr = lr[x:x+h, y:y+w, :]

    if mode == "train":
        """# Applying homography to LR frames
        #-------------------------------------------------------------------
        img0_LR = homography(img0_LR, 0.00000000001)
        img1_LR = homography(img1_LR, 0.0000000005)
        img2_LR = homography(img2_LR, 0.00000000001)
        #-------------------------------------------------------------------

        #Applying gaussian noise to HR and LR frames
        #-------------------------------------------------------------------
        img0_LR, img1_LR, img2_LR = gaussian_noise(img0_LR, img1_LR, img2_LR)
        #-------------------------------------------------------------------

        img0_LR = Image.fromarray(img0_LR.astype(np.uint8))
        img1_LR = Image.fromarray(img1_LR.astype(np.uint8))
        img2_LR = Image.fromarray(img2_LR.astype(np.uint8))
        img0 = Image.fromarray(img0.astype(np.uint8))
        gt = Image.fromarray(gt.astype(np.uint8))
        img1 = Image.fromarray(img1.astype(np.uint8))
        

        #Applying contrast to HR and LR frames
        #-------------------------------------------------------------------
        img0_LR, img1_LR, img2_LR = contrast(img0_LR, img1_LR, img2_LR)
        #-------------------------------------------------------------------"""
        
        gt = Image.fromarray(gt)
        ref = Image.fromarray(ref)
        lr = Image.fromarray(lr)

        #Applying horizontal flip with %20 probability 
        #-------------------------------------------------------------------
        p = random.uniform(0.0, 1.0)
        if(p < 0.2):
            gt, ref, lr = horizontal_flip(gt, ref, lr)
        #-------------------------------------------------------------------
    
        #Applying rotation
        #-------------------------------------------------------------------
        gt, ref, lr = rotate(gt, ref, lr)
        #-------------------------------------------------------------------
    
        gt = np.array(gt)
        ref = np.array(ref)
        lr = np.array(lr)
            
    return gt, ref, lr

class VimeoDataset(Dataset):
    def __init__(self, dataset_name, device, batch_size=32, transform=False):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.transform = transform
        self.device = device
        self.load_data()
        self.h = 256
        self.w = 448
        xx = np.arange(0, self.w).reshape(1,-1).repeat(self.h,0)
        yy = np.arange(0, self.h).reshape(-1,1).repeat(self.w,1)
        self.grid = np.stack((xx,yy),2).copy()

    def __len__(self):
        return len(self.meta_data_ref) - 1

    def load_data(self):
        self.trainlist_gt = []
        self.trainlist_ref = []
        self.trainlist_lr = []
        self.testlist_gt = []
        self.testlist_ref = []
        self.testlist_lr = []
        train_path_gt = os.path.join("/home/adastec/vimeo_new/train_gt.txt")
        test_path_gt = os.path.join("/home/adastec/vimeo_new/test_gt.txt")
        train_path_ref = os.path.join("/home/adastec/vimeo_new/train_ref.txt")
        test_path_ref = os.path.join("/home/adastec/vimeo_new/test_ref.txt")
        train_path_lr = os.path.join("/home/adastec/vimeo_new/train_lr.txt")
        test_path_lr = os.path.join("/home/adastec/vimeo_new/test_lr.txt")
        with open(train_path_gt, 'r') as f:
            self.trainlist_gt = f.read().splitlines()
        with open(train_path_ref, 'r') as f:
            self.trainlist_ref = f.read().splitlines()
        with open(train_path_lr, 'r') as f:
            self.trainlist_lr = f.read().splitlines()
        with open(test_path_gt, 'r') as f:
            self.testlist_gt = f.read().splitlines()
        with open(test_path_ref, 'r') as f:
            self.testlist_ref = f.read().splitlines()
        with open(test_path_lr, 'r') as f:
            self.testlist_lr = f.read().splitlines()

        if self.dataset_name == 'train':
            self.meta_data_ref = self.trainlist_ref
            self.meta_data_gt = self.trainlist_gt
            self.meta_data_lr = self.trainlist_lr
            print('Number of training samples in: ' + str(len(self.meta_data_ref)))
        else:
            self.meta_data_ref = self.testlist_ref#[:int(len(self.testlist_ref) / 6)]
            self.meta_data_gt = self.testlist_gt#[:int(len(self.testlist_gt) / 6)]
            self.meta_data_lr = self.testlist_lr#[:int(len(self.testlist_lr) / 6)]
            # self.meta_data_HR = self.testlist_HR[::10]
            # self.meta_data_LR = self.testlist_LR[::10]
            print('Number of validation samples in: ' + str(len(self.meta_data_ref)))
        self.nr_sample = len(self.meta_data_ref)

    
    def getimg(self, index):
        gt_path = self.meta_data_gt[index]
        ref_path = self.meta_data_ref[index]
        lr_path = self.meta_data_lr[index]
        
        """ print('gt ==== ', gt_path)
        print('ref ==== ', ref_path)
        print('lr ==== ', lr_path)
        print("="*100) """
        
        gt = padding_airforce(cv2.imread(gt_path))
        ref = padding_airforce(cv2.imread(ref_path))
        lr = padding_airforce(cv2.imread(lr_path))


        return gt, ref, lr

    def __getitem__(self, index):
        gt, ref, lr = self.getimg(index)
        if self.dataset_name == 'train':
            gt, ref, lr = aug(gt, ref, lr, 256, 256, "train")

            if self.transform:
                augs = ["blend", "rgb", "mixup", "cutout", "cutmix", "cutmixup", "cutblur"]
                prob = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                alpha = [0.6, 1.0, 1.2, 0.001, 0.7, 0.7, 0.7]
                aux_prob, aux_alpha = 1.0, 1.2

                ref = im2tensor(ref).unsqueeze(0).to(self.device)
                gt = im2tensor(gt).unsqueeze(0).to(self.device)
                lr = im2tensor(lr).unsqueeze(0).to(self.device)

                ref, gt, lr, mask, augx = apply_augment(ref, gt, lr, augs, prob, alpha, aux_prob, aux_alpha)

                ref = torch.squeeze(ref, 0)
                ref = ref.detach().to('cpu').numpy()
                ref = np.transpose(ref, (1, 2, 0))

                gt = torch.squeeze(gt, 0)
                gt = gt.detach().to('cpu').numpy()
                gt = np.transpose(gt, (1, 2, 0))

                lr = torch.squeeze(lr, 0)
                lr = lr.detach().to('cpu').numpy()
                lr = np.transpose(lr, (1, 2, 0))

                
           
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
            ref = torch.from_numpy(ref.copy()).permute(2, 0, 1)
            lr = torch.from_numpy(lr.copy()).permute(2, 0, 1)

            return torch.cat((gt, ref, lr), 0)
        
        elif self.dataset_name == "validation":
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
            ref = torch.from_numpy(ref.copy()).permute(2, 0, 1)
            lr = torch.from_numpy(lr.copy()).permute(2, 0, 1)
            
            return torch.cat((gt, ref, lr), 0)
        
        

class MAMIDataset(Dataset):
    def __init__(self, dataset_name, data_root, device, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.device = device
        self.load_data()
        self.h = 256
        self.w = 448
        xx = np.arange(0, self.w).reshape(1,-1).repeat(self.h,0)
        yy = np.arange(0, self.h).reshape(-1,1).repeat(self.w,1)
        self.grid = np.stack((xx,yy),2).copy()

    def __len__(self):
        return len(self.meta_data_HR) - 3

    def load_data(self):
        self.trainlist_HR = []
        self.trainlist_LR = []
        self.testlist_HR = []
        self.testlist_LR = []
        data_path_HR = os.path.join(self.data_root, "HR/")
        data_path_LR = os.path.join(self.data_root, "LR/")
        train_path = os.path.join(self.data_root, 'train_list.txt')
        test_path = os.path.join(self.data_root, 'test_list.txt')
        with open(train_path, 'r') as f:
            self.trainlist_HR = f.read().splitlines()
        with open(train_path, 'r') as f:
            self.trainlist_LR = f.read().splitlines()
        with open(test_path, 'r') as f:
            self.testlist_HR = f.read().splitlines()
        with open(test_path, 'r') as f:
            self.testlist_LR = f.read().splitlines()

        for i, entry in enumerate(self.trainlist_HR):
            new_entry = data_path_HR  + entry
            self.trainlist_HR[i] = new_entry
        for i, entry in enumerate(self.trainlist_LR):
            new_entry = data_path_LR  + entry
            self.trainlist_LR[i] = new_entry
        for i, entry in enumerate(self.testlist_HR):
            new_entry = data_path_HR  + entry
            self.testlist_HR[i] = new_entry
        for i, entry in enumerate(self.testlist_LR):
            new_entry = data_path_LR  + entry
            self.testlist_LR[i] = new_entry
        # SHOULD REMOVE THIS FOR BENCHMARK
        # -------------------------------------------------------------------
        self.testlist_HR_train = []
        self.testlist_LR_train = []
        for i, entry in enumerate(self.testlist_HR):
            if(i % 5 == 0):
                new_entry = data_path_HR + entry
                self.testlist_HR_train.append(new_entry)
        for i, entry in enumerate(self.testlist_LR):
            if(i % 5 == 0):
                new_entry = data_path_LR + entry
                self.testlist_LR_train.append(new_entry)
        # -------------------------------------------------------------------
        if self.dataset_name == 'train':
            self.meta_data_HR = self.trainlist_HR
            self.meta_data_LR = self.trainlist_LR
            print('Number of training samples in: ' + str(self.data_root.split("/")[-1]), len(self.meta_data_HR))
        else:
            self.meta_data_HR = self.testlist_HR
            self.meta_data_LR = self.testlist_LR
            print('Number of validation samples in: ' + str(self.data_root.split("/")[-1]), len(self.meta_data_HR))
        self.nr_sample = len(self.meta_data_HR)

    
    def getimg(self, index):
        
        gt = cv2.imread(self.meta_data_HR[index])
        ref = cv2.imread(self.meta_data_HR[index + 1])
        lr = cv2.imread(self.meta_data_LR[index])

        #print(self.meta_data_HR[index])
        #print(self.meta_data_HR[index + 1])
        #print(self.meta_data_LR[index])
        
        return gt, ref, lr


    def __getitem__(self, index):
        gt, ref, lr = self.getimg(index)
        
        if self.dataset_name == 'train':
            gt, ref, lr = aug_vis(gt, ref, lr, 256, 256, "train")
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
            ref = torch.from_numpy(ref.copy()).permute(2, 0, 1)
            lr = torch.from_numpy(lr.copy()).permute(2, 0, 1)
            return torch.cat((gt, ref, lr), 0)
        elif self.dataset_name == "validation":
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).to(self.device)
            ref = torch.from_numpy(ref.copy()).permute(2, 0, 1).to(self.device)
            lr = torch.from_numpy(lr.copy()).permute(2, 0, 1).to(self.device)
        return torch.cat((gt, ref, lr), 0)
