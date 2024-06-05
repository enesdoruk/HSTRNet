import cv2
import numpy as np
import torch
import math

def image_show(result):
    result = result.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    """cv2.imshow("win", result)
    cv2.waitKey(2000)"""
    return result

def convert_module(param):
        return {
        k.replace("module.", ""): v
            for k, v in param.items()
            if "module." 
        }
    
def convert_to_numpy(img):
        result = img.cpu().detach().numpy()
        result = result[0, :]
        result = np.transpose(result, (1, 2, 0))
        return result * 255
    
def psnr_calc(img0, img1):
    return -10 * math.log10(((img0 - img1) * (img0 - img1)).mean())

def im2tensor(im):
    np_t = np.ascontiguousarray(im.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_t).float()
    return tensor

def padding_airforce(img):
    if(img.shape[0] == 576 and img.shape[1] == 960):
        return img
    else:
        pad1 = 576 - img.shape[0]
        pad2 = 960 - img.shape[1]

        img_r = np.zeros((576, 960, 3))

        if pad1 % 2 == 0 and pad2 % 2 == 0:
            if pad1 == 0:
                img_r[:, int(pad2 / 2) : -int(pad2 / 2), :] = img
            elif pad2 == 0:
                img_r[int(pad1 / 2) : -int(pad1 / 2), :, :] = img
            else:
                img_r[int(pad1 / 2) : -int(pad1 / 2), int(pad2 / 2) : -int(pad2 / 2), :] = img
        elif pad1 % 2 == 1 and pad2 % 2 == 0:
            if pad2 == 0:
                img_r[int(pad1 / 2) + 1 : -int(pad1 / 2), :, :] = img
            else:
                img_r[int(pad1 / 2) + 1 : -int(pad1 / 2), int(pad2 / 2) : -int(pad2 / 2), :] = img
        elif pad1 % 2 == 0 and pad2 % 2 == 1:
            if pad1 == 0:
                img_r[:, int(pad2 / 2) + 1: -int(pad2 / 2), :] = img
            else:
                img_r[int(pad1 / 2) : -int(pad1 / 2), int(pad2 / 2) + 1: -int(pad2 / 2), :] = img

        else:
            img_r[int(pad1 / 2) + 1 : -int(pad1 / 2), int(pad2 / 2) + 1: -int(pad2 / 2), :] = img
        return img_r.astype(np.uint8)