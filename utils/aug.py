import numpy as np
import cv2
import torch
import random
import torchvision.transforms.functional as FF
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F


def homography(self, img):
    img = self.convert_to_numpy(img)
    homography = np.zeros((3,3))
    homography[0][0] = 1
    homography[1][1] = 1
    homography[2][2] = 1
    homography = homography + (0.0000000005 ** 0.5) * np.random.randn(homography.shape[0], homography.shape[1])
    homography[2][2] = 1
    homography_img = cv2.warpPerspective(img, homography, (img.shape[1], img.shape[0]))
    homography_img = torch.from_numpy(np.transpose(homography_img, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
    return homography_img
    
def homography_d(img, p):        
    homography = np.zeros((3,3))
    homography[0][0] = 1
    homography[1][1] = 1
    homography[2][2] = 1
    homography = homography + (p ** 0.5) * np.random.randn(homography.shape[0], homography.shape[1])
    homography[2][2] = 1
    homography_img = cv2.warpPerspective(img, homography, (img.shape[1], img.shape[0]))
    return homography_img

def gaussian_noise(img0, img1, img2):  
    factor = random.uniform(0.0, 2.5)
    img0 = img0 + np.random.randn(*img0.shape) * factor + 0
    img1 = img1 + np.random.randn(*img1.shape) * factor + 0
    img2 = img2 + np.random.randn(*img2.shape) * factor + 0
    return img0, img1, img2

def contrast(img0, img1, img2):  
    factor = random.uniform(0.9, 1.1)
    img0 = FF.adjust_contrast(img0, factor)
    img1 = FF.adjust_contrast(img1, factor)
    img2 = FF.adjust_contrast(img2, factor)
    return img0, img1, img2

def horizontal_flip(img0, img1, img2):
    img0 = FF.hflip(img0)
    img1 = FF.hflip(img1)
    img2 = FF.hflip(img2)
    return img0, img1, img2

def rotate(img0, img1, img2):
    degree = random.uniform(-10.0, 10.0)
    rotated_img0 = img0.rotate(degree)
    rotated_img1 = img1.rotate(degree)
    rotated_img2 = img2.rotate(degree)
    return rotated_img0, rotated_img1, rotated_img2



def apply_augment(
    im1, im2, im3,
    augs, probs, alphas,
    aux_prob=None, aux_alpha=None,
    mix_p=None
):
    idx = np.random.choice(len(augs), p=mix_p)
    aug = augs[idx]
    prob = float(probs[idx])
    alpha = float(alphas[idx])
    mask = None

    if aug == "none":
        im1_aug, im2_aug, im3_aug = im1.clone(), im2.clone(), im3.clone()
    elif aug == "blend":
        im1_aug, im2_aug, im3_aug = blend(
            im1.clone(), im2.clone(), im3.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "mixup":
        im1_aug, im2_aug, im3_aug = mixup(
            im1.clone(), im2.clone(), im3.clone(),
            prob=prob, alpha=alpha,
        )
    elif aug == "cutout":
        im1_aug, im2_aug, im3_aug, mask, _, _ = cutout(
            im1.clone(), im2.clone(), im3.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "cutmix":
        im1_aug, im2_aug, im3_aug = cutmix(
            im1.clone(), im2.clone(), im3.clone(),
            prob=prob, alpha=alpha,
        )
    elif aug == "cutmixup":
        im1_aug, im2_aug, im3_aug = cutmixup(
            im1.clone(), im2.clone(), im3.clone(),
            mixup_prob=aux_prob, mixup_alpha=aux_alpha,
            cutmix_prob=prob, cutmix_alpha=alpha,
        )
    elif aug == "cutblur":
        im1_aug, im2_aug, im3_aug = cutblur(
            im1.clone(), im2.clone(), im3.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "rgb":
        im1_aug, im2_aug, im3_aug = rgb(
            im1.clone(), im2.clone(), im3.clone(),
            prob=prob
        )
    else:
        raise ValueError("{} is not invalid.".format(aug))

    return im1_aug, im2_aug, im3_aug,  mask, aug


def blend(im1, im2, im3, prob=1.0, alpha=0.6):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2, im3

    c = torch.empty((im2.size(0), 3, 1, 1), device=im2.device).uniform_(0, 255)
    rim2 = c.repeat((1, 1, im2.size(2), im2.size(3)))
    rim1 = c.repeat((1, 1, im1.size(2), im1.size(3)))
    rim3 = c.repeat((1, 1, im3.size(2), im3.size(3)))

    v = np.random.uniform(alpha, 1)
    im1 = v * im1 + (1-v) * rim1
    im2 = v * im2 + (1-v) * rim2
    im3 = v * im3 + (1-v) * rim3

    return im1, im2, im3


def mixup(im1, im2, im3, prob=1.0, alpha=1.2):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2, im3

    v = np.random.beta(alpha, alpha)
    r_index = torch.randperm(im1.size(0)).to(im2.device)

    im1 = v * im1 + (1-v) * im1[r_index, :]
    im2 = v * im2 + (1-v) * im2[r_index, :]
    im3 = v * im3 + (1-v) * im3[r_index, :]
    return im1, im2, im3


def _cutmix(im2, prob=1.0, alpha=1.0):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return None

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)

    fcy = np.random.randint(0, h-ch+1)
    fcx = np.random.randint(0, w-cw+1)
    tcy, tcx = fcy, fcx
    rindex = torch.randperm(im2.size(0)).to(im2.device)

    return {
        "rindex": rindex, "ch": ch, "cw": cw,
        "tcy": tcy, "tcx": tcx, "fcy": fcy, "fcx": fcx,
    }


def cutmix(im1, im2, im3, prob=1.0, alpha=1.0):
    c = _cutmix(im2, prob, alpha)
    if c is None:
        return im1, im2, im3

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2[rindex, :, fcy:fcy+ch, fcx:fcx+cw]
    im3[..., tcy:tcy+ch, tcx:tcx+cw] = im3[rindex, :, fcy:fcy+ch, fcx:fcx+cw]
    im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[rindex, :, hfcy:hfcy+hch, hfcx:hfcx+hcw]

    return im1, im2, im3


def cutmixup(
    im1, im2, im3,
    mixup_prob=1.0, mixup_alpha=1.0,
    cutmix_prob=1.0, cutmix_alpha=1.0
):
    c = _cutmix(im2, cutmix_prob, cutmix_alpha)
    if c is None:
        return im1, im2, im3

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    v = np.random.beta(mixup_alpha, mixup_alpha)
    if mixup_alpha <= 0 or np.random.rand(1) >= mixup_prob:
        im2_aug = im2[rindex, :]
        im3_aug = im3[rindex, :]
        im1_aug = im1[rindex, :]

    else:
        im3_aug = v * im3 + (1-v) * im3[rindex, :]
        im2_aug = v * im2 + (1-v) * im2[rindex, :]
        im1_aug = v * im1 + (1-v) * im1[rindex, :]

    # apply mixup to inside or outside
    if np.random.random() > 0.5:
        im3[..., tcy:tcy+ch, tcx:tcx+cw] = im3_aug[..., fcy:fcy+ch, fcx:fcx+cw]
        im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2_aug[..., fcy:fcy+ch, fcx:fcx+cw]
        im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1_aug[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
    else:
        im3_aug[..., tcy:tcy+ch, tcx:tcx+cw] = im3[..., fcy:fcy+ch, fcx:fcx+cw]
        im2_aug[..., tcy:tcy+ch, tcx:tcx+cw] = im2[..., fcy:fcy+ch, fcx:fcx+cw]
        im1_aug[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
        im3, im2, im1 = im3_aug, im2_aug, im1_aug

    return im1, im2, im3


def cutblur(im1, im2, im3, prob=1.0, alpha=1.0):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2, im3

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)
    cy = np.random.randint(0, h-ch+1)
    cx = np.random.randint(0, w-cw+1)

    # apply CutBlur to inside or outside
    if np.random.random() > 0.5:
        im2[..., cy:cy+ch, cx:cx+cw] = im1[..., cy:cy+ch, cx:cx+cw]
        im3[..., cy:cy+ch, cx:cx+cw] = im1[..., cy:cy+ch, cx:cx+cw]
    else:
        im2_aug = im1.clone()
        im2_aug[..., cy:cy+ch, cx:cx+cw] = im2[..., cy:cy+ch, cx:cx+cw]
        im2 = im2_aug
        im3 = im2_aug
    return im1, im2, im3


def cutout(im1, im2, im3, prob=1.0, alpha=0.1):
    scale = im1.size(2) // im2.size(2)
    fsize = (im2.size(0), 1)+im2.size()[2:]

    if alpha <= 0 or np.random.rand(1) >= prob:
        fim2 = np.ones(fsize)
        fim2 = torch.tensor(fim2, dtype=torch.float, device=im2.device)
        fim3 = torch.tensor(fim2, dtype=torch.float, device=im3.device)
        fim1 = F.interpolate(fim2, scale_factor=scale, mode="nearest")
        return im1, im2, im3, fim1, fim2, fim3

    fim2 = np.random.choice([0.0, 1.0], size=fsize, p=[alpha, 1-alpha])
    fim2 = torch.tensor(fim2, dtype=torch.float, device=im2.device)
    fim3 = np.random.choice([0.0, 1.0], size=fsize, p=[alpha, 1-alpha])
    fim3 = torch.tensor(fim3, dtype=torch.float, device=im3.device)
    fim1 = F.interpolate(fim2, scale_factor=scale, mode="nearest")

    im2 *= fim2
    im3 *= fim3

    return im1, im2, im3, fim1, fim2, fim3


def rgb(im1, im2, im3, prob=1.0):
    if np.random.rand(1) >= prob:
        return im1, im2, im3

    perm = np.random.permutation(3)
    im1 = im1[:, perm]
    im2 = im2[:, perm]
    im3 = im2[:, perm]

    return im1, im2, im3
