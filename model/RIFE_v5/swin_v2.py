import torch
import cv2
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 

from utils.utils import image_show
from utils.utils import padding_swin, crop_swin
from timm.models.layers import trunc_normal_


import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('swin')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Patch_Merging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = x.view(B, int(H / 2), int(W / 2), 2 * C)
        return x

class Swin_V2_block(nn.Module):
    def __init__(self, in_channel, window_size = 5, shift_size = 2, dim=48, is_merge = False ):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.qkv_norm = nn.LayerNorm(dim * 3)
        self.qkv_prelu = nn.PReLU(dim * 3)
        self.softmax = nn.Softmax(1)
        self.num_heads = 3
        self.window_size = window_size
        self.patch_size = dim
        self.shift_size = shift_size
        self.dim = dim
        self.in_channel = in_channel
        self.merge = Patch_Merging(dim)
        self.is_merge = is_merge

        # self.gating_param = nn.Parameter(torch.ones(self.num_heads), requires_grad= True)
        # self.softmax = nn.Softmax(dim=-1) 

        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(0)

        # self.scale = (self.dim // self.num_heads) ** 0.5

        # get pair-wise relative position index for each token inside the window
        # coords_h = torch.arange(self.window_size)
        # coords_w = torch.arange(self.window_size)
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.window_size - 1
        # relative_coords[:, :, 0] *= 2 * self.window_size - 1
        # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

        # if self.shift_size > 0:
        #     H, W = self.window_size, self.window_size
        #     img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        #     h_slices = (slice(0, -self.window_size),
        #                 slice(-self.window_size, -self.shift_size),
        #                 slice(-self.shift_size, None))
        #     w_slices = (slice(0, -self.window_size),
        #                 slice(-self.window_size, -self.shift_size),
        #                 slice(-self.shift_size, None))
        #     cnt = 0
        #     for h in h_slices:
        #         for w in w_slices:
        #             img_mask[:, h, w, :] = cnt
        #             cnt += 1

        #     mask_windows = self.window_partition(img_mask)
        #     mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            
        #     self.attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        #     self.attn_mask = self.attn_mask.masked_fill(self.attn_mask != 0, float(-100.0)).masked_fill(self.attn_mask == 0, float(0.0))
        #     self.attn_mask = self.attn_mask.cuda()
        # else:
        #     self.attn_mask = None


    def window_partition(self, x):
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        return windows

    def window_reverse(self, x, H, W):
        B = int(x.shape[0] / (H * W / self.window_size / self.window_size))
        x = x.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def attention(self, lr_windows, ref_windows, mask=None):
        B_, N, C = lr_windows.shape

        qkv_lr = self.softmax(self.qkv_norm(self.qkv(lr_windows))).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_ref = self.softmax(self.qkv_norm(self.qkv(ref_windows))).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q_lr, k_lr, v_lr = qkv_lr[0], qkv_lr[1], qkv_lr[2]
        q_ref, k_ref, v_ref = qkv_ref[0], qkv_ref[1], qkv_ref[2]

        # q_lr = q_lr * self.scale

        # gat_1 = (q_lr @ k_ref.transpose(-2, -1))
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # gat_1 = gat_1 + relative_position_bias.unsqueeze(0)

        # gat_2 = (q_lr @ k_lr.transpose(-2, -1))
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # gat_2 = gat_2 + relative_position_bias.unsqueeze(0)


        # if mask is not None:
        #     nW = mask.shape[0]
        #     gat_1 = gat_1.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        #     gat_1 = gat_1.view(-1, self.num_heads, N, N)

        #     gat_2 = gat_2.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        #     gat_2 = gat_2.view(-1, self.num_heads, N, N)
        #     gat_1 = self.softmax(gat_1)
        #     gat_2 = self.softmax(gat_2)

        # else:
        #     gat_1 = self.softmax(gat_1)
        #     gat_2 = self.softmax(gat_2)


        # gating = (self.gating_param).view(1,-1,1,1) 
        # attn = ((1.-torch.sigmoid(gating)) * gat_1) + (torch.sigmoid(gating) * gat_2)

        # x = (attn @ v_ref).transpose(1, 2).reshape(B_, N, C)

        # x = self.proj(x)
        # x = self.proj_drop(x)
        
        attn1 = (q_lr @ k_ref.transpose(-2, -1))
        attn2 = (q_ref @ k_lr.transpose(-2, -1))
        
        x = (attn1 @ v_ref)
        x = (attn2 @ x).transpose(1, 2).reshape(B_, N, C)
        
        return x

    def forward(self, lr_patch, ref_patch):
        B_p, H_p, W_p, C_p = lr_patch.shape

        if self.shift_size > 0:
            shifted_lr_patch = torch.roll(lr_patch, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_ref_patch = torch.roll(ref_patch, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_lr_patch = lr_patch
            shifted_ref_patch = ref_patch


        lr_windows = self.window_partition(shifted_lr_patch)
        ref_windows = self.window_partition(shifted_ref_patch)

        B_w, w_s, w_s, C_w = lr_windows.shape
        lr_windows = lr_windows.view(-1, self.window_size * self.window_size, C_w)  # nW*B, window_size*window_size, C
        ref_windows = ref_windows.view(-1, self.window_size * self.window_size, C_w)  # nW*B, window_size*window_size, C
        
        attn_windows = self.attention(lr_windows, ref_windows, mask=None)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C_w)
        

        if self.shift_size > 0:
            shifted_x = self.window_reverse(attn_windows, H_p, W_p)  
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = self.window_reverse(attn_windows, H_p, W_p) 
            x = shifted_x
        
        if self.is_merge:
            x = self.merge(x.permute(0, 3, 1, 2))
            ref_patch = self.merge(ref_patch.permute(0, 3, 1, 2))

        return x, ref_patch


class Swin_V2(nn.Module):
    def __init__(self, in_channel, window_size = 7, dim=48):
        super().__init__()
        self.dim = dim
        self.number_of_layers = 5
        self.window_size = window_size
        self.conv = nn.Conv2d(in_channel, self.dim, kernel_size=4, stride=4)
        self.prelu = nn.PReLU(self.dim)
        self.block00 = Swin_V2_block(in_channel=in_channel, window_size=window_size, shift_size=0, dim=dim, is_merge=False).to(device)
        self.block01 = Swin_V2_block(in_channel=in_channel, window_size=window_size, shift_size=0, dim=dim, is_merge=False).to(device)
        self.block10 = Swin_V2_block(in_channel=in_channel, window_size=window_size, shift_size=2, dim=dim, is_merge = True).to(device)
        self.block11 = Swin_V2_block(in_channel=in_channel, window_size=window_size, shift_size=2, dim=dim * 2, is_merge = False).to(device)
        self.block20 = Swin_V2_block(in_channel=in_channel, window_size=window_size, shift_size=0, dim=dim * 2, is_merge = True).to(device)
        self.block21 = Swin_V2_block(in_channel=in_channel, window_size=window_size, shift_size=0, dim=dim * 4, is_merge = False).to(device)


    def patch_partition(self, x):
        B,H, W, C = x.shape
        x = self.prelu(self.conv(x)).permute(0, 2, 3, 1)
        return x

    def forward(self, lr, ref):
        B_o, C_o, H_o, W_o = lr.shape

        padded_lr = padding_swin(lr, self.dim, self.window_size)
        padded_ref = padding_swin(ref, self.dim, self.window_size)

        B_, C, H, W = padded_lr.shape

        lr_patch = self.patch_partition(padded_lr)
        ref_patch = self.patch_partition(padded_ref)

        x, ref_patch = self.block00(lr_patch, ref_patch)
        x, ref_patch = self.block01(x, ref_patch)
        
        x_r0 = x.permute(0, 3, 1, 2)
        x_r0 = crop_swin(x_r0, lr.shape[2] / 4, lr.shape[3] / 4)

        x, ref_patch = self.block10(x, ref_patch)
        x, ref_patch = self.block11(x, ref_patch)

        x_r1 = x.permute(0, 3, 1, 2)
        x_r1 = crop_swin(x_r1, lr.shape[2] / 8, lr.shape[3] / 8)

        x, ref_patch = self.block20(x, ref_patch)
        x, ref_patch = self.block21(x, ref_patch)

        x_r2 = x.permute(0, 3, 1, 2)
        x_r2 = crop_swin(x_r2, lr.shape[2] / 16, lr.shape[3] / 16)
        
        
        return x_r0, x_r1, x_r2
        

        

