import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from .warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from .IFNet import *
import torch.nn.functional as F


from .Deformable_Conv.DeformableBlock import DeformableConvBlock

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
    )

def conv_woact(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
    )

class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

c = 16

class ContextNet(nn.Module):
    def __init__(self):
        super(ContextNet, self).__init__()
        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)
        self.deformconv1 = DeformableConvBlock(c , "add")
        self.deformconv2 = DeformableConvBlock(2*c , "add")
        self.deformconv3 = DeformableConvBlock(4*c , "add")
        self.deformconv4 = DeformableConvBlock(8*c , "add")

        #self.move_to_device()

    """def move_to_device(self):
        self.deformconv1.to(self.device)
        self.deformconv2.to(self.device)
        self.deformconv3.to(self.device)
        self.deformconv4.to(self.device)"""
        

    def forward(self, x, flow):
        x = self.conv1(x)
        # f1 = warp(x, flow)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f1 = self.deformconv1(x, flow)
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        # f2 = warp(x, flow)
        f2 = self.deformconv2(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        # f3 = warp(x, flow)
        f3 = self.deformconv3(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        # f4 = warp(x, flow)
        f4 = self.deformconv4(x, flow)
        f1 = F.interpolate(f1, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        f2 = F.interpolate(f2, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        f3 = F.interpolate(f3, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        f4 = F.interpolate(f4, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        return [f1, f2, f3, f4]

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.down0 = Conv2(6, 2*c, 1)
        self.down1 = Conv2(3*c, 3*c)
        self.down2 = Conv2(5*c, 5*c)
        self.down3 = Conv2(9*c, 9*c)
        self.up0 = deconv(17*c, 17*c)
        self.up1 = deconv(22*c, 4*c)
        self.up2 = deconv(7*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, 4, 3, 2, 1)

    def forward(self, warped_gt, lr, c0):
        s0 = self.down0(torch.cat((warped_gt, lr), 1))
        s1 = self.down1(torch.cat((s0, c0[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2]), 1))
        x = self.up0(torch.cat((s3, c0[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return x


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat



class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.contextnet = ContextNet()
        self.fusionnet = FusionNet()
        self.device()
        self.optimG = AdamW(itertools.chain(
            self.flownet.parameters(),
            self.contextnet.parameters(),
            self.fusionnet.parameters()), lr=1e-6, weight_decay=1e-4)
        self.schedulerG = optim.lr_scheduler.CyclicLR(
            self.optimG, base_lr=1e-6, max_lr=1e-3, step_size_up=8000, cycle_momentum=False)
        self.epe = EPE()
        self.ter = Ternary()
        self.sobel = SOBEL()
        # self.vgg = VGGPerceptualLoss().to(device)
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[
                               local_rank], output_device=local_rank)
            self.contextnet = DDP(self.contextnet, device_ids=[
                                  local_rank], output_device=local_rank)
            self.fusionnet = DDP(self.fusionnet, device_ids=[
                                 local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()
        self.contextnet.train()
        self.fusionnet.train()

    def eval(self):
        self.flownet.eval()
        self.contextnet.eval()
        self.fusionnet.eval()

    def device(self):
        self.flownet.to(device)
        self.contextnet.to(device)
        self.fusionnet.to(device)

    def load_model(self, path, rank=-1):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            self.flownet.load_state_dict(
                convert(torch.load('{}/flownet.pkl'.format(path), map_location=device)))
            self.contextnet.load_state_dict(
                convert(torch.load('{}/contextnet.pkl'.format(path), map_location=device)))
            self.fusionnet.load_state_dict(
                convert(torch.load('{}/unet.pkl'.format(path), map_location=device)))

    def save_model(self, path, rank):
        if rank == 0:
            torch.save(self.flownet.state_dict(), '{}/flownet.pkl'.format(path))
            torch.save(self.contextnet.state_dict(), '{}/contextnet.pkl'.format(path))
            torch.save(self.fusionnet.state_dict(), '{}/unet.pkl'.format(path))

    def predict(self, imgs, flow, training=True, flow_gt=None):
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        flow = F.interpolate(flow, scale_factor=2.0, mode="bilinear",
                             align_corners=False) * 2.0
        refine_output, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt = self.fusionnet(
            img0, img1, flow, c0, c1, flow_gt)
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = warped_img0 * mask + warped_img1 * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)
        if training:
            return pred, mask, merged_img, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt
        else:
            return pred

    def inference(self, img0, img1):
        imgs = torch.cat((img0, img1), 1)
        flow, _ = self.flownet(torch.cat((img0, img1), 1))
        return self.predict(imgs, flow, training=False)

    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()
        flow, flow_list = self.flownet(imgs)
        pred, mask, merged_img, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt = self.predict(
            imgs, flow, flow_gt=flow_gt)
        loss_ter = self.ter(pred, gt).mean()
        if training:
            with torch.no_grad():
                loss_flow = torch.abs(warped_img0_gt - gt).mean()
                loss_mask = torch.abs(
                    merged_img - gt).sum(1, True).float().detach()
                loss_mask = F.interpolate(loss_mask, scale_factor=0.5, mode="bilinear",
                                          align_corners=False).detach()
                flow_gt = (F.interpolate(flow_gt, scale_factor=0.5, mode="bilinear",
                                         align_corners=False) * 0.5).detach()
            loss_cons = 0
            for i in range(3):
                loss_cons += self.epe(flow_list[i][:, :2], flow_gt[:, :2], 1)
                loss_cons += self.epe(flow_list[i][:, 2:4], flow_gt[:, 2:4], 1)
            loss_cons = loss_cons.mean() * 0.01
        else:
            loss_cons = torch.tensor([0])
            loss_flow = torch.abs(warped_img0 - gt).mean()
            loss_mask = 1
        loss_l1 = (((pred - gt) ** 2 + 1e-6) ** 0.5).mean()
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_cons + loss_ter
            # loss_G = self.vgg(pred, gt) + loss_cons + loss_ter
            loss_G.backward()
            self.optimG.step()
        return pred, merged_img, flow, loss_l1, loss_flow, loss_cons, loss_ter, loss_mask