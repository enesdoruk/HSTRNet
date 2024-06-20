import torch

from model.HSTR_RIFE_v5_scaled import HSTRNet
from model.RIFE_v5.swin_v2 import Swin_V2

device = "cuda:0"

# model = HSTRNet(device)

lr = torch.randn((1, 3, 256, 256)).to(device)
ref = torch.randn((1, 3, 256, 256)).to(device)

model = Swin_V2(3).cuda()
model(lr, ref)

# imgs = torch.cat((in1, in2), 1)
# pred = model(imgs)