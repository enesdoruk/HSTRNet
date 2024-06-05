import torch
import torch.nn as nn

from .Dynamic_offset_estimator import Dynamic_offset_estimator, Dynamic_offset_estimator_concat


from mmcv.ops.deform_conv import DeformConv2d

class DeformableConvBlock(nn.Module):
    def __init__(self, input_channels, mode):
        super(DeformableConvBlock, self).__init__()
        if mode == "concat":
            self.offset_estimator = Dynamic_offset_estimator(input_channelsize=input_channels * 2)
            self.offset_conv = nn.Conv2d(in_channels=input_channels * 2, out_channels=1 * 2 * 9, kernel_size=3, padding=1,
                                         bias=True)
        else:
            self.offset_estimator = Dynamic_offset_estimator(input_channelsize=input_channels + 2)
            self.offset_conv = nn.Conv2d(in_channels=input_channels + 2, out_channels=1 * 2 * 9, kernel_size=3, padding=1,
                                         bias=True)

        self.deformconv = DeformConv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3,
                                       padding=1)

    def forward(self, lr_features, hr_features):
        input_offset = torch.cat((lr_features, hr_features), dim=1)
        estimated_offset = self.offset_estimator(input_offset)
        estimated_offset = self.offset_conv(estimated_offset)
        output = self.deformconv(x=lr_features, offset=estimated_offset)
        return output