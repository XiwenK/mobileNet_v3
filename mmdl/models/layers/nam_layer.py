import mmcv
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmcls.models.utils.make_divisible import make_divisible

import torch.nn as nn
import torch
from torch.nn import functional as F
import pdb


class Channel_Att(BaseModule):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
      
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)


    def forward(self, x):
        # pdb.set_trace()
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        x = torch.sigmoid(x) * residual #
        
        return x


class NAMLayer(BaseModule):
    def __init__(self, channels,shape, out_channels=None, no_spatial=True):
        super(NAMLayer, self).__init__()
        self.Channel_Att = Channel_Att(channels)
  
    def forward(self, x):
        x_out1=self.Channel_Att(x)
 
        return x_out1  