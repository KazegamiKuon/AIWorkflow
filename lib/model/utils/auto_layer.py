from functools import partial
from torch import nn
from . import general as g
from ..resnet import ResNet, ResNetBasicBlock, ResNetBottleNeckBlock

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.padding = (self.kernel_size[0]//2,self.kernel_size[1]//2)

conv2d3x3 = partial(Conv2dAuto,kernel_size=3,bias=False)

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))