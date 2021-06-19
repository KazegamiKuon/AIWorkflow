import torch
from torch import nn
from . import general as g

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.padding = (self.kernel_size[0]//2,self.kernel_size[1]//2)
