from torch import nn
from ..utils.activations import get_activation

class ResidualBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.blocks = nn.Identity()
        self.activate = get_activation(self.activation,True)
        self.shortcut = nn.Identity()
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

    def forward(self,x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        if self.activate is not None:
            x = self.activate(x)
        return x