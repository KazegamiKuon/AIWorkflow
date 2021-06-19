from torch import nn
from ..utils import general as g

class ResidualBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.blocks = nn.Identity()
        self.activation = g.get_activations()[self.activation]
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
        x = self.activation(x)
        return x
        