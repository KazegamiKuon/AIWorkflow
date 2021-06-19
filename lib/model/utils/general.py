
import json
from torch import nn

def padding_convnn(w:int,f:int,s:int) -> int:
    """
    calculate padding type same
    output size = (W−F+2P)/S+1
    
    Parameters
    ----------
    w : int
        input size
    f : int
        filter size
    s : int
        strides
    
    Returns
    -------
    p : int
        padding size
    """
    return ((w-1)*s-w+f)//2

def get_activations()->nn.ModuleDict:
    """
    auto create ativation
    
    Returns
    -------
    activations: nn.ModuleDict
        dict activation was created.
    """
    activations = nn.ModuleDict([
        ['lrelu',nn.LeakyReLU()],
        ['relu',nn.ReLU()]
    ])    
    return activations