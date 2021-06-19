import unittest
from torch import nn
from torch.nn.modules import activation
from lib.model.utils import auto_layer as al
from lib.model.utils import general as g

class TestAutoLayer(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)
        pass

    def test_Conv2DAuto(self):
        conv = al.Conv2dAuto(kernel_size=5,bias=False,in_channels=32,out_channels=64)
        print(conv)
        del conv
    
    def test_something(self):
        print(nn.ReLU().__dict__)
        pass

if __name__ == '__main__':
    unittest.main()