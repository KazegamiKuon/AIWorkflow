import unittest
import torch
from torch import nn
from lib.model.resnet import ResNetBasicBlock, ResNetBottleNeckBlock, ResNetLayer

class TestAutoLayer(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)
        pass

    def test_resnet_basic_block(self):
        dummy = torch.ones((1, 32, 224, 224))
        block = ResNetBasicBlock(32, 64)
        print(block(dummy).shape)
        print(block)
        pass
    
    def test_resnet_bottleneck(self):
        dummy = torch.ones((1, 32, 10, 10))
        block = ResNetBottleNeckBlock(32, 64)
        print(block(dummy).shape)
        print(block)
        pass

    def test_resnet_layer(self):
        dummy = torch.ones((1, 64, 48, 48))
        layer = ResNetLayer(64, 128, block=ResNetBasicBlock, n=3)
        print(layer(dummy).shape)
        pass

if __name__ == '__main__':
    unittest.main()