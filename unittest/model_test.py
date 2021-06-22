import unittest
import torch
from torch import nn
import torchvision.models as models
from torchsummary import summary
from lib.model.resnet import ResNetBasicBlock, ResNetBottleNeckBlock, ResNetLayer, resnet18

class TestAutoLayer(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)
        pass

    def test_resnet_basic_block_test(self):
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
    
    def test_resnet18(self):
        model = resnet18(3, 1000)
        summary(model, (3, 224, 224),depth=5)
        summary(models.resnet18(False), (3, 224, 224))
        pass
    
    def test_resnet18_data(self):
        summary(models.resnet18(False), (3, 224, 224))
        pass

if __name__ == '__main__':
    unittest.main()