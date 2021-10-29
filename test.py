import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from scripts.quant.aware import mainQuantAware, testQuantAware_last, FakeQuantOp
from scripts.quant.normal_quant import testQuant
from scripts.model import model_main
from scripts.utils import gatherStats

kwargs = {'num_workers': 1, 'pin_memory': True}
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=10000, shuffle=True, **kwargs)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=60000, shuffle=True, **kwargs)
for data, target in test_loader:
    x = FakeQuantOp.apply(data, 4, True, 'test_input')
for data, target in train_loader:
    x = FakeQuantOp.apply(data, 4, True, 'train_input')