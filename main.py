import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from scripts.quant.aware import mainQuantAware, testQuantAware_last
from scripts.quant.normal_quant import testQuant
from scripts.model import model_main
from scripts.utils import gatherStats

# model = model_main()
# q_model = copy.deepcopy(model) 
# kwargs = {'num_workers': 1, 'pin_memory': True}
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=512, shuffle=True, **kwargs)
# stats = gatherStats(q_model, test_loader)
# testQuant(q_model, test_loader, quant=True, num_bits=8, stats=stats, sym=True)


model, old_stats = mainQuantAware()
kwargs = {'num_workers': 1, 'pin_memory': True}
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True, **kwargs)

"""## Test Quant Aware """

print(old_stats)

import copy
q_model = copy.deepcopy(model)

testQuantAware_last(q_model, test_loader, stats=old_stats, sym=False, num_bits=4, record=True)