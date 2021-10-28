import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import os
import sys
CUR_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append("{}/../../".format(CUR_ROOT))
import json
from scripts.utils import quantize_tensor, dequantize_tensor, updateStats, quantize_tensor_sym
from scripts.model import Net

def record_tensor(tensor, tensor_name_scope):
    shape = list(tensor.tensor.size())
    weight = tensor.tensor.cpu().tolist()
    try:
      scale = tensor.scale.cpu().tolist()
    except:
      scale = tensor.scale
    zero_point= tensor.zero_point
    dic = {'name_scope':tensor_name_scope, 'weight':weight, 'scale':scale, 'zero_point':zero_point, 'shape':shape}
    dic = json.dumps(dic)
    dic = dic + '\n'
    print(dic, file=open('weight.txt', mode='a+', encoding='utf-8'))  

class FakeQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8, record=False, name_scope=None, min_val=None, max_val=None):              
        x = quantize_tensor(x, num_bits=num_bits, min_val=min_val, max_val=max_val)
        if record:
            record_tensor(x, name_scope)
        x = dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None

def forward_layer(x, model, stats, num_bits):
    x = quantize_tensor_sym(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'], num_bits=num_bits)
    pass

def quantAwareTrainingForward(model, x, stats, vis=False, axs=None, sym=False, num_bits=8, act_quant=False, record=False):
  # if sym:
  #   x = quantize_tensor_sym(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'], num_bits=num_bits)
  # else:
  #   x = quantize_tensor(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'], num_bits=num_bits)
  x = FakeQuantOp.apply(x, num_bits, False, 'conv1input')
  conv1weight = model.conv1.weight.data
  model.conv1.weight.data = FakeQuantOp.apply(model.conv1.weight.data, num_bits, record, 'conv1weight')
  x = F.relu(model.conv1(x))

  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')

  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, record, 'conv1action', stats['conv1']['ema_min'], stats['conv1']['ema_max'])

  x = F.max_pool2d(x, 2, 2)
 
  conv2weight = model.conv2.weight.data
  model.conv2.weight.data = FakeQuantOp.apply(model.conv2.weight.data, num_bits, record, 'conv2weight')
  x = F.relu(model.conv2(x))

  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')
    
  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, record, 'conv2action', stats['conv2']['ema_min'], stats['conv2']['ema_max'])


  x = F.max_pool2d(x, 2, 2)

  x = x.view(-1, 4*4*50)

  fc1weight = model.fc1.weight.data
  model.fc1.weight.data = FakeQuantOp.apply(model.fc1.weight.data, num_bits, record, 'fc1weight')
  x = F.relu(model.fc1(x))

  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc1')

  if act_quant:
    x = FakeQuantOp.apply(x, num_bits, record, 'fc1action', stats['fc1']['ema_min'], stats['fc1']['ema_max'])

  fc2weight = model.fc2.weight.data
  model.fc2.weight.data = FakeQuantOp.apply(model.fc2.weight.data, num_bits, record, 'fc2weight')
  x = model.fc2(x)
  
  with torch.no_grad():
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc2')

  return F.log_softmax(x, dim=1), conv1weight, conv2weight, fc1weight, fc2weight, stats

"""# Train using Quantization Aware Training"""

def trainQuantAware(args, model, device, train_loader, optimizer, epoch, stats, act_quant=False, num_bits=4):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, conv1weight, conv2weight, fc1weight, fc2weight, stats = quantAwareTrainingForward(model, data, stats, num_bits=num_bits, act_quant=act_quant, record=False)

        model.conv1.weight.data = conv1weight
        model.conv2.weight.data = conv2weight
        model.fc1.weight.data = fc1weight
        model.fc2.weight.data = fc2weight

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return stats

def testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=4):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, conv1weight, conv2weight, fc1weight, fc2weight, _ = quantAwareTrainingForward(model, data, stats, num_bits=num_bits, act_quant=act_quant, record=False)
            
            model.conv1.weight.data = conv1weight
            model.conv2.weight.data = conv2weight
            model.fc1.weight.data = fc1weight
            model.fc2.weight.data = fc2weight

            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def mainQuantAware(mnist=True):
 
    batch_size = 64
    test_batch_size = 64
    epochs = 2
    lr = 0.01
    momentum = 0.5
    seed = 1
    log_interval = 500
    save_model = False

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True}

    if mnist:
      train_loader = torch.utils.data.DataLoader(
          datasets.MNIST('/home/zjh/python_program/data/', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
          batch_size=batch_size, shuffle=True, **kwargs)
      
      test_loader = torch.utils.data.DataLoader(
          datasets.MNIST('/home/zjh/python_program/data/', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
          batch_size=test_batch_size, shuffle=True, **kwargs)
    else:
      transform = transforms.Compose(
          [transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

      trainset = datasets.CIFAR10(root='./dataCifar', train=True,
                                              download=True, transform=transform)
      train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

      testset = datasets.CIFAR10(root='./dataCifar', train=False,
                                            download=True, transform=transform)
      test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                              shuffle=False, num_workers=2)
      
    model = Net(mnist=mnist).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    args = {}
    args["log_interval"] = log_interval
    num_bits = 4
    stats = {}
    for epoch in range(1, epochs + 1):
        if epoch > 5:
          act_quant = True 
        else:
          act_quant = False

        stats = trainQuantAware(args, model, device, train_loader, optimizer, epoch, stats, act_quant, num_bits=num_bits)
        testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=num_bits)

    if (save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")

    return model, stats

def testQuantAware_last(model, test_loader, stats=None, sym=False, num_bits=4, record=False):
    model.eval()
    test_loss = 0
    correct = 0
    device = 'cuda'
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, conv1weight, conv2weight, fc1weight, fc2weight, _ = quantAwareTrainingForward(model, data, stats, num_bits=num_bits, act_quant=True, sym=False, record=record)
            record = False
            model.conv1.weight.data = conv1weight
            model.conv2.weight.data = conv2weight
            model.fc1.weight.data = fc1weight
            model.fc2.weight.data = fc2weight

            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
