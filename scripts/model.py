import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import os
import sys
CUR_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append("{}/../".format(CUR_ROOT))

from scripts.utils import visualise

class Net(nn.Module):
    def __init__(self, mnist=True):
      
        super(Net, self).__init__()
        if mnist:
          num_channels = 1
        else:
          num_channels = 3
          
        self.conv1 = nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        if mnist:
          self.fc1 = nn.Linear(4*4*50, 500)
          self.flatten_shape = 4*4*50
        else:
          self.fc1 = nn.Linear(1250, 500)
          self.flatten_shape = 1250

        self.fc2 = nn.Linear(500, 10)
        
      
    def forward(self, x, vis=False, axs=None):
        X = 0
        y = 0

        if vis:
          axs[X,y].set_xlabel('Entry into network, input distribution visualised below: ')
          visualise(x, axs[X,y])

          axs[X,y+1].set_xlabel("Visualising weights of conv 1 layer: ")
          visualise(self.conv1.weight.data, axs[X,y+1])


        x = F.relu(self.conv1(x))

        if vis:
          axs[X,y+2].set_xlabel('Output after conv1 visualised below: ')
          visualise(x,axs[X,y+2])

          axs[X,y+3].set_xlabel("Visualising weights of conv 2 layer: ")
          visualise(self.conv2.weight.data, axs[X,y+3])

        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))

        if vis:
          axs[X,y+4].set_xlabel('Output after conv2 visualised below: ')
          visualise(x,axs[X,y+4])

          axs[X+1,y].set_xlabel("Visualising weights of fc 1 layer: ")
          visualise(self.fc1.weight.data, axs[X+1,y])

        x = F.max_pool2d(x, 2, 2)  
        x = x.view(-1, self.flatten_shape)
        x = F.relu(self.fc1(x))

        if vis:
          axs[X+1,y+1].set_xlabel('Output after fc1 visualised below: ')
          visualise(x,axs[X+1,y+1])

          axs[X+1,y+2].set_xlabel("Visualising weights of fc 2 layer: ")
          visualise(self.fc2.weight.data, axs[X+1,y+2])

        x = self.fc2(x)

        if vis:
          axs[X+1,y+3].set_xlabel('Output after fc2 visualised below: ')
          visualise(x,axs[X+1,y+3])

        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
   
        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def model_main(mnist=True):
 
    batch_size = 64
    test_batch_size = 64
    epochs = 2
    lr = 0.01
    momentum = 0.5
    seed = 1
    log_interval = 500
    save_model = False
    no_cuda = False
    
    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if mnist:
      train_loader = torch.utils.data.DataLoader(
          datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
          batch_size=batch_size, shuffle=True, **kwargs)
      
      test_loader = torch.utils.data.DataLoader(
          datasets.MNIST('../data', train=False, transform=transforms.Compose([
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
    for epoch in range(1, epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
    
    return model