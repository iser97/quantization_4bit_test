import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
CUR_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append("{}/../../".format(CUR_ROOT))

from scripts.utils import quantize_tensor, quantize_tensor_sym, visualise, dequantize_tensor, dequantize_tensor_sym, QTensor

def quantizeLayer(x, layer, stat, scale_x, zp_x, vis=False, axs=None, X=None, y=None, sym=False, num_bits=8):
  # for both conv and linear layers
  # cache old values
  W = layer.weight.data
  B = layer.bias.data

  # WEIGHTS SIMULATED QUANTISED

  # quantise weights, activations are already quantised
  if sym:
    w = quantize_tensor_sym(layer.weight.data,num_bits=num_bits) 
    b = quantize_tensor_sym(layer.bias.data,num_bits=num_bits)
  else:
    w = quantize_tensor(layer.weight.data, num_bits=num_bits) 
    b = quantize_tensor(layer.bias.data, num_bits=num_bits)

  layer.weight.data = w.tensor.float()
  layer.bias.data = b.tensor.float()

  ## END WEIGHTS QUANTISED SIMULATION


  if vis:
    axs[X,y].set_xlabel("Visualising weights of layer: ")
    visualise(layer.weight.data, axs[X,y])

  # QUANTISED OP, USES SCALE AND ZERO POINT TO DO LAYER FORWARD PASS. (How does backprop change here ?)
  # This is Quantisation Arithmetic
  scale_w = w.scale
  zp_w = w.zero_point
  scale_b = b.scale
  zp_b = b.zero_point
  
  if sym:
    scale_next, zero_point_next = calcScaleZeroPointSym(min_val=stat['min'], max_val=stat['max'])
  else:
    scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'])

  # Preparing input by saturating range to num_bits range.
  if sym:
    X = x.float()
    layer.weight.data = ((scale_x * scale_w) / scale_next)*(layer.weight.data)
    layer.bias.data = (scale_b/scale_next)*(layer.bias.data)
  else:
    X = x.float() - zp_x
    layer.weight.data = ((scale_x * scale_w) / scale_next)*(layer.weight.data - zp_w)
    layer.bias.data = (scale_b/scale_next)*(layer.bias.data + zp_b)

  # All int computation
  if sym:  
    x = (layer(X)) 
  else:
    x = (layer(X)) + zero_point_next 
  
  # cast to int
  x.round_()

  # Perform relu too
  x = F.leaky_relu(x)

  # Reset weights for next forward pass
  layer.weight.data = W
  layer.bias.data = B
  
  return x, scale_next, zero_point_next

def quantForward(model, x, stats, vis=False, axs=None, sym=False, num_bits=8):
  X = 0
  y = 0
  # Quantise before inputting into incoming layers
  if sym:
    x = quantize_tensor_sym(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'], num_bits=num_bits)
  else:
    x = quantize_tensor(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'], num_bits=num_bits)

  if vis:
    axs[X,y].set_xlabel('Entry into network, input distribution visualised below: ')
    visualise(x.tensor, axs[X,y])
  
  x, scale_next, zero_point_next = quantizeLayer(x.tensor, model.conv1, stats['conv2'], x.scale, x.zero_point, vis, axs, X=X, y=y+1, sym=sym, num_bits=num_bits)

  x = F.max_pool2d(x, 2, 2)
  
  if vis:
    axs[X,y+2].set_xlabel('Output after conv1 visualised below: ')
    visualise(x,axs[X,y+2])

  x, scale_next, zero_point_next = quantizeLayer(x, model.conv2, stats['fc1'], scale_next, zero_point_next, vis, axs, X=X, y=y+3, sym=sym, num_bits=num_bits)

  x = F.max_pool2d(x, 2, 2)

  if vis:
    axs[X,y+4].set_xlabel('Output after conv2 visualised below: ')
    visualise(x,axs[X,y+4])

  x = x.view(-1, 4*4*50)

  x, scale_next, zero_point_next = quantizeLayer(x, model.fc1, stats['fc2'], scale_next, zero_point_next, vis, axs, X=X+1, y=0, sym=sym, num_bits=num_bits)

  if vis:
    axs[X+1,1].set_xlabel('Output after fc1 visualised below: ')
    visualise(x,axs[X+1,1])
  
  # Back to dequant for final layer
  if sym:
    x = dequantize_tensor_sym(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))
  else:
    x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))

  if vis:
    axs[X+1,2].set_xlabel('Output after fc1 but dequantised visualised below: ')
    visualise(x,axs[X+1,2])

  x = model.fc2(x)

  if vis:
    axs[X+1,3].set_xlabel('Unquantised Weights of fc2 layer')
    visualise(model.fc2.weight.data,axs[X+1,3])

    axs[X+1,2].set_xlabel('Output after fc2 but dequantised visualised below: ')
    visualise(x,axs[X+1,4])

  return F.log_softmax(x, dim=1)

def testQuant(model, test_loader, quant=False, stats=None, sym=False, num_bits=8):
    device = 'cuda'
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if quant:
              output = quantForward(model, data, stats, sym=sym, num_bits=num_bits)
            else:
              output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

