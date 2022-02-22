import numpy as np
import torch
from Tools import *

def accuracy(y_hat, y):
    if len(y_hat.shape) >1 and y_hat.shape[1] >1 :
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
    # return cmp.type(y.dtype).item()

def two_classes_accuracy(y_hat, y):
    index = 0
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        index = y_hat.argmax(axis=1).item()
    temp = torch.tensor([y_hat[0][index]])
    # if temp >= 0.3 and temp <= 0.7:
    #     return 0.
    if temp > 0.5:
        cmp = torch.tensor([1],device='cuda:0')
    # elif temp < 0.5:
    else:
        cmp = torch.tensor([0],device='cuda:0')
    x = cmp == y
    return float(x.type(y.dtype).sum())


def evaluate(net, data_iter, device=None,index=0):
    acc = []
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    for X,y in data_iter:
        print(index)
        index += 1
        if isinstance(X,list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(accuracy(net(X),y),y.numel())
        # acc.append(accuracy(net(X),y))
    return metric[0] / metric[1]
    # return acc