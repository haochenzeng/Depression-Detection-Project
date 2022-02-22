import torch
import torchvision.models as models
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
from Tools import *
from eval import *
from train import *
from fs_coding import *
from lenet import *
from resnet import *

Net = ResNet
checkpoint = torch.load("D:\FS/checkpoints/resnet18.pth")
Net.load_state_dict(checkpoint)
# replace_relu(resnet18)
convert_relu(Net)

if __name__ == '__main__':
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    test_acc = evaluate(Net,test_iter)
    print(test_acc)



