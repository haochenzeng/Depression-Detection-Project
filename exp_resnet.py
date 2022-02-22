import torch

from CNN5 import *
from data import *
from resnet_test import *

net = ResNet
# convert_relu(net)
convert_relu(net)

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


if __name__ == '__main__':
    net.load_state_dict(torch.load("D:\FS/checkpoints/resnet18_Cifar10_2.pth"))
    net.to(device='cuda:0')
    test_acc = evaluate(net, test_iter, 'cuda:0')
    print(f'converting SNN test_acc={test_acc}')
    # print(np.sum(spike))
