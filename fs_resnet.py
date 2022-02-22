import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
from Tools import *
from eval import *
from train import *
from fs_coding import *
from data import *

class Reshape(torch.nn.Module):
    def forward(self,x):
        return x.view(-1,1,28,28)


class Residual(nn.Module):

    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = fs_relu

    def forward(self,X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels,num_channels))
    return blk

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2 , padding=3),
                   nn.BatchNorm2d(64),
                   fs_relu(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

ResNet = nn.Sequential(b1, b2, b3, b4, b5,
                       nn.AdaptiveAvgPool2d((1,1)),
                       nn.Flatten(),
                       nn.Linear(512,10))

# batch_size = 128
# # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
# # lr, num_epochs = 0.01, 1
#
#
# train_dataset = EEGdata(root_dir="E:\EEG/train", labelfile="E:\EEG/train.txt")
# val_dataset = EEGdata(root_dir="E:\EEG/val", labelfile="E:\EEG/val.txt")
#
# train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
# val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
# lr, num_epochs = 0.01, 10


X = torch.randn(1, 1, 94, 94)
for layer in ResNet:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)

# if __name__ == '__main__':
#     train(ResNet, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())