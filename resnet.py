import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
from Tools import *
from eval import *
from train import *
from data import *
from d2l import torch as d2l
from collections import OrderedDict

class Reshape(torch.nn.Module):
    def forward(self,x):
        return x.view(-1,1,128,128)

class IBN(nn.Module):

    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out



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

        # self.bn1 = IBN(num_channels)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)


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
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))


ResNet = nn.Sequential(b1, b2, b3, b4, b5,
                       nn.AdaptiveAvgPool2d((1,1)),
                       nn.Flatten(),

                       nn.Linear(512,2))

# batch_size = 128
# train_iter, val_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
# lr, num_epochs = 1e-4, 10





batch_size = 128
train_dataset = EEGdata(root_dir="E:\EEG/alpha_img/train", labelfile="E:\EEG/alpha_img/train.txt")
val_dataset = EEGdata(root_dir="E:\EEG/alpha_img/val", labelfile="E:\EEG/alpha_img/val.txt")
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
lr, num_epochs = 1e-4, 10



# X = torch.randn([128,1,128,128])
# for layer in ResNet:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape: \t', X.shape)

if __name__ == '__main__':
    # Net = ResNet
    # checkpoint = torch.load("D:\FS/checkpoints/test.pth")
    # Net.load_state_dict(checkpoint)
    device = 'cuda:0'
    # net = models.resnet18(pretrained=False)
    # net.fc = nn.Linear(512,2)
    # net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2 , padding=3)
    # net.load_state_dict(torch.load("D:\FS/checkpoints/alpha4.pth"))
    # print(net)
    net = ResNet
    net.load_state_dict(torch.load("D:\FS/checkpoints/alpha1.pth"))
    # x = torch.load("D:\FS/checkpoints/alpha1.pth")
    # new_dict = OrderedDict()
    # for key in x:
    #     if key == '7.weight':
    #         new_dict['8.weight'] = x[key]
    #     elif key == '7.bias':
    #         new_dict['8.bias'] = x[key]
    #     else:
    #         new_dict[key] = x[key]
    #
    #
    # net.load_state_dict(new_dict)

    # net.load_state_dict(torch.hub.load_state_dict_from_url('https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth'))

    train(net, train_iter, val_iter, num_epochs, lr,device,use_pretrained_model=True,visualize=True)




