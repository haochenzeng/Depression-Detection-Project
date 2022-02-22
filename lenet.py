import torch.nn as nn
import torch.nn.functional as F
import torch
from d2l import torch as d2l
from Tools import *
from eval import *
from train import *
from data import *


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # input(3,32,32) output(16,28,28)
        x = self.pool1(x)  # output(16，14，14)
        x = F.relu(self.conv2(x))  # output(32,10.10)
        x = self.pool2(x)  # output(32,5,5)
        x = x.view(-1, 32 * 5 * 5)  # output(5*5*32)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x

# #model调试
# import torch

# #定义shape
# input1 = torch.rand([32,3,32,32])
# model = LeNet()#实例化
# print(model)
# #输入网络中
# output = model(input1)


batch_size = 128
train_dataset = EEGdata(root_dir="E:\EEG/train", labelfile="E:\EEG/train.txt")
val_dataset = EEGdata(root_dir="E:\EEG/val", labelfile="E:\EEG/val.txt")
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,num_workers=4)
val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=4)
lr, num_epochs = 1e-3, 10

if __name__ == '__main__':
    # Net = ResNet
    # checkpoint = torch.load("D:\FS/checkpoints/test.pth")
    # Net.load_state_dict(checkpoint)
    device = 'cuda:0'
    net = LeNet()
    # net.load_state_dict(torch.load("D:\FS/checkpoints/pretrained_resnet18.pth"))
    train(net, train_iter, val_iter, num_epochs, lr,device,use_pretrained_model=False)