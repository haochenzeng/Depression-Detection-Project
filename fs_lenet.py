import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
from Tools import *
from eval import *
from train import *
from fs_coding import*
import warnings
from data import *

warnings.filterwarnings("ignore")
n_images = 0

class Reshape(torch.nn.Module):
    def forward(self,x):
        return x.view(-1, 1, 28, 28)

class Calculate(torch.nn.Module):
    def forward(self,x):
        global n_images
        n_images += 1
        return x

LeNet = torch.nn.Sequential(
    # replace_sigmoid_with_fs(),
    # Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(14400, 120),nn.Sigmoid(),
    nn.Linear(120, 84),nn.Sigmoid(),
    nn.Linear(84, 2)
)

FS_LeNet = torch.nn.Sequential(
    # replace_sigmoid_with_fs(),
    Calculate(),
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2),fs_sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), fs_sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),fs_sigmoid(),
    nn.Linear(120, 84),fs_sigmoid(),
    nn.Linear(84, 2)
)

# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
# lr, num_epochs = 0.9, 1
# for X,y in test_iter:
#     print(X.shape)

# j = 0
# for i, data in enumerate(test_iter):
#     j += 1
# print(j)


# X = torch.randn(128,1,128,128)
# for layer in LeNet:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape: \t', X.shape)

# if __name__ == '__main__':
#     # warnings.filterwarnings("ignore")
#     train(FS_LeNet, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

batch_size = 64
train_dataset = EEGdata(root_dir="E:\EEG/alpha_img/train", labelfile="E:\EEG/alpha_img/train.txt")
val_dataset = EEGdata(root_dir="E:\EEG/alpha_img/val", labelfile="E:\EEG/alpha_img/val.txt")
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
lr, num_epochs = 5e-3, 30

if __name__ == '__main__':
    # Net = ResNet
    # checkpoint = torch.load("D:\FS/checkpoints/test.pth")
    # Net.load_state_dict(checkpoint)
    device = 'cuda:0'
    net = LeNet
    # net.load_state_dict(torch.load("D:\FS/checkpoints/pretrained_resnet18.pth"))
    train(net, train_iter, val_iter, num_epochs, lr,device,use_pretrained_model=False,visualize=True)