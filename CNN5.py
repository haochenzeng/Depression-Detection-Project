import torch
import torch.nn as nn
from Tools import *
from eval import *
from train import *
from data import *
from collections import OrderedDict

spike = []


class spike_counting(torch.nn.Module):
    def forward(self, x):
        spikes = 0
        for i in x:
            m = i.flatten()
            y = torch.where(m > 0, torch.ones_like(m), torch.zeros_like(m))
            # print(y.shape)
            spikes += sum(y).item()
        print(f'Spikes: {spikes}')
        spike.append(spikes)
        relu = nn.ReLU(inplace = True)
        return relu(x)


CNN5_1 = nn.Sequential(
    nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3),
    nn.MaxPool2d(kernel_size=3),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
    nn.MaxPool2d(kernel_size=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Dropout(p=0.5),
    nn.Linear(15488,2)
)

CNN5_2 = nn.Sequential(
    nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3),
    nn.MaxPool2d(kernel_size=2),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    spike_counting(),
    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
    nn.MaxPool2d(kernel_size=2),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    spike_counting(),
    nn.Dropout(p=0.5),
    nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3),
    nn.ReLU(),
    spike_counting(),
    nn.Flatten(),
    nn.Dropout(p=0.5),
    nn.Linear(100352,2)
)



CNN8 = nn.Sequential(
    nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3),
    nn.MaxPool2d(kernel_size=2),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
    nn.MaxPool2d(kernel_size=2),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3),
    nn.MaxPool2d(kernel_size=2),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3),
    nn.MaxPool2d(kernel_size=2),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(4608,2),
    nn.Dropout(p=0.5)
)

CNN10 = nn.Sequential(
    nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3),nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),nn.BatchNorm2d(32),
    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),nn.BatchNorm2d(64),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(128),
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(256),
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(256),
    nn.Flatten(),
    nn.Linear(1024,2),
    nn.Dropout(p=0.5)
)

#


batch_size = 128
train_dataset = EEGdata(root_dir="E:\EEG/alpha_img/train", labelfile="E:\EEG/alpha_img/train.txt")
val_dataset = EEGdata(root_dir="E:\EEG/alpha_img/val", labelfile="E:\EEG/alpha_img/val.txt")
test_dataset = EEGdata(root_dir="E:\EEG/alpha_img/test", labelfile="E:\EEG/alpha_img/test.txt")
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
val_iter = DataLoader(val_dataset,batch_size=batch_size, shuffle=True,num_workers=4)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
lr, num_epochs = 1e-3, 10

if __name__ == '__main__':
    model = CNN5_2
    device = 'cuda:0'

    # new_dict = OrderedDict()
    # x = torch.load("D:\FS/checkpoints/CNN5_2_3_2.pth")
    # for key in x:
    #     if key == '8.weight':
    #         new_dict['9.weight'] = x[key]
    #     elif key == '8.bias':
    #         new_dict['9.bias'] = x[key]
    #     elif key == '12.weight':
    #         new_dict['13.weight'] = x[key]
    #     elif key == '12.bias':
    #         new_dict['13.bias'] = x[key]
    #     else:
    #         new_dict[key] = x[key]
    # model.load_state_dict(new_dict)

    model.load_state_dict(torch.load("D:\FS/checkpoints/alpha/CNN5_2_3_3.pth"))
    train(model, train_iter=train_iter,val_iter=val_iter, test_iter= test_iter,num_epochs= num_epochs, lr=lr,device=device,use_pretrained_model=True,visualize=True)



