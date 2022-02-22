import d2l.torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
import torchvision.models as models
import cv2
import skimage.io as io

class EEGdata(Dataset):

    def __init__(self, root_dir, labelfile):
        self.root_dir = root_dir
        self.labelfile = labelfile
        self.size = 0
        self.name_list = []

        if not os.path.isfile(self.labelfile):
            print(self.labelfile + 'does not exist')
        file = open(self.labelfile)
        for f in file:
            self.name_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        path = str(self.name_list[item].split(' ')[0])
        if not os.path.isfile(path):
            print(path + 'does not exist')
            return None
        # image = torch.load(path).reshape([1,28,28]).astype(np.float32)
        image = io.imread(path).reshape(1,128,128).astype(np.float32)
        # label = int(self.name_list[item].split(' ')[1])
        label = int(self.name_list[item].split(' ')[1])

        # sample = {'image':image, 'label':label}

        return image,label


if __name__ == '__main__':
    train_dataset = EEGdata(root_dir="E:\EEG/alpha_img/train", labelfile="E:\EEG/alpha_img/train.txt")
    val_dataset = EEGdata(root_dir="E:\EEG/alpha_img/val", labelfile="E:\EEG/alpha_img/val.txt")

    # print(train_dataset.name_list)


    train_iter = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=4)
    val_iter = DataLoader(val_dataset, batch_size=32, shuffle=True,num_workers=4)



    for batchidx, (x, label) in enumerate(train_iter):
        # print(x.shape)
        print(x.dtype)




    # model = models.resnet18(pretrained=True)
    # model.fc = torch.nn.Linear(512,2)
    #
    # new_train(model,train_iter,val_iter,10,0.01,d2l.torch.try_gpu())

    # for i, data in enumerate(val_iter):
    #     X,y = data['image'], data['label']
    #     print(X.shape,y.shape)
    # for x,y in enumerate(val_iter):
    #     print(x)

