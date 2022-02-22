import numpy as np
import torch
from fs_coding import *
from fs_weights import *
from CNN5 import *
from data import *

net = CNN5_2
# convert_relu(net)

test_dataset = EEGdata(root_dir="E:\EEG/alpha_img/test", labelfile="E:\EEG/alpha_img/test.txt")
test_iter = DataLoader(test_dataset, batch_size=32, shuffle=True,num_workers=4)

val_dataset = EEGdata(root_dir="E:\EEG/alpha_img/val", labelfile="E:\EEG/alpha_img/val.txt")
val_iter = DataLoader(val_dataset, batch_size=32, shuffle=True,num_workers=4)

if __name__ == '__main__':
    net.load_state_dict(torch.load("D:\FS/checkpoints/alpha/CNN5_2_3_4.pth"))
    net.to(device='cuda:0')
    test_acc = evaluate(net, test_iter, 'cuda:0')
    val_acc = evaluate(net, val_iter, 'cuda:0')
    print(f'converting SNN val_acc={val_acc}')
    print(f'converting SNN test_acc={test_acc}')
    print(np.sum(spike))
