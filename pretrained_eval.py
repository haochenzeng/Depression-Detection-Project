import torch
import torchvision.models as models
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
from Tools import *
from eval import *
from train import *
from fs_coding import *

# create a resnet18 network
model = models.resnet18(pretrained=False)
# load the pretrained resnet18 model parameters
model.load_state_dict(torch.load('checkpoints/'))
# convert to fs_neurons model
convert_relu(model)
# prepare the test dataset
batch_size = 256
test_dataset = d2l.load_data_fashion_mnist(batch_size=batch_size)[1]
# eval the network
test_acc = evaluate(model, test_dataset)

if __name__ == '__main__':
    print(test_acc)



