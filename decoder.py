import torch

def decoder(input):
    r = torch.zeros([input.shape[0]],dtype=float)
    for i in range(input.shape[0]):
        index = input[i].argmax(axis=0)
        r[i] = input[i][index]
    return r