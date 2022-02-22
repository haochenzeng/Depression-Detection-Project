import torch
import numpy as np
from fs_weights import *

n_neurons = 0
n_spikes = []
n_images = 0
print_spikes = False
print_n_neurons = False

def spike_function(x : torch.Tensor):
    z_ = torch.where(x > 0,
                     torch.ones_like(x),
                     torch.zeros_like(x))
    if z_.is_cuda :
        z_ = torch.cuda.FloatTensor(z_)
    else:
        z_ = torch.FloatTensor(z_)
    return z_

def fs(x : torch.Tensor, h, d, T, K):
    global n_neurons,n_images
    # each time enters this function, it means the net processes an image
    n_images += 1
    x = x.float()
    n_neurons += np.prod(list(x.size())[1:])
    if print_n_neurons:
        print(f'Number of neurons: {n_neurons}')
    # Initialize
    t = 0
    v = x.clone()
    z = torch.zeros_like(x)
    out = torch.zeros_like(x)
    v_reg, z_reg = 0.,0.
    def while_body(out, v_reg, z_reg, v, z, t):
        v_scaled = (v - T[t]) / (torch.abs(v) + 1)
        # transfer to spikes
        z = spike_function(v_scaled)
        # regularization
        v_reg += torch.square(torch.mean(torch.maximum(torch.abs(v_scaled)-1, torch.tensor(0))))
        z_reg += torch.mean(z)
        global n_spikes
        n_spikes.append(torch.sum(z))
        if print_spikes:
            print(f'Spikes:{torch.sum(z)}')
        # add the output at clock t to total output
        out += z * d[t]
        v =v - z * h[t]
        t += 1
        return out, v_reg, z_reg, v, z, t

    while t < K :
        out_temp, v_reg_temp, z_reg_temp, v_temp, z_temp, t_temp = while_body(out, v_reg, z_reg, v, z, t)
        #update
        out, v_reg, z_reg, v, z, t = out_temp, v_reg_temp, z_reg_temp, v_temp, z_temp, t_temp

    return out

class fs_sigmoid(torch.nn.Module):
    def forward(self, input: torch.Tensor):
        return fs(input,sigmoid_h,sigmoid_d,sigmoid_T,K=len(sigmoid_h))

class fs_relu(torch.nn.Module):
    def forward(self, input: torch.tensor):
        return fs(input,relu_h,relu_d,relu_T,K=len(relu_h))

class fs_swish(torch.nn.Module):
    def forward(self, input: torch.tensor):
        return fs(input,swish_h,swish_d,swish_T,K=len(swish_h))



def convert_sigmoid(model):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.Sigmoid):
            setattr(model, child_name, fs_sigmoid())
        else:
            convert_sigmoid(child)

def convert_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, child_name, fs_relu())
        else:
            convert_relu(child)

def convert_swish(model):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.relu_fn):
            setattr(model, child_name, fs_swish())
        else:
            convert_swish(child)



