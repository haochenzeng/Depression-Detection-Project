from calculatePLV import *
import torch
import string

# input-vector: torch.tensor [[X1],[X2],[X3],······,[Xn]]   n represents the number of channels

def divide(input_vector, division_size, dim):
    output_vector = torch.split(input_vector, division_size, dim=dim)
    return output_vector


def preprocessing(input_vector,index):
    # divided_vector = divide(input_vector, num_windows, dim=2)
    num_channels = len(input_vector)
    connectivity = torch.zeros([num_channels,num_channels])
    for m in range(num_channels):
        for n in range(num_channels):
            PLVobject = CalcPLV()
            connection = PLVobject.calplv(PLVobject.phasediff(input_vector[m], input_vector[n]))
            connectivity[m-1,n-1] = connection

    path = "E:\EEG\HC\HC"
    name = str(index) + ".pth"
    torch.save(connectivity, path + name)

def reshape_vector(input):
    x = torch.tensor(np.array(np.split(input,128,axis=0)))
    x = torch.squeeze(x)
    return divide(x, 250, dim=1)


if __name__ == '__main__':
    pass