import torch
import numpy as np
import cv2
import skimage.io as io
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict

x = torch.rand(2,3)
y = 0.5
print(x)
print(x-y)
z = x-y

m = torch.where(z>0,torch.ones_like(z),torch.zeros_like(z))
print(m)
print(f'Spikes:{torch.sum(m)}')