import torch
import numpy as np
from torchvision import transforms

toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
img = torch.load("E:\EEG/origin/train/HC/ori_train_HC0289.pth").astype(np.float32)
print(img.dtype)
pic = toPIL(img)
pic.show()

