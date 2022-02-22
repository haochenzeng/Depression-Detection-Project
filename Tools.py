import torch
from torch import nn
import time
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

class Timer:

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tick = time.time()

    def stop(self):
        self.times.append(time.time()-self.tick)
        return self.times[-1]

    def avg(self):
        return sum(self.times)/len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


class Accumulator:

    def __init__(self,n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a,b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]