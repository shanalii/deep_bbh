from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np 
from torch.utils.data import Dataset
from torch import nn
import torch
import time


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ShadedNoiseDL(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, noise_sigma=(.2, .6), num_samps=100):
        self.data_dir = data_dir
        self.dataset = ShadedNoiseDS(num_samps=num_samps, noise_sigma=noise_sigma)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ShadedNoiseDS(Dataset):

    def __init__(self, num_samps=100, noise_sigma=(0.2,0.6)):
        