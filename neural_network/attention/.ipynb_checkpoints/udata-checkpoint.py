

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_mnist(pdata, train=True):
    if train:
        mnist = datasets.MNIST(pdata, train=True, transform=transforms.ToTensor(), download=True)
    else:
        mnist = datasets.MNIST(pdata, train=False, transform=transforms.ToTensor(), download=True)
    return mnist


class LargestDigit(Dataset):
    """
    Creates a modified version of a dataset where some number of samples 
    are taken, and the true label is the largest label sampled.
    """

    def __init__(self, dataset, toSample=3):
        """
        dataset: the dataset to sample from
        toSample: the number of items from the dataset to sample
        """
        self.dataset = dataset
        self.toSample = toSample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        #Randomly select self.toSample items from self.dataset
        selected = np.random.randint(0, len(self.dataset), size=self.toSample)
        
        #Stack the n items of shape (B, *) shape into (B, self.toSample, *)
        x_new = torch.stack([self.dataset[i][0] for i in selected])
        
        #Label is the maximum label
        y_new = max([self.dataset[i][1] for i in selected])
    
        return x_new, y_new
