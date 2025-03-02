

from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

path = Path.home().joinpath("Documents","Data")
path.mkdir(exist_ok=True)

def get_data(dataset_name, batch_size):
    
    tf = transforms.ToTensor()
    if dataset_name == "mnist":
        train = datasets.MNIST(root=path, train=True, download=True, transform=tf)
        test = datasets.MNIST(root=path, train=False, download=True, transform=tf)
    elif dataset_name == "cifar10":
        train = datasets.CIFAR10(root=path, train=True, download=True, transform=tf)
        test = datasets.CIFAR10(root=path, train=False, download=True, transform=tf)

    loader = SimpleNamespace()
    loader.train = DataLoader(train, batch_size= batch_size, shuffle=True, drop_last=True)
    loader.test = DataLoader(test, batch_size= batch_size, shuffle=False, drop_last=False)
    
    return loader


def separate(loader):
    # separate feature and label from data loader
    X, Y = [], []
    for x, y in loader:
        X.append(x)
        Y.append(y)
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)
    return X, Y

