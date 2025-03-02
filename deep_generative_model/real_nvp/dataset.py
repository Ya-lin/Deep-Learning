

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from types import SimpleNamespace
from pathlib import Path

path = Path.home().joinpath("Documents", "Data")
path.mkdir(exist_ok=True)

def get_mnist(batch_size, ratio):  
    tf = transforms.ToTensor() 
    train = datasets.MNIST(root=path, train=True, download=True, transform=tf)
    test = datasets.MNIST(root=path, train=False, download=True, transform=tf)
    
    n_train = int(len(train)*ratio); n_valid = len(train)-n_train
    train, valid = random_split(train, [n_train, n_valid])
    
    loader = SimpleNamespace()
    loader.train = DataLoader(train, batch_size= batch_size, shuffle=True, drop_last=True)
    loader.valid = DataLoader(valid, batch_size= batch_size, shuffle=False, drop_last=True)
    loader.test = DataLoader(test, batch_size= batch_size, shuffle=False, drop_last=False)
    return loader



