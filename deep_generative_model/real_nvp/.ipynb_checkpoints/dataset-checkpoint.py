

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from types import SimpleNamespace
from pathlib import Path

path = Path.home().joinpath("Documents", "Data")
path.mkdir(exist_ok=True)

def get_mnist(batch_size):  
    tf = transforms.ToTensor() 
    train = datasets.MNIST(root=path, train=True, download=True, transform=tf)
    test = datasets.MNIST(root=path, train=False, download=True, transform=tf)
    loader = SimpleNamespace()
    loader.train = DataLoader(train, batch_size= batch_size, shuffle=True, drop_last=True)
    loader.test = DataLoader(test, batch_size= batch_size, shuffle=False, drop_last=False)
    return loader


