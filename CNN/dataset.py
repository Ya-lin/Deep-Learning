
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

path = Path.home().joinpath("Documents","Data")
path.mkdir(exist_ok=True)

def get_mnist(batch_size):
    tf = transforms.ToTensor()
    train = datasets.MNIST(root=path, train=True, 
                           download=True, transform=tf)
    test = datasets.MNIST(root=path, train=False, 
                          download=True, transform=tf)
    train_loader = DataLoader(train, batch_size= batch_size,
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(test, batch_size= batch_size,
                             shuffle=False, drop_last=False)
    return train_loader, test_loader


def download_cifar10():
    transform = transforms.ToTensor()
    cifar10_train = datasets.CIFAR10(root=path, train=True, download=True, 
                                     transform=transform)
    cifar10_test = datasets.CIFAR10(root=path, train=False, download=True, 
                                    transform=transform)
    return cifar10_train, cifar10_test


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = image.permute(1, 2, 0)
        return image, label


def get_cifar10(batch_size): 
    train, test = download_cifar10()
    train = CustomDataset(train)
    test = CustomDataset(test)
    train_loader = DataLoader(train, batch_size= batch_size,
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(test, batch_size= batch_size,
                             shuffle=False, drop_last=False)
    return train_loader, test_loader
    

def separate(loader):
    # separate feature and label from data loader
    X, Y = [], []
    for x, y in loader:
        X.append(x)
        Y.append(y)
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)
    return X, Y

    
