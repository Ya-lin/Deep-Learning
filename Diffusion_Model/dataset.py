

from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


path = Path('/spd', 'data')
path.mkdir(parents=True, exist_ok=True)

def download_oxford102(val=True, test=True):
    tf = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_dataset = datasets.Flowers102(root=path, split="train",
                                        transform=tf, download=True)
    val_dataset = None
    test_dataset = None
    if val:
        val_dataset = datasets.Flowers102(root=path, split="val",
                                          transform=tf, download=True)
    if test:
        test_dataset = datasets.Flowers102(root=path, split="test",
                                           transform=tf, download=True)
    repeated_train_dataset = torch.utils.data.ConcatDataset([train_dataset]*5)
    return repeated_train_dataset, val_dataset, test_dataset


class Channel2Last(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label


def get_oxford102(batch_size, val=True, test=True): 
    train_dataset, val_dataset, test_dataset = download_oxford102(val, test)
    train_dataset = Channel2Last(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = None
    test_loader = None
    if val: 
        val_dataset = Channel2Last(val_dataset)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, drop_last=False)
    if test:
        test_dataset = Channel2Last(test_dataset)
        test_loader = DataLoader(test, batch_size= batch_size,
                             shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


def separate(loader):
    # separate feature and label from data loader
    X, Y = [], []
    for x, y in loader:
        X.append(x)
        Y.append(y)
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)
    return X, Y

    
   