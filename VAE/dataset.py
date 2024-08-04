
import os
import numpy as np
from pathlib import Path
from keras import datasets
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_fmnist(path, batch_size):
    os.environ['KERAS_HOME'] = str(path)
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # pad images to be size of 32x32
    x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), constant_values=0.0)
    x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), constant_values=0.0)
    
    # Make sure images have shape (32, 32, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # make them writable
    x_train = np.copy(x_train)
    x_test = np.copy(x_test)
    y_train = np.copy(y_train)
    y_test = np.copy(y_test)
    
    train_data = TensorDataset(torch.from_numpy(x_train), 
                               torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(x_test), 
                              torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, 
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, 
                             shuffle=False, drop_last=False)
    return train_loader, test_loader
    
    
    
    