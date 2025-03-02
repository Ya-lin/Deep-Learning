# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 10:53:24 2024

@author: yalin
"""

from types import SimpleNamespace
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from pdb import set_trace

import torch
from torch import nn, optim
from torch.nn import functional as F
from dataset import get_data
from training import trainer
from testing import tester

import keras
from keras import layers, models


#%%
args = SimpleNamespace(dataset="cifar10")
args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.num_class = 10
args.batch = 32
args.lr = 5e-4
args.epoch = 100
print(args)


#%%
loader = get_data(args.dataset, args.batch)
print(len(loader.train), len(loader.test))
x, y = next(iter(loader.train))
print(x.shape, y.shape)


#%%
input_layer = layers.Input((3, 32, 32))
x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same")(input_layer)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

x = layers.Flatten()(x)

x = layers.Dense(128)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.Dropout(rate=0.5)(x)

output_layer = layers.Dense(args.num_class)(x)

keras_model = models.Model(input_layer, output_layer).to(args.device)
print(keras_model.summary())

x, y = next(iter(loader.train))
print(x.shape, y.shape)
pred_y = keras_model(x.to(args.device))
print(pred_y.shape)


#%%
optimizer = optim.Adam(keras_model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()
keras_model, train_loss = trainer(keras_model, loader.train, args.epoch, 
                                  optimizer, loss_fn, args.device)
print(keras_model.training)


#%%
plt.plot(train_loss)
plt.show()

labels, preds = tester(keras_model, loader.test, args.device, predict=False)
acc = accuracy_score(labels, preds)
print("Test accuracy: ", round(100*acc, 2))


#%%
class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = keras.Sequential([
            layers.Input((3, 32, 32)),
            layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same"),
            layers.BatchNormalization(),layers.LeakyReLU(),
            layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same"),
            layers.BatchNormalization(),layers.LeakyReLU(),
            layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same"),
            layers.BatchNormalization(),layers.LeakyReLU(),
            layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same"),
            layers.BatchNormalization(),layers.LeakyReLU(),
            layers.Flatten(),layers.Dense(128),layers.BatchNormalization(),
            layers.LeakyReLU(),layers.Dropout(rate=0.5),
            layers.Dense(num_classes)])

    def forward(self, x):
        return self.model(x)
    
    @torch.no_grad()
    def predict(self, x):
        x = self.forward(x)
        pred_y = F.softmax(x, dim=1)
        pred_class = torch.argmax(pred_y, dim=1)
        return pred_class.cpu()
    
torch_model = MyModel()
x, y = next(iter(loader.train))
print(x.shape, y.shape)
pred_y = torch_model(x)
print(pred_y.shape)


#%%
torch_model = MyModel().to(args.device)
optimizer = optim.Adam(torch_model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()
torch_model, train_loss = trainer(torch_model, loader.train, args.epoch, 
                                  optimizer, loss_fn, args.device)
print(torch_model.training)


#%%
plt.plot(train_loss)
plt.show()

labels, preds = tester(torch_model, loader.test, args.device)
acc = accuracy_score(labels, preds)
print("Test accuracy: ", round(100*acc, 2))



