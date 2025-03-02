# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 09:56:09 2024

@author: yalin
"""

from types import SimpleNamespace
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import torch
from torch import nn, optim
from torch.nn import functional as F
from dataset import get_data
from training import trainer
from testing import tester

import keras
from keras import layers, models


#%%
args = SimpleNamespace(dataset="mnist")
args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.lr = 5e-4
args.batch = 32
args.epoch = 15
print(args)


#%%
loader = get_data(args.dataset, args.batch)
print(len(loader.train), len(loader.test))
x, y = next(iter(loader.train))
print(x.shape, y.shape)


#%%
class Model(nn.Module):
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.model = keras.Sequential([layers.Input((1, 28, 28)),
                                       layers.Flatten(),
                                       layers.Dense(200, activation="relu"),
                                       layers.Dense(150, activation="relu"),
                                       layers.Dense(num_classes)])
    
    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        x = self.forward(x)
        pred_y = F.softmax(x, dim=1)
        pred_class = torch.argmax(pred_y, dim=1)
        return pred_class.cpu()

model = Model()
print(model.model.summary())
pred_y = model(x)
print(pred_y.shape)


#%%
model = Model().to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()
model, train_loss = trainer(model, loader.train, args.epoch, 
                            optimizer, loss_fn, args.device)
print(model.training)


#%%
plt.plot(train_loss)
plt.show()

labels, preds = tester(model, loader.test, args.device)
acc = accuracy_score(labels, preds)
print("Test accuracy: ", round(100*acc, 2))


#%%
args.dataset = "cifar10"
args.epoch = 25
print(args)


#%%
loader = get_data(args.dataset, args.batch)
print(len(loader.train), len(loader.test))
x, y = next(iter(loader.train))
print(x.shape, y.shape)


#%%
class Model(nn.Module):
    
    def __init__(self, num_classes=10):
        super().__init__()
        input_layer = layers.Input((3, 32, 32))
        x = layers.Flatten()(input_layer)
        x = layers.Dense(200, activation="relu")(x)
        x = layers.Dense(150, activation="relu")(x)
        output_layer = layers.Dense(num_classes)(x)
        self.model =  models.Model(input_layer, output_layer)
    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        x = self.forward(x)
        pred_y = F.softmax(x, dim=1)
        pred_class = torch.argmax(pred_y, dim=1)
        return pred_class.cpu()

model = Model()
print(model.model.summary())
pred_y = model(x)
print(pred_y.shape)


#%%
model = Model().to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()
model, train_loss = trainer(model, loader.train, args.epoch, 
                            optimizer, loss_fn, args.device)
print(model.training)


#%%
plt.plot(train_loss)
plt.show()

labels, preds = tester(model, loader.test, args.device)
acc = accuracy_score(labels, preds)
print("Test accuracy: ", round(100*acc, 2))



