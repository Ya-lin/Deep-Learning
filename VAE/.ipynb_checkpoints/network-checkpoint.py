

import numpy as np
import keras
from keras import layers, models
import torch
from torch import nn


def get_encoder(image_size, channels, embedding_dim):
    encoder_input = layers.Input(shape=(channels, image_size, image_size))
    x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
    x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
    shape_bf = x.shape[1:] 
    x = layers.Flatten()(x)
    encoder_output = layers.Dense(embedding_dim, name="encoder_output")(x)
    encoder = models.Model(encoder_input, encoder_output)
    return encoder, shape_bf

def get_decoder(embedding_dim, channels, shape_bf):
    decoder_input = layers.Input(shape=(embedding_dim, ), name="decoder_input")
    x = layers.Dense(np.prod(shape_bf))(decoder_input)
    x = layers.Reshape(shape_bf)(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    decoder_output = layers.Conv2D(channels, (3, 3), strides=1, activation="sigmoid",
                                   padding="same", name="decoder_output")(x)
    decoder = models.Model(decoder_input, decoder_output)
    return decoder

class VAE(nn.Module):
    
    def __init__(self, image_size, channels, embedding_dim, gamma):
        super().__init__()
        self.encoder, self.shape_bf = get_encoder(image_size, channels, embedding_dim)
        self.decoder = get_decoder(embedding_dim, channels, self.shape_bf)
        self.for_mean = layers.Dense(embedding_dim)
        self.for_logvar = layers.Dense(embedding_dim)
        self.gamma = gamma
        self.bce_loss = nn.BCELoss()

    def sampling(self, x):
        en_z = self.encoder(x)
        mu = self.for_mean(en_z)
        log_var = self.for_logvar(en_z)
        epsilon = torch.randn_like(log_var).to(log_var.device)
        sigma = torch.exp(0.5*log_var)
        z = mu + sigma*epsilon
        return z, mu, log_var
        
    def forward(self, x):
        z, mu, log_var = self.sampling(x)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var
        
    def total_loss(self, x):
        x_hat, mu, log_var = self(x)
        res_loss = self.bce_loss(x_hat, x).mean()
        kl_loss = -0.5*torch.mean(torch.sum(1+log_var-mu**2-torch.exp(log_var), dim=1))
        total_loss = self.gamma*res_loss + kl_loss
        return total_loss
    
    


