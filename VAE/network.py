

import numpy as np
import keras
from keras import layers, models
import torch
from torch import nn
from torch.distributions import MultivariateNormal


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
    
    def __init__(self, image_size, channels, embedding_dim, gamma, device):
        super().__init__()
        self.encoder, self.shape_bf = get_encoder(image_size, channels, embedding_dim)
        self.decoder = get_decoder(embedding_dim, channels, self.shape_bf)
        self.for_mean = layers.Dense(embedding_dim)
        self.for_logvar = layers.Dense(embedding_dim)
        self.bce_loss = nn.BCELoss()
        self.emb_dim = embedding_dim
        self.gamma = gamma
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        en_z = self.encoder(x)
        mu = self.for_mean(en_z)
        log_var = self.for_logvar(en_z)
        epsilon = torch.randn_like(log_var).to(self.device)
        sigma = torch.exp(0.5*log_var)
        z = mu + sigma*epsilon
        x_hat = self.decoder(z)
        return x_hat, mu, log_var, z 
        
    def total_loss(self, x):
        x = x.to(self.device)
        x_hat, mu, log_var, _ = self(x)
        res_loss = self.bce_loss(x_hat, x).mean()
        kl_loss = -0.5*torch.mean(torch.sum(1+log_var-mu**2-torch.exp(log_var), dim=1))
        total_loss = self.gamma*res_loss + kl_loss
        return total_loss

    @torch.no_grad()
    def sampling(self, num_sample):
        # generate new images
        z = torch.randn(num_sample, self.emb_dim)
        x = self.decoder(z.to(self.device))
        return x.cpu()
        
        