

import numpy as np
from pdb import set_trace

import keras
from keras import layers, models
from torch import nn

def get_encoder(image_size, channels, embedding_dim):
    encoder_input = layers.Input(shape=(image_size, image_size, channels))
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

class AE(nn.Module):
    
    def __init__(self, image_size, channels, embedding_dim):
        super().__init__()
        self.encoder, shape_bf = get_encoder(image_size, channels, embedding_dim)
        self.decoder = get_decoder(embedding_dim, channels, shape_bf)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


