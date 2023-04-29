# import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, Dropout, LeakyReLU, Activation
from keras.models import Model, Sequential
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from SAE_model import *


# Define the input shape
input_shape = (256, 256, 3)
latent_dim = 100



def Generator(latent_dim):
    model = Sequential()
    model.add(Dense(256 * 16 * 16, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((16, 16, 256)))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding='same'))
    model.add(Activation('tanh'))
    return model

    
def Discriminator(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='softmax'))
    return model


class GAN():
   
    def __init__(self):
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.encoder = encoder()
        self.decoder = decoder()
        self.autoencoder = semantic_autoencoder()
        self.generator = Generator(latent_dim)
        print("\n\nGenerator")
        print(self.generator.summary())
        self.discriminator = Discriminator(input_shape)
        print("\n\nDiscriminator")
        print(self.discriminator.summary())
        self.discriminator.compile(loss='categorical_crossentropy',
                                   optimizer=Adam(0.0002, 0.5),
                                   metrics=['accuracy'])
        self.generator.compile(loss='categorical_crossentropy', optimizer=Adam(0.0002, 0.5))
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        validity = self.discriminator(img)
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))



    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :]) # type: ignore
                axs[i, j].axis('off') # type: ignore
                cnt += 1
        fig.savefig(f"images/{epoch}.jpg")
        plt.close()
            