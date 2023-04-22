from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, Dropout, LeakyReLU, Activation, ZeroPadding2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np

# Set the image dimensions
img_rows = 256
img_cols = 256
channels = 3
img_shape = (img_rows, img_cols, channels)

# Set the size of the learned feature space and dimension of the noise data
latent_dim = 100

# encoder layer of the semantic autoencoder
encoder_inputs = Input(shape=img_shape)
x = Conv2D(32, kernel_size=3, strides=2, padding='same')(encoder_inputs)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.25)(x)
x = Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
x = ZeroPadding2D(padding=((0,1),(0,1)))(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.25)(x)
x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.25)(x)
x = Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
encoded = Dense(latent_dim)(x)

# decoder layer of the semantic autoencoder
decoder_inputs = Input(shape=(latent_dim,))
x = Dense(8 * 8 * 256)(decoder_inputs)
x = Reshape((8, 8, 256))(x)
x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.25)(x)
x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.25)(x)
x = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.25)(x)
x = Conv2DTranspose(16, kernel_size=4, strides=2, padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.25)(x)
x = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same')(x)
decoded = Activation('sigmoid')(x)



# Define the encoder model
encoder = Model(encoder_inputs, encoded, name='encoder')

# Define the decoder model
decoder = Model(decoder_inputs, decoded, name='decoder')

# Define the semantic autoencoder model
autoencoder_outputs = decoder(encoded)
semantic_autoencoder = Model(encoder_inputs, autoencoder_outputs, name='autoencoder')

# Compile the semantic autoencoder model
semantic_autoencoder.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

