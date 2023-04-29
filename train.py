import numpy as np
# import tensorflow as tf
from PIL import Image
import torch.autograd as autograd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, Normalize, Resize
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, Dropout, LeakyReLU, Activation
from keras.models import Model, Sequential
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import torch
import json
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from GAN_model import *
from SAE_model import *
from utils import *
from captions import CaptionImageDataset

batch_size = 32
learning_rate = 0.0002
num_epochs = 1000
latent_dim = 100
image_size = 64
 # Define the device (use GPU if available, otherwise use CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_sae_gan(latent_dim, num_epochs=100, batch_size=32):

    # Build the generator and discriminator models
    generator = Generator(latent_dim)
    discriminator = Discriminator(input_shape=(256, 256, 3))
    
    # Define a function to generate a batch of random noise samples for the GAN
    def generate_noise(batch_size, latent_dim):
        return np.random.normal(0, 1, (batch_size, latent_dim))

    def save_image_grid(images, filename):
        # Rescale images to 0-255 range
        images = (images * 255).astype('uint8')
        
        # Create a grid of images
        grid_size = int(np.ceil(np.sqrt(images.shape[0])))
        grid = np.zeros((grid_size*images.shape[1], grid_size*images.shape[2], images.shape[3]), dtype='uint8')
        for i in range(images.shape[0]):
            row = i // grid_size
            col = i % grid_size
            grid[row*images.shape[1]:(row+1)*images.shape[1], col*images.shape[2]:(col+1)*images.shape[2], :] = images[i]
    
        # Save the grid of images
        Image.fromarray(grid).save(filename)
    
    # Build the combined model, which stacks the generator on top of the discriminator
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    criterion_autoencoder = nn.MSELoss()
    criterion_discriminator = nn.BCELoss()
    criterion_generator = nn.BCELoss()
    optimizer_autoencoder = optim.Adam(semantic_autoencoder.parameters(), lr=learning_rate)
    optimizer_discriminator = optim.Adam(gan.discriminator.parameters(), lr=learning_rate)
    optimizer_generator = optim.Adam(gan.generator.parameters(), lr=learning_rate)
    optimizer_encoder_decoder = optim.Adam(gan.encoder_decoder.parameters(), lr=learning_rate)


    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
   

    dataset = CaptionImageDataset(image_paths='data/COCO2017/images', seen_caption_paths='data/COCO2017/captions',unseen_caption_path="data/COCO2017/new_captions", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    num_epochs = 2000
    for epoch in range(num_epochs):
        for i, (images, captions, unseen, idx) in enumerate(dataloader):
            # Train Discriminator
            optimizer_discriminator.zero_grad()
           
            # Generate fake images
            z = gan.encoder(images)
            fake_images = generator(z.detach())

            # Compute the discriminator loss on the fake images
            # d_fake = discriminator(fake_images, captions)
            d_fake = fit_classifier(fake_images,captions,idx,gan.discriminator,optimizer_discriminator,criterion_discriminator)
            d_fake_loss = criterion_discriminator(d_fake, torch.zeros((fake_images.shape[0], 1)).to(device)) # loss calculated by the discriminitator

            # Compute the discriminator loss on the real images
            d_real = fit_classifier(images,captions,idx,gan.discriminator,optimizer_discriminator,criterion_discriminator)
            d_real_loss = criterion_discriminator(d_real, torch.ones((images.shape[0], 1)).to(device))
        

            # Update discriminator weights
            discriminator_loss = d_fake_loss + d_real_loss
            discriminator_loss.backward()
            optimizer_discriminator.step()

            # Train Encoder and Decoder
            optimizer_encoder_decoder.zero_grad()

            # reconstructing the input image
            fake_images = gan.decoder(z)

            # Compute the autoencoder loss
            reconstruction_loss = criterion_autoencoder(fake_images, images) # used to find the reconstruction loss from real images to fake images

            # Compute the generator loss
            generator_real = fit_classifier(images,unseen,idx,gan.generator,optimizer_generator,criterion_generator)
            generator_loss = criterion_generator(gan.discriminator(z, unseen), torch.ones((images.shape[0], 1)).to(device))
            generator_loss += generator_real
            generator_loss.backward()
            optimizer_generator.step()
            # This loss is calculated based on the discriminator's output when it is fed with the generated data

            # Update encoder and decoder weights
            total_loss = reconstruction_loss + generator_loss
            total_loss.backward()
            optimizer_encoder_decoder.step()

            # Fine-tune the autoencoder
            optimizer_autoencoder.zero_grad()
            autoencoder_loss = semantic_autoencoder.train_on_batch(images, captions)
            autoencoder_loss.backward()                                                                                                                                                                                                                                                                                                                                                         # type: ignore
            optimizer_autoencoder.step()

            l = fit_final_classifier(images, captions, idx, gan, optimizer_autoencoder, criterion_autoencoder)
            # Print the loss every 100 iterations
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Discriminator Loss: {discriminator_loss.item():.4f}, Generator Loss: {generator_loss.item():.4f}, Reconstruction Loss: {reconstruction_loss.item():.4f, Autoencoder Loss: {autoencoder_loss :.4f}}, Final Loss: {l}")
            
            # Save the generated images
            if i % 100 == 0:
                save_image(fake_images, 'saved_data/generated_images/fake_samples_epoch_%03d_batch_%03d.png' % (epoch, i), normalize=True)

        # Save the model
        torch.save(gan.state_dict(), f"saved_data/saved_models/gan_epoch_{epoch+1}.pth")

    
    return semantic_autoencoder, generator, discriminator, gan



# train_sae_gan(100)