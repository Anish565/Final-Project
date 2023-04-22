import tensorflow as tf
import keras.applications
import keras.applications.inception_v3
import numpy as np
import os
import time
import functools
import math
import sys
# import tensorflow.contrib.gan as tfgan
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from captions import CaptionImageDataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from data.images.files import *

# Load the InceptionV2 model
inception_model_v2 = keras.applications.InceptionResNetV2(                                                                                                                                                                             
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000,
    classifier_activation='softmax'
)

# Load the InceptionV3 model
inception_model_v3_torch = models.inception_v3(pretrained=True)


# Load Inception-v3 model.
inception_model_v3_keras = keras.applications.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(256, 256, 3),
    pooling='avg'
)
inception_model_v3_keras.trainable = False


# Define a function to calculate the activations of the Inception-v3 model.
def get_inception_activations(images, batch_size=64):
    num_batches = int(math.ceil(float(len(images)) / float(batch_size)))
    activations = np.zeros([len(images), 2048], dtype=np.float32)
    for i in range(num_batches):
        start = i * batch_size
        end = min(len(images), (i + 1) * batch_size)
        batch = images[start:end]
        inp = keras.applications.inception_v3.preprocess_input(batch)
        activations[start:end] = inception_model_v3_keras.predict(inp)
    return activations

# Define a function to calculate the inception score for a set of generated images.
def calculate_inception_score(generated_images, batch_size=64, num_splits=10):
    assert generated_images.shape[1] == 3
    assert np.max(generated_images[0]) > 10
    # Split generated images into num_splits parts.
    splits = np.array_split(generated_images, num_splits)
    # Compute the activations for each split.
    activations = []
    for i, split in enumerate(splits):
        activations.append(get_inception_activations(split, batch_size))
    activations = np.concatenate(activations, axis=0)
    # Compute the marginal entropy and conditional entropy.
    scores = []
    for i in range(num_splits):

        split = activations[i * (len(activations) // num_splits):(i + 1) * (len(activations) // num_splits), :]
        p_y = np.expand_dims(np.mean(split, 0), 0)
        p_y_log_prob = np.log(p_y + 1e-12)
        kl_divergence = np.sum(split * (np.log(split + 1e-12) - p_y_log_prob), 1) # type: ignore
        kl_divergence_mean = np.mean(kl_divergence)
        scores.append(np.exp(kl_divergence_mean))
        
    return np.mean(scores), np.std(scores)
# Define the device (use GPU if available, otherwise use CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visual_inspection(path):
    # Create a new instance of GAN
    gan_loaded = GAN()

    # Load the state dict of the saved model
    gan_loaded.load_state_dict(torch.load(path)) # type: ignore

    # Set the model to evaluation mode
    gan_loaded.eval()

    # Generate new images using the generator
    for i in range(10):
        z = torch.randn(1, 256).to(device)
        generated_images = gan_loaded.decoder(z)

        # Get the output of the discriminator for the generated images
        discriminator_output = gan_loaded.discriminator(gan_loaded.encoder(generated_images))
        print(torch.round(discriminator_output.sigmoid()))

        # You can then save the generated images using the save_image function as before
        save_image(generated_images.data, "saved_data/discriminator_output/generated_images{}.png".format(i), normalize=True)

def ZSL_accuracy(path,batch_size,t):
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score

    # Load the saved GAN model
    gan_model = torch.load(path)
    

    # Load the test data
    test_data = CaptionImageDataset(image_paths='data/train2017/images', seen_caption_paths='data/train2017/captions', unseen_caption_path='data/train2017/new_captions')
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Evaluate the model
    gan_model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, captions in test_dataloader:
            images = images.to(device)
            captions = captions.to(device)
            generated_images = gan_model.generator(images, captions)
            # Flatten the images and captions for evaluation
            generated_images = generated_images.view(generated_images.size(0), -1)
            captions = captions.view(captions.size(0), -1)
            # Calculate the cosine similarity between the generated images and captions
            similarity = cosine_similarity(generated_images, captions, dim=1)
            # Predict the class with the highest similarity score
            _, pred = torch.max(similarity, 0)
            y_pred.append(pred.item())
            y_true.append(test_data.label)

    # Calculate the zero-shot accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print("Zero-shot accuracy: {:.2f}%".format(accuracy * 100))