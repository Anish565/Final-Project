import numpy as np
import tensorflow as tf
from PIL import Image
from utils import *
import torch.autograd as autograd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, Normalize, Resize
import argparse
import importlib

# python -c "from main import *; ZSL_SAE_GAN_RUN('C:/Users/Anisn/Documents/Final Project/data/videos/pedestrians.avi',['bicycle','cars','motorcycle','truck','train','bus'],'C:/Users/Anisn/Documents/Final Project/saved_data/saved_models/ZSL-SAE-GAN.pt')"
