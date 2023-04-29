import torch.autograd as autograd
import torch
import torch.nn as nn
from testing import *
from matplotlib import pyplot as plt
import numpy as np
import torch
import json
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_conditional_input(X, C_Y):
        new_X = torch.cat([X, C_Y], dim=1).float()
        return autograd.Variable(new_X).to(device)

def convert():
    total = []
    with open('saved_data/count_info.txt',"r") as f:
        details = f.readlines()
        # print(details[0])
        for i in details:
            element_dict={}
            elements = i.split('\t')
            # print(elements[0])
            element_dict["TIME"] = elements[0]
            # print(elements[1])
            dets = elements[1].split(',')
            # print(dets)
            for j in dets[:-1]:
                count = j.split(' ')
                element_dict[count[1]] = count[0]
            # print(element_dict)
            total.append(element_dict)

    with open('saved_data/count_info.json', "w") as f:
        json.dump(total,f)


def ZSL_SAE_GAN_RUN(path,classes, model):
    with open("saved_data/count_info.txt", "w") as f:
         f.write("")
    with open("saved_data/count_info.json", "w") as f:
         f.write("")
    # model = torch.load("C:/Users/Anisn/Documents/Final Project/saved_data/yolov8s.pt")
    model = GAN(model)
    results=model.predict(source=path,save=True,show=True,classes=classes)
    # print(results[0].probs)
    convert()
    # model.save("ZSL_SAE_GAN_PRED.pth")
    # print(results)



def save_images(generator, epoch, latent_dim):
    n = 5
    figure = np.zeros((256 * n, 256 * n, 3))
    for i in range(n):
        for j in range(n):
            random_latent_vector = np.random.normal(size=(1, latent_dim))
            generated_image = generator.predict(random_latent_vector)
            figure[i * 256: (i + 1) * 256,
                   j * 256: (j + 1) * 256] = generated_image[0]
    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.axis('off')
    plt.savefig(f"generated_images_{epoch}.png")


def fit_classifier(img_features, label_attr, label_idx, classifier, optim_cls, criterion_cls):
        # Train the classifier in supervised manner on a single minibatch of available data
        img_features = autograd.Variable(img_features).to(device)
        label_attr = autograd.Variable(label_attr).to(device)
        label_idx = autograd.Variable(label_idx).to(device)

        X_inp = get_conditional_input(img_features, label_attr)
        Y_pred = classifier(X_inp)

        optim_cls.zero_grad()
        loss = criterion_cls(Y_pred, label_idx)
        loss.backward()
        optim_cls.step()

        return loss.item()

def fit_final_classifier(img_features, label_attr, label_idx, final, optim_cls, criterion_cls):
        img_features = autograd.Variable(img_features.float()).to(device)
        label_attr = autograd.Variable(label_attr.float()).to(device)
        label_idx = label_idx.to(device)

        X_inp = get_conditional_input(img_features, label_attr)
        Y_pred = final(X_inp)

        optim_cls.zero_grad()
        loss = criterion_cls(Y_pred, label_idx)
        loss.backward()
        optim_cls.step()

        return loss.item()