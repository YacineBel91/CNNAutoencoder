
import os
from random import shuffle

import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import Dataset

import imghdr

def listImagesInDir(directory, ignoreDirs=[]):
    res = []
    ignoreDirs = [d.split("/")[-1] for d in ignoreDirs]
    for root, dirs, files in os.walk(directory, topdown=True):
        dirs[:] = [d for d in set(dirs) - set(ignoreDirs)]
        for fileName in files:
            if imghdr.what(os.path.join(root, fileName)) in ['rgb', 'gif', 'pbm', 'pgm', 'ppm', 'tiff', 'xbm', 'jpeg', 'bmp', 'png', 'webp']:
                res.append(os.path.join(root, fileName))
    return res

class AutoEncoderLocalFileDataset(Dataset):
    def __init__(self, **kwargs):
        if "dirPath" in kwargs:
            if "ignoreDirs" in kwargs:
                self.imagePaths = listImagesInDir(kwargs["dirPath"], ignoreDirs=kwargs["ignoreDirs"])
            else:
                self.imagePaths = listImagesInDir(kwargs["dirPath"])

        if "csvFilesList" in kwargs and os.path.isfile(kwargs["csvFilesList"]):
            self.imagePaths = [line.strip() for line in open(kwargs["csvFilesList"], "r")]  # Just load every line from the list of files

        if "fileNames" in kwargs:
            self.imagePaths = kwargs["fileNames"]

        self.indices = list(range(len(self.imagePaths)))
        shuffle(self.indices)

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, index):
        path = self.imagePaths[self.indices[index]]  # We shuffled the indices ourselves, so we need to use our indices
        img = Image.open(path)
        if img.size[0] / img.size[1] < 1:
            img = img.transpose(Image.ROTATE_90)
        img = img.resize((292, 196))
        return torchvision.transforms.ToTensor()(img)


class AutoEncoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 21, 3, stride=3, padding=1, groups=3),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(21, 3, 3, stride=2, padding=1, groups=3),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 21, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(21, 7, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(7, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, stride=3, padding=1),  # Stride was 3
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),  # Stride was 2
        #     nn.Conv2d(16, 8, 3, stride=2, padding=1),  # Stride was 2
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=1)  # Stride was 1
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(8, 16, 3, stride=2),  # Stride was 2
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
        #     nn.Tanh()
        # )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
