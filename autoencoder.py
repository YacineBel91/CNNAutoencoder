
import os
from random import shuffle

import torch.nn as nn
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import imghdr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def createAndTrainModel(**kwargs):
    """
    :param dataset: The dataset you want the autoencoder to be trained on.
    Dataset's __getitem__ function should return a C,H,W-shaped tensor
    :param batch_size: The size of the batches (defaults to 4)
    :param learning_rate: The learning rate (defaults to 1e-3)
    :param num_epochs: The number of epochs (defaults to 50)
    :param criterion: The Loss function (defaults to MSELoss)
    :param logFileName: (very optional) Path to a file where to store the history of the training
    :return: 
    """

    if "dataset" in kwargs:
        aeDataset = kwargs["dataset"]
    else:
        raise ValueError("No dataset provided (use dataset keyword argument)")

    if not isinstance(aeDataset,Dataset):
        raise ValueError("Dataset provided is not a valid torch dataset")

    if "batch_size" in kwargs:
        batch_size=kwargs["batch_size"]
    else:
        batch_size=4

    if "learning_rate" in kwargs:
        learning_rate=kwargs["learning_rate"]
    else:
        learning_rate=1e-3

    if "num_epochs" in kwargs:
        num_epochs=kwargs["num_epochs"]
    else:
        num_epochs=50

    aeDataloader = DataLoader(aeDataset, batch_size=batch_size,pin_memory=True)

    model=AutoEncoder()
    model.to(device)

    if "criterion" in kwargs:
        criterion=kwargs["criterion"]
    else:
        criterion=nn.MSELoss()

    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

    lossLog=[]

    for epoch in range(num_epochs):
        loss=0
        for batch in aeDataloader:
            batch=batch.to(device)
            optimizer.zero_grad()

            output = model(batch)

            train_loss=criterion(output,batch)# The goal is to denoise, so making this look like a noisy image is maybe not the best. To be continued...

            train_loss.backward()
            optimizer.step()

            loss+=train_loss.item()

        loss=loss/(len(aeDataloader))

        lossLog.append((epoch,loss))

        print(f"epoch [{epoch+1}/{num_epochs}], loss:{loss:.4f}")
        # if epoch%10==0 and "directory" in kwargs:
        #     pic=torchvision.transforms.ToPILImage()(output[0].cpu())
        #     pic.save(f"{kwargs['directory']}/SomeImprobableName__Epoch{epoch+1}.jpg")

    if "logFileName" in kwargs:
        f=open(kwargs["logFileName"],"w")
        f.write(f"Epoch,"
                f"Loss,"
                f"Criterion,"
                f"LearningRate\n")
        for epochX,lossY in lossLog:
            f.write(f"{epochX}"
                    f",{lossY},"
                    f"{criterion.__class__.__name__},"
                    f"{learning_rate}"
                    f"\n")
        f.close()

    return model,loss # Return the trained model, and the latest training loss it yielded