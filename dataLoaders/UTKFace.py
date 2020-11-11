import os
import sys
import tarfile

import torch
from PIL import Image
from parse import parse
from torch.utils.data import Dataset

from config import config
from dataLoaders.datasetHandler import DatasetHandler


class UTKFaceHandler(DatasetHandler):
    def __init__(self):
        self.feature = None
        self.directory = os.path.join(config.absDatasetDir, 'UTKFace')
        self.zipFile = os.path.join(config.absDatasetDir, 'UTKFace.tar.gz')
        self.dataset = None
        self.trainDataset = None
        self.testDataset = None
        self.validateDataset = None

    def createDataset(self, feature, transform, **kwargs):
        self.feature = feature
        self.__prepareOnDisk()
        self.dataset = UTKFaceDataset(directory=self.directory, feature=self.feature, transform=transform, **kwargs)
        return self.dataset

    def __prepareOnDisk(self):
        if os.path.exists(self.directory):
            if len(os.listdir(self.directory)) != 0:
                print('UTK Already Exists on',
                      self.directory, '. We will use it!')
                return
        print('Could not find UTK on', self.directory)
        print('Looking for ', self.zipFile)
        if os.path.exists(self.zipFile):
            print(self.zipFile, 'is found. Trying to extract:')
            try:
                tar = tarfile.open(self.zipFile, "r:gz")
                tar.extractall(path=config.absDatasetDir)
                tar.close()
                print('Successfully extracted')
            except tarfile.TarError:
                print('Extract Failed!')
        else:
            sys.exit('UTK Zip file not found!')
        pass


class UTKFaceDataset(Dataset):
    def __init__(self, directory, feature, transform, **kwargs):
        self.feature = feature
        self.directory = directory
        self.transform = transform
        self.labels = []
        self.imagesPath = []
        self.images = []
        self.preload = kwargs.get('preload', 0)
        if self.preload:
            for file in os.listdir(directory):
                label = parse('{age}_{gender}_{}_{}.jpg.chip.jpg', file)
                if label is not None:
                    self.labels.append(int(label[self.feature]))
                    image = Image.open(os.path.join(self.directory, file))
                    if self.transform is not None:
                        image = self.transform(image)
                    self.images.append(image.to(config.device))
        else:
            for file in os.listdir(directory):
                label = parse('{age}_{gender}_{}_{}.jpg.chip.jpg', file)
                if label is not None:
                    self.imagesPath.append(file)
                    self.labels.append(int(label[self.feature]))
        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.preload:
            return self.images[idx].to(config.device), self.labels[idx]
        else:
            imagePath = os.path.join(self.directory,
                                     self.imagesPath[idx])
            image = Image.open(imagePath)
            if self.transform is not None:
                image = self.transform(image)
            return image.to(config.device), self.labels[idx]
