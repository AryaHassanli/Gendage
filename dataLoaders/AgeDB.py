import os
import sys
import zipfile

import torch
from PIL import Image
from parse import parse
from torch.utils.data import Dataset, random_split, DataLoader

from config import config
from dataLoaders.datasetHandler import DatasetHandler


class AgeDBHandler(DatasetHandler):
    def __init__(self):
        self.directory = os.path.join(config.absDatasetDir, 'AgeDB')
        self.zipFile = os.path.join(config.absDatasetDir, 'AgeDB.zip')
        self.feature = None
        self.dataset = None
        self.trainDataset = None
        self.testDataset = None
        self.validateDataset = None
        self.usePreProcessed = None
        self.forcePreProcess = None

    def createDataset(self, feature, transform, **kwargs):
        self.feature = feature
        self.__prepareOnDisk()
        self.dataset = AgeDBDataset(directory=self.directory,
                                    feature=feature,
                                    transform=transform,
                                    **kwargs)
        return self.dataset

    def __prepareOnDisk(self):
        if os.path.exists(self.directory):
            if len(os.listdir(self.directory)) != 0:
                print('AgeDB Already Exists on ' +
                      self.directory + '.We will use it!')
                return
        print('Could not find AgeDB on', self.directory)
        print('Looking for ', self.zipFile)
        if os.path.exists(self.zipFile):
            print(self.zipFile, 'is found. Trying to extract:')
            with zipfile.ZipFile(self.zipFile) as zf:
                zf.extractall(pwd=b'ibugAgeDBv2017a', path=config.absDatasetDir)
            print('Successfully extracted')
        else:
            sys.exit('AgeDB Zip file not found!')
            # TODO: In case of zip file is missing, simply download it!


class AgeDBDataset(Dataset):
    def __init__(self, directory, feature, transform, usePreProcessed, **kwargs):
        self.directory = directory if usePreProcessed == 0 else os.path.join(directory, 'preProcessed')
        self.transform = transform
        genderToClassId = {'m': 0, 'f': 1}
        self.labels = []
        self.imagePath = []
        self.images = []
        self.minAge = 1000
        self.maxAge = 0

        self.preload = kwargs.get('preload', 0)
        if self.preload:
            for i, file in enumerate(os.listdir(self.directory)):
                label = parse('{}_{}_{age}_{gender}.jpg', file)
                if label is None:
                    continue
                image = Image.open(os.path.join(self.directory, file))
                if self.transform is not None:
                    image = self.transform(image)
                self.images.append(image.to(config.device))

                if feature == 'gender':
                    self.labels.append(genderToClassId[label['gender']])
                elif feature == 'age':
                    age = int(label['age'])
                    self.labels.append(age)
        else:
            for i, file in enumerate(os.listdir(self.directory)):
                label = parse('{}_{}_{age}_{gender}.jpg', file)
                if label is None:
                    continue
                self.imagePath.append(os.path.join(self.directory, file))
                if feature == 'gender':
                    self.labels.append(genderToClassId[label['gender']])
                elif feature == 'age':
                    age = int(label['age'])
                    self.labels.append(age)
                    self.maxAge = age if age > self.maxAge else self.maxAge
                    self.minAge = age if age < self.minAge else self.minAge
        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.preload:
            image = self.images[idx]
            return image.to(config.device), self.labels[idx]
        else:
            image = Image.open(self.imagePath[idx])
            if self.transform is not None:
                image = self.transform(image)
            return image.to(config.device), self.labels[idx]
