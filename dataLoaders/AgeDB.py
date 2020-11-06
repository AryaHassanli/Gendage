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
        self.usePreProcessed = kwargs.get('usePreProcessed', 1)
        self.__prepareOnDisk()
        self.dataset = AgeDBDataset(directory=self.directory,
                                    feature=feature,
                                    transform=transform,
                                    usePreProcessed=self.usePreProcessed,
                                    **kwargs)
        return self.dataset

    def splitDataset(self, trainSize=0.7, validateSize=0.2, testSize=0.1, **kwargs):
        if round(trainSize + validateSize + testSize, 1) != 1.0:
            sys.exit("Sum of the percentages should be equal to 1. it's " + str(
                trainSize + validateSize + testSize) + " now!")
        trainLen = int(len(self.dataset) * trainSize)
        validateLen = int(len(self.dataset) * validateSize)
        testLen = len(self.dataset) - trainLen - validateLen
        self.trainDataset, self.validateDataset, self.testDataset = random_split(
            self.dataset, [trainLen, validateLen, testLen])

    def dataLoaders(self, batchSize=15, **kwargs):
        trainLoader = DataLoader(self.trainDataset, batch_size=batchSize)
        validateLoader = DataLoader(self.validateDataset, batch_size=batchSize)
        testLoader = DataLoader(self.testDataset, batch_size=batchSize)
        return trainLoader, validateLoader, testLoader

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

        self.minAge = 1000
        self.maxAge = 0

        for i, file in enumerate(os.listdir(self.directory)):
            if kwargs.get('reduced', 0):
                if i % 20:
                    continue
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
        image = Image.open(self.imagePath[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image.to(config.device), self.labels[idx]
