import os
import sys
import tarfile
import zipfile

import torch
from PIL import Image
from parse import parse
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np

from config import config
from facenet_pytorch import MTCNN


class AgeDBClass:
    def __init__(self, feature='gender'):
        self.directory = os.path.join(config.absDatasetDir, 'AgeDB')
        self.zipFile = os.path.join(config.absDatasetDir, 'AgeDB.zip')
        self.feature = feature
        self.dataset = None
        self.trainDataset = None
        self.testDataset = None
        self.validateDataset = None

    def createDataset(self, transform=None):
        self.__prepareOnDisk()
        self.dataset = AgeDBDataset(directory=self.directory, feature=self.feature, transform=transform)
        return self.dataset

    def splitDataset(self, trainSize=0.7, validateSize=0.2, testSize=0.1):
        if round(trainSize + validateSize + testSize, 1) != 1.0:
            sys.exit("Sum of the percentages should be equal to 1. it's " + str(
                trainSize + validateSize + testSize) + " now!")
        trainLen = int(len(self.dataset) * trainSize)
        validateLen = int(len(self.dataset) * validateSize)
        testLen = len(self.dataset) - trainLen - validateLen
        self.trainDataset, self.validateDataset, self.testDataset = random_split(
            self.dataset, [trainLen, validateLen, testLen])

    def dataLoaders(self, batchSize=15):
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
                zf.extractall(pwd=b'***REMOVED***', path=config.absDatasetDir)
            print('Successfully extracted')
        else:
            sys.exit('AgeDB Zip file not found!')
            # TODO: In case of zip file is missing, simply download it!


class AgeDBDataset(Dataset):
    def __init__(self, directory=None, feature='gender', transform=None):
        self.directory = directory
        self.transform = transform
        genderToClassId = {'m': 0, 'f': 1}
        self.labels = []
        self.imagesPath = []
        self.mtcnn = MTCNN(keep_all=True, device=config.device)
        for file in os.listdir(directory):
            label = parse('{}_{}_{age}_{gender}.jpg', file)
            if label is not None:
                self.imagesPath.append(file)
                if feature == 'gender':
                    self.labels.append(genderToClassId[label['gender']])
                elif feature == 'age':
                    age = int(label['age'])
                    # self.labels.append(age)

                    if 0 <= age < 2:
                        self.labels.append(0)
                    elif 2 <= age < 5:
                        self.labels.append(1)
                    elif 5 <= age < 9:
                        self.labels.append(2)
                    elif 9 <= age < 16:
                        self.labels.append(3)
                    elif 16 <= age < 20:
                        self.labels.append(4)
                    elif 20 <= age < 30:
                        self.labels.append(5)
                    elif 30 <= age < 40:
                        self.labels.append(6)
                    elif 40 <= age < 50:
                        self.labels.append(7)
                    elif 50 <= age < 60:
                        self.labels.append(8)
                    elif 60 <= age < 70:
                        self.labels.append(9)
                    elif 70 <= age:
                        self.labels.append(10)

        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imagePath = os.path.join(self.directory,
                                 self.imagesPath[idx])
        image = Image.open(imagePath)
        #image = np.array(image)
        #print(image.shape)
        # boxes, _ = self.mtcnn.detect(image)
        if image.mode == 'L':
            image = image.convert(mode='RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image.to(config.device), self.labels[idx]
