import os
import sys
import zipfile

import torch
from PIL import Image
from parse import parse
import torch.utils.data

from config import config
from dataLoaders.datasetHandler import DatasetHandler


class AgeDBHandler(DatasetHandler):
    def __init__(self):
        self.directory = os.path.join(config.absDatasetsDir, 'AgeDB')
        self.zipFile = os.path.join(config.absDatasetsDir, 'AgeDB.zip')
        self.dataset = None
        self.trainDataset = None
        self.testDataset = None
        self.validateDataset = None

    def createDataset(self, transform, **kwargs):
        self.__prepareOnDisk()
        self.dataset = AgeDBDataset(directory=self.directory,
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
                zf.extractall(pwd=b'***REMOVED***', path=config.absDatasetsDir)
            print('Successfully extracted')
        else:
            sys.exit('AgeDB Zip file not found!')
            # TODO: In case of zip file is missing, simply download it!


class AgeDBDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, **kwargs):
        self.directory = directory if kwargs.get('usePreprocessed', 0) == 0 else os.path.join(directory, 'preProcessed')
        self.transform = transform
        genderToClassId = {'m': 0, 'f': 1}
        self.labels = []
        self.images = []
        self.preload = kwargs.get('preload', 0)

        for i, file in enumerate(os.listdir(self.directory)):
            fileLabels = parse('{}_{}_{age}_{gender}.jpg', file)
            if fileLabels is None:
                continue
            if self.preload:
                image = Image.open(os.path.join(self.directory, file)).convert('RGB')
                if self.transform is not None:
                    image = self.transform(image).to(config.device)
            else:
                image = os.path.join(self.directory, file)

            self.images.append(image)
            gender = genderToClassId[fileLabels['gender']]
            age = int(fileLabels['age'])
            self.labels.append({
                'age': age,
                'gender': gender
            })
        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]

        if not self.preload:
            image = Image.open(image)
            if self.transform is not None:
                image = self.transform(image).to(config.device)

        labels = {'age': self.labels[idx]['age'], 'gender': self.labels[idx]['gender']}
        return image.to(config.device), labels
