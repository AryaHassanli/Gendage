import os
import sys
import tarfile

import torch
from parse import parse
from skimage import io
from torch.utils.data import Dataset, random_split

from config import config


class UTKFaceClass:
    def __init__(self):
        self.dataset = None
        self.trainDataset = None
        self.testDataset = None
        self.validateDataset = None

    def importDataset(self, transform=None):
        """
        The method to generate the UTKFaceDataset from file.
        The database should be either available extracted on
        config.datasetDir/UTKFace or as UTKFace.tar.gz on config.datasetDir

        Args:
            transform:

        Returns:
            UTKFaceDataset Dataset object
            The usage of return value is optional.
        """
        self.dataset = UTKFaceDataset(transform)
        return self.dataset

    def splitDataset(self, trainPercentage=70, validatePercentage=20, testPercentage=10):
        """
        Splits the dataset to three subsets for train, validation, and test.
        The splitted dataset will be on:
            self.trainDataset
            self.validateDataset
            self.testDataset
        The usage of the returned value is optional.

        Args:
            trainPercentage: percentage of the main database to be allocated to *train* set
            validatePercentage: percentage of the main database to be allocated to *validate* set
            testPercentage: percentage of the main database to be allocated to *test* set

        Returns:
            [trainDataset, validateDataset, testDataset]
        """
        if trainPercentage + validatePercentage + testPercentage != 100:
            sys.exit("Sum of the percentages should be equal to 100.")
        trainSize = int(len(self.dataset) * trainPercentage / 100)
        validateSize = int(len(self.dataset) * validatePercentage / 100)
        testSize = len(self.dataset) - trainSize - validateSize
        self.trainDataset, self.validateDataset, self.testDataset = random_split(
            self.dataset, [trainSize, validateSize, testSize], generator=torch.Generator().manual_seed(42))

    def generateDataLoaders(self):
        pass


class UTKFaceDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        self.directory = os.path.join(config.absDatasetDir, 'UTKFace')
        self.zipFile = os.path.join(config.absDatasetDir, 'UTKFace.tar.gz')
        self.prepareOnDisk()
        self.labels = []
        self.imagesPath = []
        for file in os.listdir(self.directory):
            label = parse('{age}_{gender}_{}_{}.jpg.chip.jpg', file)
            if label is not None:
                self.imagesPath.append(file)
                self.labels.append(int(label['age']))
        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.directory,
                                self.imagesPath[idx])
        image = io.imread(img_name)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]

    def prepareOnDisk(self):
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
