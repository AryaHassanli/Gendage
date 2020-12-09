import os
import sys
import tarfile

import torch
from PIL import Image
from parse import parse
import torch.utils.data

from src.helpers.config import config
from . import DatasetHandler


class UTKFaceHandler(DatasetHandler):
    def __init__(self):
        self.directory = os.path.join(config.abs_datasets_dir, 'UTKFace')
        self.zipFile = os.path.join(config.abs_datasets_dir, 'UTKFace.tar.gz')
        self.dataset = None
        self.trainDataset = None
        self.testDataset = None
        self.validateDataset = None

    def create_dataset(self, transform, **kwargs):
        self.__prepare_on_disk()
        self.dataset = UTKFaceDataset(directory=self.directory, transform=transform, **kwargs)
        return self.dataset

    def __prepare_on_disk(self):
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
                tar.extractall(path=config.abs_datasets_dir)
                tar.close()
                print('Successfully extracted')
            except tarfile.TarError:
                print('Extract Failed!')
        else:
            sys.exit('UTK Zip file not found!')
        pass


class UTKFaceDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, **kwargs):
        self.directory = directory
        self.transform = transform
        self.labels = []
        self.images = []
        self.preload = kwargs.get('preload', 0)

        for i, file in enumerate(os.listdir(self.directory)):
            fileLabels = parse('{age}_{gender}_{}_{}.jpg.chip.jpg', file)
            if fileLabels is not None:
                if self.preload:
                    image = Image.open(os.path.join(self.directory, file))
                    if self.transform is not None:
                        image = self.transform(image).to(config.device)
                else:
                    image = os.path.join(self.directory, file)

                self.images.append(image)
                self.labels.append({
                    'age': int(fileLabels['age']),
                    'gender': int(fileLabels['gender']),
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
