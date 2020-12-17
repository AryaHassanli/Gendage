import os
import sys
import tarfile

import torch
import torch.utils.data
from PIL import Image
from parse import parse


class UTKFaceHandler:
    def __init__(self, datasets_dir, preload=False, use_preprocessed=True, device: torch.device = torch.device('cpu')):
        self.device = device
        self.preload = preload

        self.datasets_dir = datasets_dir
        self.directory = os.path.join(datasets_dir, 'UTKFace' if not use_preprocessed else 'UTKFace_preprocessed')
        self.zipFile = os.path.join(datasets_dir, 'UTKFace.tar.gz')

        self.trainDataset = None
        self.testDataset = None
        self.validateDataset = None

        self.__prepare_on_disk()

    def get_loaders(self, transform, train_size=0.7, validate_size=0.2, test_size=0.1, batch_size=15, **kwargs):
        if round(train_size + validate_size + test_size, 1) > 1.0:
            sys.exit("Sum of the percentages should be less than 1. it's " + str(
                train_size + validate_size + test_size) + " now!")

        dataset = UTKFaceDataset(directory=self.directory,
                                 transform=transform,
                                 preload=self.preload,
                                 device=self.device,
                                 **kwargs)

        train_len = int(len(dataset) * train_size)
        validate_len = int(len(dataset) * validate_size)
        test_len = int(len(dataset) * test_size)
        others_len = len(dataset) - train_len - validate_len - test_len

        self.trainDataset, self.validateDataset, self.testDataset, _ = torch.utils.data.random_split(
            dataset, [train_len, validate_len, test_len, others_len])

        # noinspection PyUnusedLocal
        dataset = None

        train_loader = torch.utils.data.DataLoader(self.trainDataset, batch_size=batch_size)
        validate_loader = torch.utils.data.DataLoader(self.validateDataset, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(self.testDataset, batch_size=batch_size)

        return train_loader, validate_loader, test_loader

    def __prepare_on_disk(self):
        # TODO: Unzipping process is not working
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
                tar.extractall(path=self.datasets_dir)
                tar.close()
                print('Successfully extracted')
            except tarfile.TarError:
                print('Extract Failed!')
        else:
            sys.exit('UTK Zip file not found!')
        pass


class UTKFaceDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, device: torch.device = torch.device('cpu'), **kwargs):
        self.directory = directory
        self.transform = transform
        self.labels = []
        self.images = []
        self.preload = kwargs.get('preload', 0)
        self.device = device

        for i, file in enumerate(os.listdir(self.directory)):
            file_labels = parse('{age}_{gender}_{}_{}.jpg', file)
            if file_labels is not None:
                if int(file_labels['age']) > 120 or int(file_labels['gender']) > 1:
                    continue

                if self.preload:
                    image = Image.open(os.path.join(self.directory, file))
                    if self.transform is not None:
                        image = self.transform(image).to(device)
                else:
                    image = os.path.join(self.directory, file)

                self.images.append(image)
                self.labels.append({
                    'age': int(file_labels['age']),
                    'gender': int(file_labels['gender']),
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
                image = self.transform(image).to(self.device)

        labels = {'age': self.labels[idx]['age'], 'gender': self.labels[idx]['gender']}
        return image.to(self.device), labels
