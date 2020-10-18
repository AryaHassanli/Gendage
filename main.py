import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from config import config
from dataLoaders import UTKFaceClass, AgeDBClass

# Handle arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('-G', '--gradient', action='store_true', help='to run on gradient')
parser.add_argument('--train', action='store_true', help='train')
args = parser.parse_args()

if args.gradient:
    config.set(datasetDir='/storage/datasets',
               outputDir='/artifacts')
    print(config.datasetDir)
else:
    config.set(datasetDir="D:\\MsThesis\\datasets",
               outputDir='output')
    print(config.datasetDir)

# Prepare UTKFace
UTKFace = UTKFaceClass()
UTKFace.createDataset(transform=transforms.Compose(
    [transforms.Resize((60, 60)),
     transforms.ToTensor()]))
UTKFace.splitDataset(trainSize=0.7, validateSize=0.2, testSize=0.1)

# Prepare AgeDB
AgeDB = AgeDBClass()
AgeDB.createDataset(transform=transforms.Compose(
    [transforms.Resize((60, 60)),
     transforms.ToTensor()]))
AgeDB.splitDataset(trainSize=0.7, validateSize=0.2, testSize=0.1)

trainLoader, validateLoader, testLoader = AgeDB.dataLoaders(batchSize=15)
