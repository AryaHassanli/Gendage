import torch
from torch.utils.data import DataLoader

from config import config
from dataLoaders import UTKFaceClass

config.set(baseDir=None,
           datasetDir=None)  # nothing to set

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

UTKFace = UTKFaceClass()
UTKFace.importDataset()
UTKFace.splitDataset()

trainLoader = DataLoader(UTKFace.trainDataset)
validateLoader = DataLoader(UTKFace.validateDataset)
testLoader = DataLoader(UTKFace.testDataset)
