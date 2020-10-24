import torchvision.transforms as transforms
# noinspection PyUnresolvedReferences
from dataLoaders import *


def getLoaders(dataset, feature, splitSize, batchSize):
    dataObj = eval(dataset+"."+dataset+'Class(feature="' + feature + '")')
    dataObj.createDataset(transform=transforms.Compose(
            [transforms.Resize((60, 60)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    dataObj.splitDataset(trainSize=splitSize[0], validateSize=splitSize[1], testSize=splitSize[2])
    return dataObj.dataLoaders(batchSize=batchSize)
