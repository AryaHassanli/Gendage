# noinspection PyUnresolvedReferences
from dataLoaders import *


def getLoaders(dataset, feature, splitSize, batchSize, transform, usePreProcessed=1):
    dataObj = eval(dataset+"."+dataset+'Class(feature="' + feature + '")')
    dataObj.createDataset(transform=transform,usePreProcessed=usePreProcessed)
    dataObj.splitDataset(trainSize=splitSize[0], validateSize=splitSize[1], testSize=splitSize[2])
    return dataObj.dataLoaders(batchSize=batchSize)
