import torchvision.transforms as transforms

from dataLoaders.UTKFace import UTKFaceClass
from dataLoaders.AgeDB import AgeDBClass


def getLoaders(dataset, feature, splitSize, batchSize):
    if dataset == 'AgeDB':
        AgeDB = AgeDBClass(feature=feature)
        AgeDB.createDataset(transform=transforms.Compose(
            [transforms.Resize((60, 60)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
        AgeDB.splitDataset(trainSize=splitSize[0], validateSize=splitSize[1], testSize=splitSize[2])
        return AgeDB.dataLoaders(batchSize=batchSize)
    elif dataset == 'UTKFace':
        UTKFace = UTKFaceClass(feature=feature)
        UTKFace.createDataset(transform=transforms.Compose(
            [transforms.Resize((60, 60)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
        UTKFace.splitDataset(trainSize=splitSize[0], validateSize=splitSize[1], testSize=splitSize[2])
        return UTKFace.dataLoaders(batchSize=batchSize)
