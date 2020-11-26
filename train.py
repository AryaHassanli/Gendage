import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1

from config import config
from defaults import trainConfig
from helpers import getDatasetHandler
from helpers import getNet
from helpers import logger
from helpers import parseArguments

# Handle arguments
cliArgs = parseArguments.parse('train')
fileArgs = trainConfig.options
config.setup(cliArgs, fileArgs)

# Setup Logging
# The logs will shown on stdout and saved in $outputDir/output.log
log = logger.Train()

log.environment()

minValidateMAE = np.inf
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(config.device)


def main():
    global minValidateMAE
    preTransforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.Pad(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    models = {}
    for i, feature in enumerate(config.features):
        if feature not in models:
            models[feature] = getNet.get(config.nets[i], numOfClasses=config.numOfClasses[i]).to(config.device)
        model = models[feature]

        # Creating an instance of desired datasetHandler
        datasetHandler = getDatasetHandler.get(dataset=config.datasets[i])

        # Creating the dataset object. It will take care of Downloading and unpacking the dataset if it's not available
        # If the dataset is preprocessed before,
        # it can be used by setting the --usePreprocessed argument in command-line
        # preload option, will load the whole dataset to memory on init. use it with --preload
        datasetHandler.createDataset(transform=preTransforms,
                                     preload=config.preload
                                     )
        # Splitting the created dataset in train, validate, and test set.
        trainLoader, validateLoader, testLoader = datasetHandler.getLoaders(trainSize=config.splitSize[0],
                                                                            validateSize=config.splitSize[1],
                                                                            testSize=config.splitSize[2],
                                                                            batchSize=config.batchSize)

        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.001)
        criterion = nn.CrossEntropyLoss().to(config.device)
        minValidateMAE = np.inf
        for epoch in range(1, config.epochs + 1):
            log.epochBegin(epoch, config.epochs)

            train(model, trainLoader, criterion, optimizer, feature=feature)
            validate(model, validateLoader, criterion, feature=feature)

        test(model, testLoader, criterion, feature=feature)


class AverageMeter(object):
    # https://github.com/VHCC/PyTorch-age-estimation
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def train(model, trainLoader, criterion, optimizer, feature=None):
    log.trainBegin()
    lossMonitor = AverageMeter()
    accMonitor = AverageMeter()

    model.train()
    numOfBatches = len(trainLoader)
    for batch, (x, y) in enumerate(trainLoader):
        x = x.to(config.device)
        y = y[feature].to(config.device) if feature is not None else y.to(config.device)

        x = resnet(x).detach().to(config.device)

        # compute output
        outputs = model(x)

        batchSize = x.size(0)
        # calc loss
        loss = criterion(outputs, y)
        batchLoss = loss.item()

        # calc accuracy
        _, predicted = outputs.max(1)
        correctCount = predicted.eq(y).sum().item()

        # measure accuracy and record loss
        lossMonitor.update(batchLoss, batchSize)
        accMonitor.update(correctCount, batchSize)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            batchAcc = correctCount / batchSize
            log.batch(feature, batch, numOfBatches, batchLoss, batchAcc)

    trainLoss = lossMonitor.avg
    trainAcc = accMonitor.avg

    log.trainEnd(feature, trainLoss, trainAcc)
    return


def validate(model, validateLoader, criterion, feature=None):
    global minValidateMAE

    model.eval()
    lossMonitor = AverageMeter()
    accMonitor = AverageMeter()
    preds = []
    gt = []

    with torch.no_grad():
        for batch, (x, y) in enumerate(validateLoader):
            x = x.to(config.device)
            y = y[feature].to(config.device) if feature is not None else y.to(config.device)

            x = resnet(x).detach()

            # compute output
            outputs = model(x)

            preds.append(F.softmax(outputs, dim=-1).cpu().numpy())
            gt.append(y.cpu().numpy())

            # calc loss
            loss = criterion(outputs, y)
            batchLoss = loss.item()

            batchSize = x.size(0)
            # calc accuracy
            _, predicted = outputs.max(1)
            correctCount = predicted.eq(y).sum().item()

            # measure accuracy and record loss
            lossMonitor.update(batchLoss, batchSize)
            accMonitor.update(correctCount, batchSize)

    preds = np.concatenate(preds, axis=0)
    classes = np.arange(0, preds.shape[1])
    gt = np.concatenate(gt, axis=0)
    ave_preds = (preds * classes).sum(axis=-1)
    diff = ave_preds - gt
    mae = np.abs(diff).mean()

    validateLoss = lossMonitor.avg
    validateAcc = accMonitor.avg
    validateMAE = mae

    isLearned = validateMAE < minValidateMAE
    if isLearned:
        minValidateMAE = validateMAE
        torch.save(model.state_dict(),
                   os.path.join(config.outputDir, str(feature) + '_model.pt'))

    log.validate(feature, validateLoss, validateAcc, validateMAE, isLearned)
    return


def test(model, testLoader, criterion, feature=None):
    model.eval()
    lossMonitor = AverageMeter()
    accMonitor = AverageMeter()

    preds = []
    gt = []
    with torch.no_grad():
        for batch, (x, y) in enumerate(testLoader):
            x = x.to(config.device)
            y = y[feature].to(config.device) if feature is not None else y.to(config.device)

            x = resnet(x).detach()

            # compute output
            outputs = model(x)
            preds.append(F.softmax(outputs, dim=-1).cpu().numpy())
            gt.append(y.cpu().numpy())

            # calc loss
            loss = criterion(outputs, y)
            batchLoss = loss.item()

            batchSize = x.size(0)

            # calc accuracy
            _, predicted = outputs.max(1)
            correctCount = predicted.eq(y).sum().item()

            # measure accuracy and record loss
            lossMonitor.update(batchLoss, batchSize)
            accMonitor.update(correctCount, batchSize)

    preds = np.concatenate(preds, axis=0)
    classes = np.arange(0, preds.shape[1])
    gt = np.concatenate(gt, axis=0)
    ave_preds = (preds * classes).sum(axis=-1)
    diff = ave_preds - gt
    mae = np.abs(diff).mean()

    testLoss = lossMonitor.avg
    testAcc = accMonitor.avg
    testMAE = mae

    log.test(feature, testLoss, testAcc, testMAE)
    return


if __name__ == '__main__':
    main()
