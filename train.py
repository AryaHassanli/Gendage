import datetime
import os

import numpy as np
import pytz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from config import config
from helpers import getDatasetHandler
from helpers import getNet
from helpers import logger
from helpers import parseArguments

# Handle arguments
args = parseArguments.parse('train')
outputSubDir = '_'.join(
    [datetime.datetime.now(pytz.utc).strftime("%Y-%m-%d_%H-%M-%S"), args.net, args.task, args.dataset,
     '+'.join(args.features),
     args.tag])

# Setting up the config according to the system that runs the code
# On Gradient the directories would be different
if args.gradient:
    outputDir = os.path.join('/artifacts', outputSubDir)
    config.set(datasetDir='/storage/datasets',
               outputDir=outputDir)
else:
    outputDir = os.path.join('output', outputSubDir)
    config.set(datasetDir="D:\\MsThesis\\datasets",
               outputDir=outputDir)

# Setup Logging
# The logs will shown on stdout and saved in $outputDir/output.log
log = logger.Train()

# Set the parameters for nets and datasets in kwargs to pass them to their handlers
kwargs = {
}

# Set the number of output classes to "one" if the task is regression
if args.task == 'regression':
    # num_classes has been used in resnet family models and mobilenet_v2
    kwargs['num_classes'] = 1
    # n_class has been used in mobilenet_v3 model
    kwargs['n_class'] = 1
else:
    kwargs['num_classes'] = 118
    kwargs['n_class'] = 118
    kwargs['age_classes'] = 118

# TODO: log the train parameters
log.environment(args, kwargs)


def main():
    preTransforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.Pad(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    runtimeTrainTransform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomRotation(30),
        ], 0.7),
        transforms.RandomPerspective(0.5, 0.5),
        transforms.RandomHorizontalFlip(0.5)
    ])
    validTransform = preTransforms

    log.transforms(preTransforms, runtimeTrainTransform, validTransform)

    # Creating an instance of desired datasetHandler
    datasetHandler = getDatasetHandler.get(dataset=args.dataset)

    # Creating the dataset object. It will take care of Downloading and unpacking the dataset if it's not available
    # If the dataset is preprocessed before, it can be used by setting the --usePreprocessed argument in command-line
    # preload option, will load the whole dataset to memory on init. use it with --preload
    datasetHandler.createDataset(features=args.features,
                                 transform=preTransforms,
                                 preload=args.preload,
                                 usePreprocessed=args.usePreprocessed,
                                 **kwargs
                                 )

    # Splitting the created dataset in train, validate, and test set.
    trainLoader, validateLoader, testLoader = datasetHandler.getLoaders(trainSize=args.splitSize[0],
                                                                        validateSize=args.splitSize[1],
                                                                        testSize=args.splitSize[2],
                                                                        batchSize=args.batchSize, **kwargs)

    model = getNet.get(args.net, **kwargs).to(config.device)

    numOfEpochs = args.epochs
    torch.backends.cudnn.benchmark = True
    if args.task == 'classification':
        # https://github.com/VHCC/PyTorch-age-estimation
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
        ageCriterion = nn.CrossEntropyLoss().to(config.device)
        genderCriterion = nn.CrossEntropyLoss().to(config.device)
        criterions = {'age': ageCriterion, 'gender': genderCriterion}
        for epoch in range(1, numOfEpochs + 1):
            log.epochBegin(epoch, numOfEpochs)

            train(model, args.features, trainLoader, criterions, optimizer, runtimeTrainTransform)
            validate(model, args.features, validateLoader, criterions)

        test(model, args.features, testLoader, criterions)
        pass


class AverageMeter(object):
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


def train(model, features, trainLoader, criterions, optimizer, runtimeTrainTransform):
    lossMonitor = {feature: AverageMeter() for feature in features}
    accMonitor = {feature: AverageMeter() for feature in features}

    model.train()
    numOfBatches = len(trainLoader)
    for batch, (image, labels) in enumerate(trainLoader):
        image = image.to(config.device)
        labels = {feature: labels[feature].to(config.device) for feature in features}

        if runtimeTrainTransform is not None:
            image = runtimeTrainTransform(image)

        # compute output
        outputs = model(image)
        loss = {}
        batchLoss = {}
        predicted = {}
        correctCount = {}
        batchSize = image.size(0)
        for feature in features:
            # calc loss
            loss[feature] = criterions[feature](outputs[feature], labels[feature])
            batchLoss[feature] = loss[feature].item()

            # calc accuracy
            _, predicted[feature] = outputs[feature].max(1)
            correctCount[feature] = predicted[feature].eq(labels[feature]).sum().item()

            # measure accuracy and record loss
            lossMonitor[feature].update(batchLoss[feature], batchSize)
            accMonitor[feature].update(correctCount[feature], batchSize)

        # compute gradient and do SGD step
        loss = sum(loss.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            batchAcc = {feature: correctCount[feature] / batchSize for feature in features}
            log.batch(features, batch, numOfBatches, batchLoss, batchAcc)

    trainLoss = {feature: lossMonitor[feature].avg for feature in features}
    trainAcc = {feature: accMonitor[feature].avg for feature in features}

    log.train(features, trainLoss, trainAcc)
    return


minValidateMAE = {}


def validate(model, features, validateLoader, criterions):
    global minValidateMAE
    if len(minValidateMAE) == 0:
        minValidateMAE = {feature: np.inf for feature in features}

    model.eval()
    lossMonitor = {feature: AverageMeter() for feature in features}
    accMonitor = {feature: AverageMeter() for feature in features}
    preds = {feature: [] for feature in features}
    gt = {feature: [] for feature in features}

    with torch.no_grad():
        for batch, (image, labels) in enumerate(validateLoader):
            image = image.to(config.device)
            labels = {feature: labels[feature].to(config.device) for feature in features}

            # compute output
            outputs = model(image)
            for feature in features:
                preds[feature].append(F.softmax(outputs[feature], dim=-1).cpu().numpy())
                gt[feature].append(labels[feature].cpu().numpy())

            # calc loss
            loss = {feature: criterions[feature](outputs[feature], labels[feature]) for feature in features}
            batchLoss = {feature: loss[feature].item() for feature in features}

            predicted = {}
            correctCount = {}
            batchSize = image.size(0)
            for feature in features:
                # calc accuracy
                _, predicted[feature] = outputs[feature].max(1)
                correctCount[feature] = predicted[feature].eq(labels[feature]).sum().item()

                # measure accuracy and record loss
                lossMonitor[feature].update(batchLoss[feature], batchSize)
                accMonitor[feature].update(correctCount[feature], batchSize)

    mae = {}
    for feature in features:
        preds[feature] = np.concatenate(preds[feature], axis=0)
        classes = np.arange(0, preds[feature].shape[1])
        gt[feature] = np.concatenate(gt[feature], axis=0)
        ave_preds = (preds[feature] * classes).sum(axis=-1)
        diff = ave_preds - gt[feature]
        mae[feature] = np.abs(diff).mean()

    validateLoss = {feature: lossMonitor[feature].avg for feature in features}
    validateAcc = {feature: accMonitor[feature].avg for feature in features}
    validateMAE = {feature: mae[feature] for feature in features}

    isLearned = {feature: validateMAE[feature] < minValidateMAE[feature] for feature in features}
    # TODO
    isLearned = sum(isLearned.values())
    if isLearned:
        minValidateMAE = {feature: validateMAE[feature] for feature in features}
        torch.save(model.state_dict(),
                   os.path.join(config.outputDir, 'model.pt'))

    log.validate(features, validateLoss, validateAcc, validateMAE, isLearned)
    return


def test(model, features, testLoader, criterions):
    model.eval()
    lossMonitor = {feature: AverageMeter() for feature in features}
    accMonitor = {feature: AverageMeter() for feature in features}
    preds = {feature: [] for feature in features}
    gt = {feature: [] for feature in features}

    with torch.no_grad():
        for batch, (image, labels) in enumerate(testLoader):
            image = image.to(config.device)
            labels = {feature: labels[feature].to(config.device) for feature in features}

            # compute output
            outputs = model(image)
            for feature in features:
                preds[feature].append(F.softmax(outputs[feature], dim=-1).cpu().numpy())
                gt[feature].append(labels[feature].cpu().numpy())

            # calc loss
            loss = {feature: criterions[feature](outputs[feature], labels[feature]) for feature in features}
            batchLoss = {feature: loss[feature].item() for feature in features}

            predicted = {}
            correctCount = {}
            batchSize = image.size(0)
            for feature in features:
                # calc accuracy
                _, predicted[feature] = outputs[feature].max(1)
                correctCount[feature] = predicted[feature].eq(labels[feature]).sum().item()

                # measure accuracy and record loss
                lossMonitor[feature].update(batchLoss[feature], batchSize)
                accMonitor[feature].update(correctCount[feature], batchSize)

    mae = {}
    for feature in features:
        preds[feature] = np.concatenate(preds[feature], axis=0)
        classes = np.arange(0, preds[feature].shape[1])
        gt[feature] = np.concatenate(gt[feature], axis=0)
        ave_preds = (preds[feature] * classes).sum(axis=-1)
        diff = ave_preds - gt[feature]
        mae[feature] = np.abs(diff).mean()

    testLoss = {feature: lossMonitor[feature].avg for feature in features}
    testAcc = {feature: accMonitor[feature].avg for feature in features}
    testMAE = {feature: mae[feature] for feature in features}

    log.test(features, testLoss, testAcc, testMAE)
    return


if __name__ == '__main__':
    main()
