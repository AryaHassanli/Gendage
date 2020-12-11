import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim

from src.helpers import getDatasetHandler
from src.helpers import getEncoder
from src.helpers import getClassifier
from src.helpers import logger
from src.helpers.config import config

log = logger.Train()
log.environment()


# TODO: preprocess online and offline


def main():
    global minValidateMAE

    pre_transforms = config.pre_transforms
    models = {}
    for i, feature in enumerate(config.features):
        if feature not in models:
            models[feature] = getClassifier.get(config.classifiers[i], inputSize=512,
                                                numOfClasses=config.num_classes[i],
                                                pretrained=config.classifier_pretrain[i]).to(config.device)
        model = models[feature]

        # Creating an instance of desired dataset_handler
        dataset_handler = getDatasetHandler.get(dataset=config.datasets[i])

        # Creating the dataset object. It will take care of Downloading and unpacking the dataset if it's not available
        # If the dataset is preprocessed before,
        # it can be used by setting the --usePreprocessed argument in command-line
        # preload option, will load the whole dataset to memory on init. use it with --preload
        dataset_handler.create_dataset(transform=pre_transforms,
                                       preload=config.preload
                                       )
        # Splitting the created dataset in train, validate, and test set.
        train_loader, validate_loader, test_loader = dataset_handler.get_loaders(train_size=config.split_size[0],
                                                                                 validate_size=config.split_size[1],
                                                                                 test_size=config.split_size[2],
                                                                                 batch_size=config.batch_size)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.001)
        criterion = nn.CrossEntropyLoss().to(config.device)
        minValidateMAE = np.inf
        if len(train_loader) > 0:
            for epoch in range(1, config.epochs + 1):
                log.epochBegin(epoch, config.epochs)

                train(model, train_loader, criterion, optimizer, feature=feature)

                if len(validate_loader) > 0:
                    validate(model, validate_loader, criterion, feature=feature)

        if len(test_loader) > 0:
            test(model, test_loader, criterion, feature=feature)


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


path = config.encoder_pretrain
encoder = getEncoder.get(config.encoder, pretrained=path).to(config.device)
encoder.eval()


def train(model, train_loader, criterion, optimizer, feature=None):
    global log
    log.trainBegin()
    loss_monitor = AverageMeter()
    acc_monitor = AverageMeter()

    model.train()
    num_of_batches = len(train_loader)
    for batch, (x, y) in enumerate(train_loader):
        x = x.to(config.device)
        y = y[feature].to(config.device) if feature is not None else y.to(config.device)

        x = encoder(x).detach().to(config.device)

        # compute output
        outputs = model(x)

        batch_size = x.size(0)
        # calc loss
        loss = criterion(outputs, y)
        batch_loss = loss.item()

        # calc accuracy
        _, predicted = outputs.max(1)
        correct_count = predicted.eq(y).sum().item()

        # measure accuracy and record loss
        loss_monitor.update(batch_loss, batch_size)
        acc_monitor.update(correct_count, batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            batch_acc = correct_count / batch_size
            log.batch(feature, batch, num_of_batches, batch_loss, batch_acc)

    train_loss = loss_monitor.avg
    train_acc = acc_monitor.avg

    log.trainEnd(feature, train_loss, train_acc)
    return


minValidateMAE = np.inf


def validate(model, validate_loader, criterion, feature=None):
    global minValidateMAE
    global log
    model.eval()
    loss_monitor = AverageMeter()
    acc_monitor = AverageMeter()
    predicts = []
    gt = []

    with torch.no_grad():
        for batch, (x, y) in enumerate(validate_loader):
            x = x.to(config.device)
            y = y[feature].to(config.device) if feature is not None else y.to(config.device)

            x = encoder(x).detach()

            # compute output
            outputs = model(x)

            predicts.append(torch.nn.functional.softmax(outputs, dim=-1).cpu().numpy())
            gt.append(y.cpu().numpy())

            # calc loss
            loss = criterion(outputs, y)
            batch_loss = loss.item()

            batch_size = x.size(0)
            # calc accuracy
            _, predicted = outputs.max(1)
            correct_count = predicted.eq(y).sum().item()

            # measure accuracy and record loss
            loss_monitor.update(batch_loss, batch_size)
            acc_monitor.update(correct_count, batch_size)

    predicts = np.concatenate(predicts, axis=0)
    classes = np.arange(0, predicts.shape[1])
    gt = np.concatenate(gt, axis=0)
    ave_predicts = (predicts * classes).sum(axis=-1)
    diff = ave_predicts - gt
    mae = np.abs(diff).mean()

    validate_loss = loss_monitor.avg
    validate_acc = acc_monitor.avg
    validate_mae = mae

    is_learned = validate_mae < minValidateMAE
    if is_learned:
        minValidateMAE = validate_mae
        torch.save(model.state_dict(),
                   os.path.join(config.output_dir, str(feature) + '_model.pt'))

    log.validate(feature, validate_loss, validate_acc, validate_mae, is_learned)
    return


def test(model, test_loader, criterion, feature=None):
    global log
    model.eval()
    loss_monitor = AverageMeter()
    acc_monitor = AverageMeter()

    predicts = []
    gt = []
    with torch.no_grad():
        for batch, (x, y) in enumerate(test_loader):
            x = x.to(config.device)
            y = y[feature].to(config.device) if feature is not None else y.to(config.device)

            x = encoder(x).detach()

            # compute output
            outputs = model(x)
            predicts.append(torch.nn.functional.softmax(outputs, dim=-1).cpu().numpy())
            gt.append(y.cpu().numpy())

            # calc loss
            loss = criterion(outputs, y)
            batch_loss = loss.item()

            batch_size = x.size(0)

            # calc accuracy
            _, predicted = outputs.max(1)
            correct_count = predicted.eq(y).sum().item()

            # measure accuracy and record loss
            loss_monitor.update(batch_loss, batch_size)
            acc_monitor.update(correct_count, batch_size)

    predicts = np.concatenate(predicts, axis=0)
    classes = np.arange(0, predicts.shape[1])
    gt = np.concatenate(gt, axis=0)
    ave_predicts = (predicts * classes).sum(axis=-1)
    diff = ave_predicts - gt
    mae = np.abs(diff).mean()

    test_loss = loss_monitor.avg
    test_acc = acc_monitor.avg
    test_mae = mae

    log.test(feature, test_loss, test_acc, test_mae)
    return
