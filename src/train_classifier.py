import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, fixed_image_standardization

import dataLoaders
import models
from src.helpers import AverageMeter
from src.helpers import logger
from src.helpers.types import Features, TrainParameters

log: logger.Train


def main(output_dir: str = 'output',
         features: Features = Features(),
         parameters: TrainParameters = TrainParameters(),
         datasets_dir: str = 'datasets',
         dataset: str = 'UTKFace',
         pretrained=None,
         pretrained_encoder='models/encoder/05_mobilenet_v3_small_003.pt',
         preload: bool = False,
         use_preprocessed: bool = False,
         device: torch.device = torch.device('cpu')
         ):
    global log
    log = logger.Train(output_dir)

    model = models.get(
        age=features.age,
        gender=features.gender,
        pretrained=pretrained,
        pretrained_encoder=pretrained_encoder
    ).to(device)

    mtcnn = MTCNN(
        keep_all=True,
        min_face_size=100,
        image_size=160,
        margin=14,
        selection_method="center_weighted_size",
        post_process=True,
        device=device,
    )

    pre_transforms = transforms.Compose([
        transforms.Resize((160, 160)),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization])

    dataset_handler = dataLoaders.get(
        dataset=dataset,
        datasets_dir=datasets_dir,
        preload=preload,
        use_preprocessed=use_preprocessed,
        device=device)
    train_loader, validate_loader, test_loader = dataset_handler.get_loaders(
        transform=pre_transforms,
        train_size=parameters.train_size,
        validate_size=parameters.validate_size,
        test_size=parameters.test_size,
        batch_size=parameters.batch_size)

    criterion = nn.CrossEntropyLoss().to(device)

    if len(train_loader) > 0:
        for epoch in range(1, parameters.epochs + 1):
            log.epochBegin(epoch, parameters.epochs)
            train_all(model, features, train_loader, criterion, parameters, device)

            if len(validate_loader) > 0:
                validate_all(model, features, validate_loader, criterion, output_dir, device)

    if len(test_loader) > 0:
        test_all(model, features, test_loader, criterion, device)

    torch.save(model.state_dict(),
               os.path.join(output_dir, 'final_model.pt'))
    pass


def train_all(model: models.IntegratedModel, features: Features, train_loader, criterion, parameters, device):
    if features.age:
        train_age(model, train_loader, criterion, parameters, device)
    if features.gender:
        train_gender(model, train_loader, criterion, parameters, device)
    pass


def train_age(model: models.IntegratedModel, train_loader, criterion, parameters, device):
    global log
    log.trainBegin()
    loss_monitor = AverageMeter()
    acc_monitor = AverageMeter()

    model.encoder.eval()
    model.gender.eval()
    model.age.train()

    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.gender.parameters():
        param.requires_grad = False
    for param in model.age.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.age.parameters(),
                                  lr=parameters.lr,
                                  weight_decay=0.001)

    num_of_batches = len(train_loader)
    for batch, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y['age'].to(device)

        # compute output
        outputs = model(x)
        outputs = outputs['age']

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
            log.batch('age', batch, num_of_batches, batch_loss, batch_acc)

    train_loss = loss_monitor.avg
    train_acc = acc_monitor.avg

    log.trainEnd('age', train_loss, train_acc)
    return


def train_gender(model: models.IntegratedModel, train_loader, criterion, parameters, device):
    global log
    log.trainBegin()
    loss_monitor = AverageMeter()
    acc_monitor = AverageMeter()

    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.gender.parameters():
        param.requires_grad = True
    for param in model.age.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.gender.parameters(),
                                  lr=parameters.lr,
                                  weight_decay=0.001)

    num_of_batches = len(train_loader)
    for batch, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y['gender'].to(device)

        # compute output
        outputs = model(x)
        outputs = outputs['gender']

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
            log.batch('gender', batch, num_of_batches, batch_loss, batch_acc)

    train_loss = loss_monitor.avg
    train_acc = acc_monitor.avg

    log.trainEnd('gender', train_loss, train_acc)
    return


minValidateMAE = np.inf


def validate_all(model: models.IntegratedModel, features: Features, validate_loader, criterion, output_dir,
                 device):
    if features.age:
        validate_age(model, validate_loader, criterion, output_dir, device)
    if features.gender:
        validate_gender(model, validate_loader, criterion, output_dir, device)
    pass


def validate_age(model: models.IntegratedModel, validate_loader, criterion, output_dir, device):
    global minValidateMAE
    global log

    model.eval()
    loss_monitor = AverageMeter()
    acc_monitor = AverageMeter()
    predicts = []
    gt = []

    with torch.no_grad():
        for batch, (x, y) in enumerate(validate_loader):
            x = x.to(device)
            y = y['age'].to(device)

            # compute output
            outputs = model(x)
            outputs = outputs['age']

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
                   os.path.join(output_dir, 'model.pt'))

    log.validate('age', validate_loss, validate_acc, validate_mae, is_learned)
    return


def validate_gender(model: models.IntegratedModel, validate_loader, criterion, output_dir, device):
    global minValidateMAE
    global log

    model.eval()
    loss_monitor = AverageMeter()
    acc_monitor = AverageMeter()
    predicts = []
    gt = []

    with torch.no_grad():
        for batch, (x, y) in enumerate(validate_loader):
            x = x.to(device)
            y = y['gender'].to(device)

            # compute output
            outputs = model(x)
            outputs = outputs['gender']

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
                   os.path.join(output_dir, 'model.pt'))

    log.validate('gender', validate_loss, validate_acc, validate_mae, is_learned)
    return


def test_all(model: models.IntegratedModel, features: Features, test_loader, criterion, device):
    if features.age:
        test_age(model, test_loader, criterion, device)
    if features.gender:
        test_gender(model, test_loader, criterion, device)
    pass


def test_age(model, test_loader, criterion, device):
    global log

    model.eval()
    loss_monitor = AverageMeter()
    acc_monitor = AverageMeter()

    predicts = []
    gt = []
    with torch.no_grad():
        for batch, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y['age'].to(device)

            # compute output
            outputs = model(x)
            outputs = outputs['age']
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

    log.test('age', test_loss, test_acc, test_mae)
    return


def test_gender(model, test_loader, criterion, device):
    global log

    model.eval()
    loss_monitor = AverageMeter()
    acc_monitor = AverageMeter()

    predicts = []
    gt = []
    with torch.no_grad():
        for batch, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y['gender'].to(device)

            # compute output
            outputs = model(x)
            outputs = outputs['gender']
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

    log.test('gender', test_loss, test_acc, test_mae)
    return
