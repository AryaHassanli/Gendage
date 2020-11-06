import datetime
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn import MSELoss
from torch.optim import SGD, Adam

from config import config
from helpers.getDatasetHandler import getDatasetHandler
from helpers.getNet import getNet
from helpers.parseArguments import parseArguments

# Handle arguments
args = parseArguments('train')
outputSubDir = '_'.join(
    [datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.net, args.task, args.dataset, args.feature, args.tag])
if args.gradient:
    outputDir = os.path.join('/artifacts', outputSubDir)
    config.set(datasetDir='/storage/datasets',
               outputDir=outputDir)
else:
    outputDir = os.path.join('output', outputSubDir)
    config.set(datasetDir="D:\\MsThesis\\datasets",
               outputDir=outputDir)

# Setup Logging
log = logging.getLogger()
log.setLevel(logging.INFO)
output_file_handler = logging.FileHandler(os.path.join(config.outputDir, 'output.log'))
stdout_handler = logging.StreamHandler(sys.stdout)
log.addHandler(output_file_handler)
log.addHandler(stdout_handler)

# set kwargs
kwargs = {'mode': 'small'}
if args.dataset == 'AgeDB':
    kwargs['usePreProcessed'] = 1
if args.task == 'regression':
    kwargs['num_classes'] = 1
    kwargs['n_class'] = 1

log.info(str(args))
log.info(str(kwargs))


def main():
    datasetHandler = getDatasetHandler(dataset=args.dataset)
    datasetHandler.createDataset(feature=args.feature,
                                 transform=transforms.Compose([
                                     transforms.Resize((60, 60)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                 ])
                                 )
    datasetHandler.splitDataset(trainSize=args.splitSize[0], validateSize=args.splitSize[1], testSize=args.splitSize[2],
                                **kwargs)
    trainLoader, validateLoader, testLoader = datasetHandler.dataLoaders(batchSize=args.batchSize, **kwargs)

    numOfEpochs = args.epochs

    model = getNet(args.net, **kwargs).to(config.device)

    torch.backends.cudnn.benchmark = True
    # Mostly from https://www.kaggle.com/basu369victor/pytorch-tutorial-the-classification
    if args.task == 'classification':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        n_epochs = numOfEpochs
        print_every = 10
        valid_loss_min = np.Inf
        val_loss = []
        val_acc = []
        train_loss = []
        train_acc = []
        total_step = len(trainLoader)
        startTime = time.time()
        for epoch in range(1, n_epochs + 1):
            running_loss = 0.0
            correct = 0
            total = 0
            log.info(f'Epoch {epoch}\n')
            for batch_idx, (data_, target_) in enumerate(trainLoader):
                data_, target_ = data_.to(config.device), target_.to(config.device)  # on GPU
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(data_)
                loss = criterion(outputs, target_)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                _, pred = torch.max(outputs, dim=1)
                correct += torch.sum(pred == target_).item()
                total += target_.size(0)
                if batch_idx % 20 == 0:
                    log.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                             .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
            train_acc.append(100 * correct / total)
            train_loss.append(running_loss / total_step)
            log.info(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')
            batch_loss = 0
            total_t = 0
            correct_t = 0
            with torch.no_grad():
                model.eval()
                for data_t, target_t in validateLoader:
                    data_t, target_t = data_t.to(config.device), target_t.to(config.device)  # on GPU
                    outputs_t = model(data_t)
                    loss_t = criterion(outputs_t, target_t)
                    batch_loss += loss_t.item()
                    _, pred_t = torch.max(outputs_t, dim=1)
                    correct_t += torch.sum(pred_t == target_t).item()
                    total_t += target_t.size(0)
                val_acc.append(100 * correct_t / total_t)
                val_loss.append(batch_loss / len(validateLoader))
                network_learned = batch_loss < valid_loss_min
                log.info(
                    f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
                # Saving the best weight
                if network_learned:
                    valid_loss_min = batch_loss
                    torch.save(model.state_dict(),
                               os.path.join(config.outputDir,
                                            args.net + '_' + args.dataset + '_' + args.feature + '_model.pt'))
                    log.info('Detected network improvement, saving current model')
            model.train()

        batch_loss = 0
        total_t = 0
        correct_t = 0
        with torch.no_grad():
            model.eval()
            for data_t, target_t in testLoader:
                data_t, target_t = data_t.to(config.device), target_t.to(config.device)  # on GPU
                outputs_t = model(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)
            val_loss.append(batch_loss / len(validateLoader))
            log.info(f'test acc: {(100 * correct_t / total_t):.4f}\n')
        log.info("Total Time:{}s".format(time.time() - startTime))
    elif args.task == 'regression':
        criterion = MSELoss().to(config.device)
        optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.9)
        valid_loss_min = np.inf
        for epoch in range(1, numOfEpochs + 1):
            batch_loss = 0
            for batch_idx, (data_, target_) in enumerate(trainLoader):
                data_ = data_.to(config.device)
                target_ = target_.to(config.device).to(torch.float32)  # on GPU
                y_pred = model(data_)
                loss = criterion(y_pred.squeeze(1), target_)
                batch_loss = loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % 20 == 0:
                    log.info("batch: {}, loss: {}".format(batch_idx, batch_loss))
            log.info("Epoch: {}, loss: {}".format(epoch, batch_loss))
            with torch.no_grad():
                model.eval()
                val_loss = 0
                total_t = 0
                for data_t, target_t in validateLoader:
                    data_t, target_t = data_t.to(config.device), target_t.to(config.device)  # on GPU
                    outputs_t = model(data_t)
                    loss_t = criterion(outputs_t.squeeze(1), target_t)
                    val_loss += loss_t.item()
                    total_t += 1
                val_loss /= total_t
                network_learned = val_loss < valid_loss_min
                log.info('validation loss: {}\n'.format(np.mean(val_loss)))
                # Saving the best weight
                if network_learned:
                    valid_loss_min = val_loss
                    torch.save(model.state_dict(),
                               os.path.join(config.outputDir, 'model.pt'))
                    log.info('Detected network improvement, saving current model')
            model.train()

        pass


if __name__ == '__main__':
    main()
