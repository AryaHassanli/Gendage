import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from config import config
from helpers.getLoaders import getLoaders
from helpers.getNet import getNet
from helpers.parseArguments import parseArguments

# Handle arguments
args = parseArguments('train')
if args.gradient:
    config.set(datasetDir='/storage/datasets',
               outputDir='/artifacts')
else:
    config.set(datasetDir="D:\\MsThesis\\datasets",
               outputDir='output')


def main():
    trainLoader, validateLoader, testLoader = getLoaders(dataset=args.dataset,
                                                         batchSize=args.batchSize,
                                                         feature=args.feature,
                                                         splitSize=args.splitSize,
                                                         transform=transforms.Compose(
                                                             [transforms.Resize((100, 100)),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                   std=[0.229, 0.224, 0.225])]),
                                                         usePreProcessed=1)

    numOfEpochs = args.epochs

    # Mostly from https://www.kaggle.com/basu369victor/pytorch-tutorial-the-classification
    model = getNet(args.net).to(config.device)

    torch.backends.cudnn.benchmark = True
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
        print(f'Epoch {epoch}\n')
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
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss / total_step)
        print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')
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
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
            # Saving the best weight
            if network_learned:
                valid_loss_min = batch_loss
                torch.save(model.state_dict(),
                           os.path.join(config.outputDir,
                                        args.net + '_' + args.dataset + '_' + args.feature + '_model.pt'))
                print('Detected network improvement, saving current model')
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
        print(f'test acc: {(100 * correct_t / total_t):.4f}\n')
    print("Total Time:{}s".format(time.time() - startTime))


if __name__ == '__main__':
    main()
