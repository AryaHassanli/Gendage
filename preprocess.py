import os
import shutil

import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from parse import parse

from config import config
from helpers import parseArguments

# Handle arguments
args = parseArguments.parse('train')
if args.gradient:
    config.set(datasetDir='/storage/datasets',
               outputDir='/artifacts')
else:
    config.set(datasetDir="D:\\MsThesis\\datasets",
               outputDir='output')


def preProcessAgeDB(mtcnn, img):
    if img.mode == 'L':
        img = img.convert(mode='RGB')
    img = np.array(img)
    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        return None
    if len(boxes) > 1:
        return None
    box = boxes[0]
    box = box.astype(int)
    x1, y1, x2, y2 = box
    # Todo: use min and max!
    if x1 < 0 or y1 < 0:
        return None
    croppedFace = img[y1:y2, x1:x2, :]
    img = Image.fromarray(croppedFace)
    x, y = img.size
    size = max(x, y)
    imgSq = Image.new('RGB', (size, size), (0, 0, 0))
    imgSq.paste(img, (int((size - x) / 2), int((size - y) / 2)))
    return imgSq


if args.dataset == 'AgeDB':
    directory = os.path.join(config.absDatasetDir, 'AgeDB')
    if not os.path.exists(directory):
        # TODO: Look for Zip file or Download it!
        print('AgeDB is not available!')
        exit()
    preProcessedDirectory = os.path.join(directory, 'preProcessed')

    mtcnn = MTCNN(keep_all=True, device=config.device)

    if os.path.exists(preProcessedDirectory):
        shutil.rmtree(preProcessedDirectory)
    os.mkdir(preProcessedDirectory)

    for file in os.listdir(directory):
        # TODO: Print progress
        print('\r Working on:', file, '\t\t')
        label = parse('{}_{}_{age}_{gender}.jpg', file)
        if label is None:
            continue
        image = Image.open(os.path.join(directory, file))
        image = preProcessAgeDB(mtcnn, image)
        if image is None:
            continue
        image.save(os.path.join(preProcessedDirectory, file))
