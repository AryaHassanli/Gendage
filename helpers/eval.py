import time

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from config import config
from helpers import getNet

genderModel = None
ageModel = None

ageLabels = {0: '0-2',
             1: '2-5',
             2: '5-9',
             3: '9-16',
             4: '16-20',
             5: '20-30',
             6: '30-40',
             7: '40-50',
             8: '50-60',
             9: '60-70',
             10: '>70'}
totalFaces = 0
totalError = 0
totalTime = 0


def gendageV1(face):
    """
    Two separate Models
    Gender: Classification. resnet18
    Age:    Classification on Age Groups. resnet50
    Args:
        face:

    Returns:

    """
    global genderModel, ageModel
    if genderModel is None:
        genderModel = getNet.get('resnet18')
        genderModel.load_state_dict(
            torch.load('D:/MsThesis/0-gender/resnet18_AgeDB_gender_model/resnet18_AgeDB_gender_model.pt'))
        genderModel.eval()
    if ageModel is None:
        ageModel = getNet.get('resnet50')
        ageModel.load_state_dict(
            torch.load('D:/MsThesis/1-age/AgeClasses_resnet50_AgeDB/resnet50_AgeDB_age_model.pt'))
        ageModel.eval()

    global totalError, totalFaces, totalTime, ageLabels
    transform = transforms.Compose(
        [transforms.Resize((60, 60)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])
    face = Image.fromarray(face)
    x, y = face.size
    size = max(x, y)
    faceSq = Image.new('RGB', (size, size), (0, 0, 0))
    faceSq.paste(face, (int((size - x) / 2), int((size - y) / 2)))
    faceSq = transform(faceSq)
    faceSq = faceSq.unsqueeze(0).to(config.device)

    startTime = time.time()

    gender = genderModel(faceSq)
    _, gender = torch.max(gender, dim=1)
    gender = 'male' if gender == 0 else 'female'

    age = ageModel(faceSq)
    _, age = torch.max(age, dim=1)
    age = age.cpu().numpy()
    age = age[0]
    age = ageLabels[age]

    endTime = time.time()

    totalFaces += 1
    totalTime += endTime - startTime
    return [gender, age]


def gendageV2(face):
    """
    For now, it just returns age.
    using regression task.
    Args:
        face:

    Returns:

    """
    global ageModel
    if ageModel is None:
        ageModel = getNet.get('resnet18', mode='small', n_class=118, num_classes=118).to(config.device)
        ageModel.load_state_dict(
            torch.load('D:/Gendage/output/model3.pt')
        )
        ageModel.eval()
    with torch.no_grad():
        global totalError, totalFaces, totalTime, ageLabels
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.Grayscale(3),
            transforms.Pad(10),
            transforms.ToTensor(),
        ])

        face = Image.fromarray(face)
        x, y = face.size
        size = max(x, y)
        # faceSq = Image.new('RGB', (size, size), (0, 0, 0))
        # faceSq.paste(face, (int((size - x) / 2), int((size - y) / 2)))
        faceSq = face
        faceSq = transform(faceSq)
        faceSq = faceSq.unsqueeze(0).to(config.device)

        startTime = time.time()

        outputs = F.softmax(ageModel(faceSq), dim=-1).cpu().numpy()
        ages = np.arange(0, 118)
        predicted_ages = (outputs * ages).sum(axis=-1)
        age = predicted_ages[0]
        age = str(age)
        """
        age = ageModel(faceSq)
        _, age = torch.max(age, dim=1)
        age = str(round(age.item(), 1))
        """
        endTime = time.time()

        totalFaces += 1
        totalTime += endTime - startTime
    return [age]


def gendageV3(face):
    global ageModel
    if ageModel is None:
        ageModel = getNet.get('resnet18', mode='small', n_class=118, num_classes=118).to(config.device)
        ageModel.load_state_dict(
            torch.load('D:/Gendage/output/model4.pt')
        )
        ageModel.eval()
    with torch.no_grad():
        global totalError, totalFaces, totalTime, ageLabels
        transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.Pad(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        face = transform(face)
        face = face.unsqueeze(0).to(config.device)
        outputs = F.softmax(ageModel(face), dim=-1).cpu().numpy()
        ages = np.arange(0, 118)
        predicted_ages = (outputs * ages).sum(axis=-1)
        age = predicted_ages[0]
        age = str(age)
    return [age]
