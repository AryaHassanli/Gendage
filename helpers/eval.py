import time

import torch
import torchvision.transforms as transforms
from PIL import Image

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
        ageModel = getNet.get('mobilenet_v3', mode='small', n_class=1)
        ageModel.load_state_dict(
            torch.load('D:/Gendage/output/2020-11-03_19-27-36_mobilenet_v3_regression_AgeDB_age_momentum/model.pt')
        )
        ageModel.eval()

    global totalError, totalFaces, totalTime, ageLabels
    transform = transforms.Compose(
        [transforms.Resize((100, 100)),
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

    age = ageModel(faceSq)
    age = str(round(age.item(), 1))

    endTime = time.time()

    totalFaces += 1
    totalTime += endTime - startTime
    return [age]
