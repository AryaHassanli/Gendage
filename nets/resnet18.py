import torch
from config import config


def resnet18():
    return torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False).to(config.device)
