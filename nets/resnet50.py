import torch

from config import config


def resnet50():
    return torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False).to(config.device)
