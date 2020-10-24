import torch

from config import config


def mobilenet_v2():
    return torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=False).to(config.device)
