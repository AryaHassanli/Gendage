import torch

from .AgeDB import *
from .UTKFace import *


def get(dataset: str,
        datasets_dir: str = 'datasets',
        preload: bool = False,
        device: torch.device = torch.device('cpu'),
        **kwargs):
    dataset_handler_class = None
    if dataset == 'AgeDB':
        dataset_handler_class = AgeDBHandler
    if dataset == 'UTKFace':
        dataset_handler_class = UTKFaceHandler

    data_obj = dataset_handler_class(datasets_dir=datasets_dir, preload=preload, device=device, **kwargs)
    return data_obj
