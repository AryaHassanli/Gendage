import torch

import models.integrated
from models.integrated import IntegratedModel


def get(age: bool = True,
        gender: bool = True,
        pretrained: str = None,
        device: torch.device = torch.device('cpu'),
        **kwargs
        ) -> IntegratedModel:
    """
    Returns the integrated model with the given options. If a feature (gender or age) is set to False the model
    will not produce it. To load a pre-trained model give the path to state_dict to `pretrained` parameter.

    Parameters
    ----------
    age: bool
    gender: bool
    pretrained: str
    device: torch.device

    Returns
    -------
        Integrated Model
    """
    model = models.integrated.integrated(age=age,
                                         gender=gender,
                                         pretrained=pretrained,
                                         device=device,
                                         **kwargs).to(device)
    return model
