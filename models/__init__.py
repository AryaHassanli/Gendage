import os

import models.integrated
from models.integrated import IntegratedModel

models_dir = 'models'
__all__ = [f[:-3] for f in os.listdir(models_dir) if
           os.path.isfile(os.path.join(models_dir, f)) and not f.endswith(
               ('__init__.py', '__pycache__')) and f.endswith('.py')]


def get(age=True,
        gender=True,
        pretrained=None,
        device='cpu',
        **kwargs
        ) -> IntegratedModel:
    model = models.integrated.integrated(age=age,
                                         gender=gender,
                                         pretrained=pretrained,
                                         device=device,
                                         **kwargs).to(device)
    return model
