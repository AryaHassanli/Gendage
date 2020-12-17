import numpy as np
import torch
import torch.nn.functional as functional

import models
from src.helpers.types import Features


class EncoderMultitask:
    def __init__(self, features: Features = Features(), device=torch.device('cpu')):
        self.features = features
        self.device = device

        self.model = models.get(age=features.age,
                                gender=features.gender,
                                pretrained='models/integrated_3.pt',
                                pretrained_encoder=None,
                                pretrained_age=None,
                                pretrained_gender=None).to(self.device)
        pass

    def __call__(self, faces):
        self.model.eval()
        face_tensors = [face.image_tensor for face in faces]
        face_tensors = torch.stack(face_tensors).to(self.device)
        ages = None
        genders = None
        with torch.no_grad():
            outputs = self.model(face_tensors)

            if self.features.age:
                output = outputs['age']
                preds = functional.softmax(output, dim=-1).cpu().numpy()
                classes = np.arange(0, preds.shape[1])
                predicted_output = (preds * classes).sum(axis=-1)
                ages = predicted_output

            if self.features.gender:
                output = outputs['gender']
                preds = functional.softmax(output, dim=-1).cpu().numpy()
                classes = np.arange(0, preds.shape[1])
                predicted_output = (preds * classes).sum(axis=-1)
                genders = predicted_output

        return {'ages': ages, 'genders': genders, 'encoded': outputs['encoded']}
