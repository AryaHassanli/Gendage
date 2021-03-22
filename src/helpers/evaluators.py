import torch

import models
from src.helpers.types import Features


class EncoderMultitask:
    def __init__(self, features: Features = Features(), device=torch.device('cpu')):
        self.features = features
        self.device = device

        self.model = models.get(age=features.age,
                                gender=features.gender,
                                pretrained='models/integrated.pt',
                                ).to(self.device)

    def __call__(self, faces):
        self.model.eval()
        face_tensors = [face.image_tensor for face in faces]
        face_tensors = torch.stack(face_tensors).to(self.device)

        ages = None
        genders = None
        with torch.no_grad():
            outputs = self.model(face_tensors)

            # TODO!: IMPORTANT
            """
                preds = F.softmax(gender, dim=-1).cpu().numpy()
                classes = np.arange(0, 2)
                predicted_output = (preds * classes).sum(axis=-1)
                gender = predicted_output
            """

            if self.features.age:
                ages = outputs['age']

            if self.features.gender:
                genders = outputs['gender']

        return {'ages': ages, 'genders': genders, 'encoded': outputs['encoded']}
