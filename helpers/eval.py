import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from config import config


def faceImage(face, model, features):
    model.eval()
    with torch.no_grad():
        preTransforms = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.Pad(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        face = preTransforms(face)
        face = face.unsqueeze(0).to(config.device)
        outputs = model(face)

        preds = {feature: [] for feature in features}
        ave_preds = {feature: 0 for feature in features}
        for feature in features:
            preds[feature].append(F.softmax(outputs[feature], dim=-1).cpu().numpy())
            preds[feature] = np.concatenate(preds[feature], axis=0)
            classes = np.arange(0, preds[feature].shape[1])
            ave_preds[feature] = (preds[feature] * classes).sum(axis=-1)[0]
    return ave_preds
