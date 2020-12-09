import numpy as np
import torch
import torch.nn.functional as functional

from src.helpers.config import config


def encoder_multitask(face, features, encoder, classifiers):
    outputs = {}
    encoder.eval()
    for i, feature in enumerate(features):
        classifiers[i].eval()

    with torch.no_grad():
        pre_transforms = config.pre_transforms
        face = pre_transforms(face).to(config.device)
        face = face.unsqueeze(0)
        embedded_face = encoder(face).detach()

        for i, feature in enumerate(features):
            output = classifiers[i](embedded_face)
            preds = functional.softmax(output, dim=-1).cpu().numpy()
            classes = np.arange(0, preds.shape[1])
            predicted_output = (preds * classes).sum(axis=-1)[0]
            outputs[feature] = predicted_output

    return outputs
