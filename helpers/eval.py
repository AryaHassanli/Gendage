import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1

from config import config

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(config.device)


def encoderMultiTask(face, ageModel, genderModel):
    ageModel.eval()
    genderModel.eval()
    with torch.no_grad():
        preTransforms = config.preTransforms
        face = preTransforms(face).to(config.device)
        face = face.unsqueeze(0)
        face = resnet(face).detach()

        ageOutputs = ageModel(face)
        genderOutputs = genderModel(face)

        preds = F.softmax(ageOutputs, dim=-1).cpu().numpy()
        classes = np.arange(0, preds.shape[1])
        agePred = (preds * classes).sum(axis=-1)[0]

        preds = F.softmax(genderOutputs, dim=-1).cpu().numpy()
        classes = np.arange(0, preds.shape[1])
        genderPred = (preds * classes).sum(axis=-1)[0]
    return {"age": agePred, "gender": genderPred}
