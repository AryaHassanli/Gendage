from typing import NoReturn

import torch
from facenet_pytorch import MTCNN, extract_face, fixed_image_standardization

from src.helpers.types import Face


class MTCNNDetector:

    def __init__(self,
                 device: torch.device = torch.device('cpu')):
        self.device = device

        pass

    def __call__(self, frame) -> NoReturn:
        mtcnn = MTCNN(
            keep_all=True,
            min_face_size=100,
            image_size=160,
            margin=14,
            selection_method="center_weighted_size",
            post_process=True,
            device=self.device,
        )
        boxes, probs = mtcnn.detect(frame)
        faces = []
        if boxes is None:
            return faces
        for i, box in enumerate(boxes):
            if probs[i] < 0.93:
                continue
            box = box.astype(int)
            faces.append(Face(
                box=box,
                labels={},
                image_tensor=fixed_image_standardization(extract_face(frame, box))
            ))
        return faces
