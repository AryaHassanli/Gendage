import torch
from PIL import Image


class Face:
    def __init__(self,
                 box: tuple = None,
                 image: Image = None,
                 image_tensor: torch.Tensor = None,
                 labels: dict = None):
        self.box = box

        self.image = image
        self.image_tensor = image_tensor

        self.labels = labels if labels is not None else {}


class Features:
    def __init__(self,
                 age: bool = True,
                 gender: bool = True,
                 recognition: bool = False):
        self.age: bool = age
        self.gender: bool = gender
        self.recognition: bool = recognition


class TrainParameters:
    def __init__(self,
                 train_size: float = 0.7,
                 validate_size: float = 0.2,
                 test_size: float = 0.1,
                 batch_size: int = 128,
                 lr: float = 0.002,
                 epochs: int = 20):
        self.train_size: float = train_size
        self.validate_size: float = validate_size
        self.test_size: float = test_size
        self.batch_size: int = batch_size
        self.lr: float = lr
        self.epochs: int = epochs


class AnnotateParameters:
    def __init__(self,
                 max_recognize_dist: float = 1.20,
                 min_img_per_idx: int = 25,
                 max_img_per_idx: int = 50,
                 substitution_prob: float = 0.05,

                 process_every: int = 2,
                 clean_every: int = 180
                 ):
        self.max_recognize_dist = max_recognize_dist
        self.min_img_per_idx = min_img_per_idx
        self.max_img_per_idx = max_img_per_idx
        self.substitution_prob = substitution_prob

        self.process_every = process_every
        self.clean_every = clean_every
        pass
