import os

import torch


class Config:
    """
        An object for the main configs

        Attributes:
            baseDir: The base working directory. Should be the directory which main.py is located. 
                Default is Current Directory
            datasetDir: The directory which includes Datasets. Absolute and Relative paths are accepted.
                Default is "datasets"
            absDataset: The absolute dataset directory generated from baseDir and datasetDir
        
        Methods:

    """

    def __init__(self):
        self.baseDir = os.path.dirname(os.path.realpath(__file__))
        self.datasetDir = "datasets"
        self.absDatasetDir = os.path.join(self.baseDir, self.datasetDir)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        pass

    def set(self,
            baseDir=None,
            datasetDir=None,
            device=None
            ):
        self.baseDir = baseDir if baseDir is not None else self.baseDir
        self.datasetDir = datasetDir if datasetDir is not None else self.datasetDir
        self.device = device if device is not None else self.device
        self.absDatasetDir = os.path.join(self.baseDir, self.datasetDir)


config = Config()
