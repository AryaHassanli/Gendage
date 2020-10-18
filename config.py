import os

import torch


class Config:
    def __init__(self):
        self.baseDir = os.path.dirname(os.path.realpath(__file__))

        self.datasetDir = "datasets"
        self.absDatasetDir = self.datasetDir if os.path.isabs(self.datasetDir) else os.path.join(self.baseDir,
                                                                                                 self.datasetDir)
        self.outputDir = "output"
        self.absOutputDir = self.outputDir if os.path.isabs(self.outputDir) else os.path.join(self.baseDir,
                                                                                              self.outputDir)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        pass

    def set(self,
            baseDir=None,
            datasetDir=None,
            outputDir=None,
            device=None
            ):
        self.baseDir = baseDir if baseDir is not None else self.baseDir

        self.datasetDir = datasetDir if datasetDir is not None else self.datasetDir
        self.absDatasetDir = self.datasetDir if os.path.isabs(self.datasetDir) else os.path.join(self.baseDir,
                                                                                                 self.datasetDir)
        self.outputDir = outputDir if outputDir is not None else self.outputDir
        self.absOutputDir = self.outputDir if os.path.isabs(self.outputDir) else os.path.join(self.baseDir,
                                                                                              self.outputDir)

        self.device = device if device is not None else self.device


config = Config()
