import datetime
import os
import posixpath

import pytz
import torch


class Config:
    def __init__(self):
        self.baseDir = os.path.dirname(os.path.realpath(__file__))
        self.datasetsDir = None
        self.absDatasetsDir = None
        self.outputDir = None
        self.absOutputDir = None
        self.remoteDir = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        pass

    def setup(self, args, **kwargs):
        for key in args.__dict__:
            self.__setattr__(key, args.__getattribute__(key))

        outputSubDir = datetime.datetime.now(pytz.utc).strftime("%Y-%m-%d_%H-%M-%S")

        self.outputDir = os.path.join(self.outputDir, outputSubDir)
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        self.absDatasetsDir = self.datasetsDir if os.path.isabs(self.datasetsDir) else os.path.join(self.baseDir,
                                                                                                    self.datasetsDir)
        self.absOutputDir = self.outputDir if os.path.isabs(self.outputDir) else os.path.join(self.baseDir,
                                                                                              self.outputDir)
        self.remoteDir = os.path.join(self.remoteDir, outputSubDir)
        self.remoteDir = self.remoteDir.replace(os.sep, posixpath.sep)
        pass


config = Config()
