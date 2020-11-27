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
        self.sshCredsFile = '.ps_project/ssh.creds'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.device != 'cpu':
            torch.backends.cudnn.benchmark = True
        pass

    def setup(self, cliArgs, fileArgs, **kwargs):
        for key, value in fileArgs.items():
            self.__setattr__(key, value)

        for key, value in cliArgs.__dict__.items():
            if value is not None:
                self.__setattr__(key, value)

        outputSubDir = datetime.datetime.now(pytz.utc).strftime("%Y-%m-%d_%H-%M-%S")

        self.outputDir = os.path.join(self.outputDir, outputSubDir)
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        self.absDatasetsDir = self.datasetsDir if os.path.isabs(self.datasetsDir) else os.path.join(self.baseDir,
                                                                                                    self.datasetsDir)
        self.absOutputDir = self.outputDir if os.path.isabs(self.outputDir) else os.path.join(self.baseDir,
                                                                                              self.outputDir)
        self.remoteDir = os.path.join(self.remoteDir, outputSubDir)\
            .replace(os.sep, posixpath.sep) if 'remoteDir' in cliArgs.__dict__ else None
        pass


config = Config()
