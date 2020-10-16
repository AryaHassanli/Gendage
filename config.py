import os


class Config:
    """
        An object for the main configs

        Attributes:
            baseDir: The base working directory. Should be the directory which main.py is located. 
                Default is Current Directory
            datasetDir: The directory which includes Datasets. Absolute and Relative paths are accepted.
                Default is "Datasets"
            absDataset: The absolute dataset directory generated from baseDir and datasetDir
        
        Methods:

    """

    def __init__(self):
        self.baseDir = os.path.dirname(os.path.realpath(__file__))
        self.datasetDir = "Datasets"
        self.absDatasetDir = os.path.join(self.baseDir, self.datasetDir)
        pass

    def set(self,
            baseDir=None,
            datasetDir=None
            ):
        self.baseDir = baseDir if baseDir is not None else os.path.dirname(os.path.realpath(__file__))
        self.datasetDir = datasetDir if datasetDir is not None else "Datasets"
        self.absDatasetDir = datasetDir if os.path.isabs(self.datasetDir) else os.path.join(self.baseDir,
                                                                                            self.datasetDir)


config = Config()
