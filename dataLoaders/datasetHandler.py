class DatasetHandler:
    def createDataset(self, feature, transform, **kwargs):
        pass

    def splitDataset(self, trainSize, validateSize, testSize, **kwargs):
        pass

    def dataLoaders(self, batchSize, **kwargs):
        pass

    def __prepareOnDisk(self):
        pass

