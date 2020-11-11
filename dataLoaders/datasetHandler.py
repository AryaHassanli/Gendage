from torch.utils.data import random_split, DataLoader


class DatasetHandler:
    def createDataset(self, feature, transform, **kwargs):
        pass

    def getLoaders(self, trainSize=0.7, validateSize=0.2, testSize=0.1, batchSize=15, **kwargs):
        if round(trainSize + validateSize + testSize, 1) != 1.0:
            sys.exit("Sum of the percentages should be equal to 1. it's " + str(
                trainSize + validateSize + testSize) + " now!")

        trainLen = int(len(self.dataset) * trainSize)
        validateLen = int(len(self.dataset) * validateSize)
        testLen = len(self.dataset) - trainLen - validateLen

        self.trainDataset, self.validateDataset, self.testDataset = random_split(
            self.dataset, [trainLen, validateLen, testLen])

        trainLoader = DataLoader(self.trainDataset, batch_size=batchSize)
        validateLoader = DataLoader(self.validateDataset, batch_size=batchSize)
        testLoader = DataLoader(self.testDataset, batch_size=batchSize)

        return trainLoader, validateLoader, testLoader

    def __prepareOnDisk(self):
        pass
