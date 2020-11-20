import logging
import os
import posixpath
import sys
import time

from config import config
from helpers import remote


class Train:
    def __init__(self):
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.logFile = os.path.join(config.outputDir, 'output.log')
        self.remoteLogFile = os.path.join(config.remoteDir, 'output.log').replace(os.sep, posixpath.sep)
        self.log.addHandler(logging.FileHandler(self.logFile))
        self.log.addHandler(logging.StreamHandler(sys.stdout))
        self.dictionary = {}

        logging.getLogger("paramiko").setLevel(logging.WARNING)
        self.remoteHandler = remote.Remote()

    def uploadLog(self):
        self.remoteHandler.upload(self.logFile, self.remoteLogFile)
        pass

    def addToDict(self, dictionary):
        for key, value in dictionary.items():
            self.dictionary[key] = value

    def environment(self):
        self.log.info(str(config.__dict__).replace(',', ',\n'))
        pass

    def transforms(self, preTransforms, runtimeTrainTransform, validTransform):
        self.log.info('preTransforms:')
        for transform in preTransforms.transforms:
            self.log.info('\t' + str(transform))
        self.log.info('runtimeTrainTransform:')
        for transform in runtimeTrainTransform.transforms:
            self.log.info('\t' + str(transform))
        self.log.info('validTransform:')
        for transform in validTransform.transforms:
            self.log.info('\t' + str(transform))
        pass

    def epochBegin(self, epoch, numOfEpochs):
        self.log.info("Epoch: {}/{}".format(epoch, numOfEpochs))

    def train(self, features, trainLoss, trainAcc):
        self.log.info('\n\tTrain:')
        for feature in features:
            self.log.info('\t\t{} -> Loss: {}, Accuracy: {}%'.format(
                feature,
                round(trainLoss[feature], 4),
                round(100 * trainAcc[feature], 2)))
        pass

    def batch(self, features, batch, numOfBatches, batchLoss, batchAcc):
        for feature in features:
            self.log.info('\tBatch: {}/{}, {} -> loss: {}, accuracy: {}%'.format(
                batch, numOfBatches,
                feature,
                round(batchLoss[feature], 4),
                round(100 * batchAcc[feature], 2)
            ))

    def validate(self, features, validateLoss, validateAcc, validateMAE, isLearned):
        self.log.info('\tValidation:')
        for feature in features:
            self.log.info('\t\t{} -> Loss: {}, Accuracy: {}%, MAE: {}'.format(
                feature,
                round(validateLoss[feature], 4),
                round(100 * validateAcc[feature], 2),
                round(validateMAE[feature], 2)))
        if isLearned:
            self.log.info('\tNetwork Improved. saving the model.')
        pass

    def test(self, features, testLoss, testAcc, testMAE):
        self.log.info('Test:')
        for feature in features:
            self.log.info('\t{} -> Loss: {}, Accuracy: {}%, MAE: {}'.format(
                feature,
                round(testLoss[feature], 4),
                round(100 * testAcc[feature], 2),
                round(testMAE[feature], 2)))


class Demo:
    def __init__(self):
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.logFile = os.path.join(config.outputDir, 'output.log')
        self.log.addHandler(logging.FileHandler(self.logFile))
        self.log.addHandler(logging.StreamHandler(sys.stdout))

        self.__frameCount = 0
        self.__frameStart = 0
        self.__frameTime = 0
        self.__totalFrameTime = 0

        self.__detectStart = 0
        self.__detectTime = 0
        self.__totalDetectTime = 0
        self.__totalDetections = 0

    def environment(self):
        self.log.info(str(config.__dict__).replace(',', ',\n'))
        pass

    def frameBegin(self):
        self.__frameStart = time.time()
        pass

    def frameEnd(self, frameNo):
        self.__frameTime = time.time() - self.__frameStart
        self.__totalFrameTime += self.__frameTime
        self.log.info('Frame {}/{} Processed in {}ms\t'.format(
            frameNo,
            self.__frameCount,
            round(1000 * self.__frameTime, 1)
        ))
        pass

    def detectBegin(self):
        self.__detectStart = time.time()
        pass

    def detectEnd(self, numOfFaces):
        self.__detectTime = time.time() - self.__detectStart
        self.__totalDetectTime += self.__detectTime
        self.__totalDetections += numOfFaces
        pass

    def programBegin(self, frameCount):
        self.__frameCount = int(frameCount)

    def programEnd(self):
        self.log.info("\n")
        self.log.info("Total Frames: {}".format(self.__frameCount))
        self.log.info("Total Frame Process Time: {}ms, Average: {}ms".format(
            round(1000 * self.__totalFrameTime, 1),
            round(1000 * self.__totalFrameTime / self.__frameCount, 1)
        ))
        self.log.info('\n')
        self.log.info("Total Faces detected: {}".format(self.__totalDetections))
        self.log.info("Total Detection Time: {}ms, Average: {}ms".format(
            round(1000 * self.__totalDetectTime, 1),
            round(1000 * self.__totalDetectTime / self.__totalDetections, 1)
        ))
        pass
