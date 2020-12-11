import logging
import os
import sys
import time

from src.helpers.config import config


# TODO: reformat

class Train:
    def __init__(self):
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.logFile = os.path.join(config.output_dir, 'output.log')
        self.log.addHandler(logging.FileHandler(self.logFile))
        self.log.addHandler(logging.StreamHandler(sys.stdout))

        self.dictionary = {}
        self.trainBeginTime = 0

        logging.getLogger("paramiko").setLevel(logging.WARNING)

    def environment(self):
        self.log.info(str(config.__dict__).replace(',', ',\n'))
        pass

    def transforms(self, pre_transforms, runtimeTrainTransform, validTransform):
        self.log.info('preTransforms:')
        for transform in pre_transforms.transforms:
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

    def trainBegin(self):
        self.trainBeginTime = time.time()

    def trainEnd(self, feature, trainLoss, trainAcc):
        self.log.info('\n\tTrain Done in {}ms:'.format(
            round(1000 * (time.time() - self.trainBeginTime), 2)
        ))
        self.log.info('\t\t{} -> Loss: {}, Accuracy: {}%'.format(
            feature,
            round(trainLoss, 4),
            round(100 * trainAcc, 2)))
        pass

    def batch(self, feature, batch, numOfBatches, batchLoss, batchAcc):
        self.log.info('\tBatch: {}/{}, {} -> loss: {}, accuracy: {}%'.format(
            batch, numOfBatches,
            feature,
            round(batchLoss, 4),
            round(100 * batchAcc, 2)
        ))

    def validate(self, feature, validateLoss, validateAcc, validateMAE, isLearned):
        self.log.info('\tValidation:')
        self.log.info('\t\t{} -> Loss: {}, Accuracy: {}%, MAE: {}'.format(
            feature,
            round(validateLoss, 4),
            round(100 * validateAcc, 2),
            round(validateMAE, 2)))
        if isLearned:
            self.log.info('\tNetwork Improved. saving the model.')
        pass

    def test(self, feature, testLoss, testAcc, testMAE):
        self.log.info('Test:')
        self.log.info('\t{} -> Loss: {}, Accuracy: {}%, MAE: {}'.format(
            feature,
            round(testLoss, 4),
            round(100 * testAcc, 2),
            round(testMAE, 2)))


class Demo:
    def __init__(self):
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.logFile = os.path.join(config.output_dir, 'output.log')
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


class Run:
    def __init__(self):
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.logFile = os.path.join(config.output_dir, 'output.log')
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
