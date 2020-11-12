import logging
import os
import sys

from config import config


class Train:
    def __init__(self):
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.log.addHandler(logging.FileHandler(os.path.join(config.outputDir, 'output.log')))
        self.log.addHandler(logging.StreamHandler(sys.stdout))

    def environment(self, args, kwargs):
        self.log.info(str(args).replace(',', ',\n'))
        self.log.info(str(kwargs).replace(',', ',\n'))
        self.log.info(str(config.__dict__).replace(',', ',\n'))
        pass

    def transforms(self, preTransforms, runtimeTrainTransform, validTransform):
        self.log.info('preTransforms:')
        for transform in preTransforms.transforms:
            self.log.info('\t'+str(transform))
        self.log.info('runtimeTrainTransform:')
        for transform in runtimeTrainTransform.transforms:
            self.log.info('\t'+str(transform))
        self.log.info('validTransform:')
        for transform in validTransform.transforms:
            self.log.info('\t'+str(transform))
        pass

    def epochBegin(self, epoch, numOfEpochs):
        self.log.info("Epoch: {}/{}".format(epoch, numOfEpochs))

    def train(self,features, trainLoss, trainAcc):
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

    def test(self,features, testLoss, testAcc, testMAE):
        self.log.info('Test:')
        for feature in features:
            self.log.info('\t{} -> Loss: {}, Accuracy: {}%, MAE: {}'.format(
                feature,
                round(testLoss[feature], 4),
                round(100 * testAcc[feature], 2),
                round(testMAE[feature], 2)))


class TrainMulti:
    def __init__(self):
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.log.addHandler(logging.FileHandler(os.path.join(config.outputDir, 'output.log')))
        self.log.addHandler(logging.StreamHandler(sys.stdout))

    def environment(self, args, kwargs):
        self.log.info(str(args).replace(',', ',\n'))
        self.log.info(str(kwargs).replace(',', ',\n'))
        self.log.info(str(config.__dict__).replace(',', ',\n'))
        pass

    def transforms(self, preTransforms, runtimeTrainTransform, validTransform):
        self.log.info('preTransforms:')
        for transform in preTransforms.transforms:
            self.log.info('\t'+str(transform))
        self.log.info('runtimeTrainTransform:')
        for transform in runtimeTrainTransform.transforms:
            self.log.info('\t'+str(transform))
        self.log.info('validTransform:')
        for transform in validTransform.transforms:
            self.log.info('\t'+str(transform))
        pass

    def epochBegin(self, epoch, numOfEpochs):
        self.log.info("Epoch: {}/{}".format(epoch, numOfEpochs))

    def train(self, ageTrainLoss, ageTrainAcc , genderTrainLoss, genderTrainAcc):
        self.log.info('\n\tTrain:')
        self.log.info('\t\tAge -> Loss: {}, Accuracy: {}%'.format(
            round(ageTrainLoss, 4),
            round(100 * ageTrainAcc, 2)))
        self.log.info('\t\tGender -> Loss: {}, Accuracy: {}%'.format(
            round(genderTrainLoss, 4),
            round(100 * genderTrainAcc, 2)))
        pass

    def batch(self, batch, numOfBatches, ageBatchLoss, ageBatchAcc, genderBatchLoss, genderBatchAcc):
        self.log.info('\tBatch: {}/{}, Age -> loss: {}, accuracy: {}%'.format(
            batch, numOfBatches,
            round(ageBatchLoss, 4),
            round(100 * ageBatchLoss, 2)
        ))
        self.log.info('\tBatch: {}/{}, Gender -> loss: {}, accuracy: {}%'.format(
            batch, numOfBatches,
            round(genderBatchLoss, 4),
            round(100 * genderBatchLoss, 2)
        ))

    def validate(self, ageValidateLoss, ageValidateAcc, ageValidateMAE , genderValidateLoss, genderValidateAcc, genderValidateMAE, isLearned):
        self.log.info('\tValidation:')
        self.log.info('\t\tAge -> Loss: {}, Accuracy: {}%, MAE: {}'.format(
            round(ageValidateLoss, 4),
            round(100 * ageValidateAcc, 2),
            round(ageValidateMAE, 2)))
        self.log.info('\t\tGender -> Loss: {}, Accuracy: {}%, MAE: {}'.format(
            round(genderValidateLoss, 4),
            round(100 * genderValidateAcc, 2),
            round(genderValidateMAE, 2)))
        if isLearned:
            self.log.info('\tNetwork Improved. saving the model.')
        pass

    def test(self, ageTestLoss, ageTestAcc, ageTestMAE, genderTestLoss, genderTestAcc, genderTestMAE):
        self.log.info('Test:')
        self.log.info('\tAge -> Loss: {}, Accuracy: {}%, MAE: {}'.format(
            round(ageTestLoss, 4),
            round(100 * ageTestAcc, 2),
            round(ageTestMAE, 2)))
        self.log.info('\tGender -> Loss: {}, Accuracy: {}%, MAE: {}'.format(
            round(genderTestLoss, 4),
            round(100 * genderTestAcc, 2),
            round(genderTestMAE, 2)))
