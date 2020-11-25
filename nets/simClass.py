import torch.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self, numOfClasses):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(512, numOfClasses)

    def forward(self, x):
        x = self.fc(x)
        return x


def simClass(**kwargs):
    numOfClasses = kwargs.get('numOfClasses', 1000)
    model = SimpleClassifier(numOfClasses)
    return model
