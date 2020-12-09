import torch.nn as nn
import torch

class SimpleClassifier(nn.Module):
    def __init__(self, inputSize, numOfClasses):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(inputSize, numOfClasses)

    def forward(self, x):
        x = self.fc(x)
        return x


def simple(pretrained=None, **kwargs):
    numOfClasses = kwargs.get('numOfClasses', 1000)
    inputSize = kwargs.get('inputSize', 512)
    model = SimpleClassifier(inputSize, numOfClasses)
    if pretrained:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict, strict=True)

    return model
