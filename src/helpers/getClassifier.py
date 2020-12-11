import sys
# noinspection PyUnresolvedReferences
from models.classifier import *


def get(classifier_name, **kwargs):
    classifier = getattr(eval(classifier_name), classifier_name)
    if callable(classifier):
        print(classifier_name + " Classifier is Found!")
        classifier = classifier(**kwargs)
    else:
        sys.exit('classifier not found')

    return classifier
