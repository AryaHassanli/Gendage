# noinspection PyUnresolvedReferences
from dataLoaders import *


def getDatasetHandler(dataset):
    dataObj = eval(dataset+"."+dataset+'Handler()')
    return dataObj
