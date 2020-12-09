# noinspection PyUnresolvedReferences
from dataLoaders import *


def get(dataset):
    dataObj = eval(dataset+"."+dataset+'Handler()')
    return dataObj
