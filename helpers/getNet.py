import sys
# noinspection PyUnresolvedReferences
from nets import *


def getNet(netName):
    net = getattr(eval(netName), netName)
    if callable(net):
        print(netName + " Found!")
        net = net()
    else:
        sys.exit('network not found')

    return net
