import sys
# noinspection PyUnresolvedReferences
from nets import *


def get(netName, **kwargs):
    net = getattr(eval(netName), netName)
    if callable(net):
        print(netName + " Found!")
        net = net(**kwargs)
    else:
        sys.exit('network not found')

    return net
