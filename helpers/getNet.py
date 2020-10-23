import sys
from nets.resnet18 import resnet18


def getNet(net):
    if net == 'resnet18':
        print('ResNet18 Found: ')
        net = resnet18()
        print(net)
        return net
    sys.exit('network not found')
