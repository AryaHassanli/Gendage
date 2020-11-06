import argparse
import json

with open('options/train.json') as json_file:
    trainOptions = json.load(json_file)


def parseArguments(function):
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    if function == 'main':
        parser.add_argument('-G', '--gradient',
                            action='store_true',
                            help='to run on gradient')
    if function == 'preprocess':
        parser.add_argument('-G', '--gradient',
                            action='store_true',
                            help='to run on gradient')
        parser.add_argument('--dataset',
                            metavar='dataset',
                            type=str,
                            choices=['UTKFace', 'AgeDB'],
                            help='select the dataset: %(choices)s',
                            default='AgeDB'
                            )
    if function == 'run':
        parser.add_argument('-G', '--gradient',
                            action='store_true',
                            help='to run on gradient')

        pass
    if function == 'train':
        parser.add_argument('-G', '--gradient',
                            action='store_true',
                            help='to run on gradient',
                            default=trainOptions['gradient']
                            )
        parser.add_argument('--dataset',
                            metavar='dataset',
                            type=str,
                            choices=['UTKFace', 'AgeDB'],
                            help='select the dataset: %(choices)s',
                            default=trainOptions['dataset']
                            )
        parser.add_argument('--net',
                            metavar='net',
                            type=str,
                            choices=['resnet18', 'resnet50', 'mobilenet_v2', 'mobilenet_v3'],
                            help='select the net: %(choices)s',
                            default=trainOptions['net']
                            )
        parser.add_argument('--task',
                            metavar='task',
                            type=str,
                            choices=['classification', 'regression'],
                            help='select the task type: %(choices)s',
                            default=trainOptions['task']
                            )
        parser.add_argument('--feature',
                            metavar='feature',
                            type=str,
                            choices=['age', 'gender'],
                            help='select the feature: %(choices)s',
                            default=trainOptions['feature']
                            )

        parser.add_argument('--splitSize',
                            metavar='splitSize',
                            nargs=3,
                            type=float,
                            help='specify the train, validation and test size',
                            default=trainOptions['splitSize']
                            )
        parser.add_argument('--batchSize',
                            metavar='batchSize',
                            type=int,
                            help='set the batch size',
                            default=trainOptions['batchSize']
                            )
        parser.add_argument('--epochs',
                            metavar='num of epochs',
                            type=int,
                            help='set the number of epochs',
                            default=trainOptions['epochs']
                            )
        parser.add_argument('--tag',
                            metavar='tag',
                            help='add a tag at the end of the file',
                            default=''
                            )

    args = parser.parse_args()
    return args
