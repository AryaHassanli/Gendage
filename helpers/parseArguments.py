import argparse
import json

with open('defaults/train.json') as json_file:
    trainOptions = json.load(json_file)
with open('defaults/run.json') as json_file:
    runOptions = json.load(json_file)
with open('defaults/preprocess.json') as json_file:
    preprocessOptions = json.load(json_file)


def parse(function):
    parser = argparse.ArgumentParser(description='', fromfile_prefix_chars='@',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    if function == 'main':
        pass
    if function == 'preprocess':
        parser.add_argument('--datasetsDir',
                            metavar='datasetsDir',
                            type=str,
                            help='root directory of datasets',
                            default=preprocessOptions['datasetsDir']
                            )
        parser.add_argument('--outputDir',
                            metavar='outputDir',
                            type=str,
                            help='directory of output',
                            default=preprocessOptions['outputDir']
                            )
        parser.add_argument('--dataset',
                            metavar='dataset',
                            type=str,
                            choices=['UTKFace', 'AgeDB'],
                            help='select the dataset: %(choices)s',
                            default=preprocessOptions['dataset']
                            )
    if function == 'run':
        parser.add_argument('--datasetsDir',
                            metavar='datasetsDir',
                            type=str,
                            help='root directory of datasets',
                            default=runOptions['datasetsDir']
                            )
        parser.add_argument('--outputDir',
                            metavar='outputDir',
                            type=str,
                            help='directory of output',
                            default=runOptions['outputDir']
                            )
        pass
    if function == 'train':
        parser.add_argument('--datasetsDir',
                            metavar='datasetsDir',
                            type=str,
                            help='root directory of datasets',
                            default=trainOptions['datasetsDir']
                            )
        parser.add_argument('--outputDir',
                            metavar='outputDir',
                            type=str,
                            help='directory of output',
                            default=trainOptions['outputDir']
                            )

        parser.add_argument('--remoteDir',
                            metavar='remoteDir',
                            type=str,
                            help='remote directory',
                            default=trainOptions['remoteDir']
                            )

        parser.add_argument('--dataset',
                            metavar='dataset',
                            type=str,
                            choices=['UTKFace', 'AgeDB'],
                            help='select the dataset: %(choices)s',
                            default=trainOptions['dataset']
                            )
        parser.add_argument('--preload',
                            action='store_true',
                            help='preload the dataset to memory',
                            default=trainOptions['preload']
                            )
        parser.add_argument('--usePreprocessed',
                            action='store_true',
                            help='use the preprocessed variation of the dataset. You have to first run preprocess.py '
                                 'on it',
                            default=trainOptions['usePreprocessed']
                            )
        parser.add_argument('--net',
                            metavar='net',
                            type=str,
                            choices=['resnet18', 'resnet50', 'mobilenet_v2', 'mobilenet_v3', 'resnet18Multi'],
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
        parser.add_argument('--features',
                            metavar='features',
                            type=str,
                            choices=['age', 'gender'],
                            help='select the features: %(choices)s',
                            default=trainOptions['features']
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
        parser.add_argument('--lr',
                            metavar='lr',
                            type=float,
                            help='learning rate',
                            default=trainOptions['lr']
                            )

    args = parser.parse_args()
    return args
