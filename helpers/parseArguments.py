import argparse


def parseArguments(function):
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    if function == 'main':
        parser.add_argument('-G', '--gradient',
                            action='store_true',
                            help='to run on gradient')

    if function == 'train':
        parser.add_argument('-G', '--gradient',
                            action='store_true',
                            help='to run on gradient')
        parser.add_argument('--dataset',
                            metavar='dataset',
                            type=str,
                            choices=['UTKFace', 'AgeDB'],
                            help='select the dataset: %(choices)s',
                            default='UTKFace'
                            )
        parser.add_argument('--feature',
                            metavar='feature',
                            type=str,
                            choices=['age', 'gender'],
                            help='select the feature: %(choices)s',
                            default='gender'
                            )

        parser.add_argument('--splitSize',
                            metavar='splitSize',
                            nargs=3,
                            type=float,
                            help='specify the train, validation and test size',
                            default=[0.7, 0.2, 0.1]
                            )
        parser.add_argument('--batchSize',
                            metavar='batchSize',
                            type=int,
                            help='set the batch size',
                            default=15
                            )
        parser.add_argument('--epochs',
                            metavar='num of epochs',
                            type=int,
                            help='set the number of epochs',
                            default=10
                            )
    args = parser.parse_args()
    return args
