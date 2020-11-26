import argparse


def parse(function):
    parser = argparse.ArgumentParser(description='', fromfile_prefix_chars='@')
    if function == 'preprocess':
        parser.add_argument('--datasetsDir',
                            metavar='datasetsDir',
                            type=str,
                            help='root directory of datasets'
                            )
        parser.add_argument('--outputDir',
                            metavar='outputDir',
                            type=str,
                            help='directory of output'
                            )
        parser.add_argument('--dataset',
                            metavar='dataset',
                            type=str,
                            choices=['UTKFace', 'AgeDB'],
                            help='select the dataset: %(choices)s'
                            )
    if function == 'demo':
        parser.add_argument('--datasetsDir',
                            metavar='datasetsDir',
                            type=str,
                            help='root directory of datasets',
                            default=demoOptions['datasetsDir']
                            )
        parser.add_argument('--outputDir',
                            metavar='outputDir',
                            type=str,
                            help='directory of output',
                            default=demoOptions['outputDir']
                            )
        pass
    if function == 'train':
        parser.add_argument('--datasetsDir',
                            metavar='datasetsDir',
                            type=str,
                            help='root directory of the datasets. e.g. /home/datasets/ or datasets'
                            )
        parser.add_argument('--outputDir',
                            metavar='outputDir',
                            type=str,
                            help='directory to save the outputs. e.g. /artifacts/output/ or output'
                            )
        parser.add_argument('--features',
                            metavar='features',
                            type=str,
                            choices=['age', 'gender'],
                            help='Select the features to learn: %(choices)s e.g. age gender'
                            )
        parser.add_argument('--datasets',
                            metavar='datasets',
                            type=str,
                            nargs='+',
                            choices=['UTKFace', 'AgeDB'],
                            help='Select on dataset for each task: %(choices)s e.g. UTKFace AgeDB'
                            )
        parser.add_argument('--nets',
                            metavar='nets',
                            type=str,
                            nargs='+',
                            choices=['simClass'],
                            help='Select the network for each learning task. %(choices)s '
                                 'e.g. simClass simClass'
                            )
        parser.add_argument('--numOfClasses',
                            metavar='numOfClasses',
                            type=int,
                            nargs='+',
                            help='Number of classes for each task. e.g. 120 2 (age,gender)'
                            )

        parser.add_argument('--preload',
                            action='store_true',
                            help='Set True to load the whole dataset to memory at beginning'
                            )

        parser.add_argument('--splitSize',
                            metavar='splitSize',
                            nargs=3,
                            type=float,
                            help='Specify the train, validation and test size. e.g. 0.7 0.2 0.1'
                            )
        parser.add_argument('--batchSize',
                            metavar='batchSize',
                            type=int,
                            help='Set the batch size. e.g. 128'
                            )
        parser.add_argument('--epochs',
                            metavar='num of epochs',
                            type=int,
                            help='Set the number of epochs. e.g. 40'
                            )
        parser.add_argument('--lr',
                            metavar='lr',
                            type=float,
                            help='Set the Learning Rate. e.g. 0.001'
                            )

    args = parser.parse_args()
    return args
