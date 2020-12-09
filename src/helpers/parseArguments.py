import argparse


def parse():
    parser = argparse.ArgumentParser(description='', fromfile_prefix_chars='@')

    subparsers = parser.add_subparsers(help='sub-command help', dest='main_function')
    parser_train = subparsers.add_parser('train', help='a help')
    add_train(parser_train)

    parser_run = subparsers.add_parser('run', help='a help')
    add_run(parser_run)

    args = parser.parse_args()
    return args


def add_train(parser_train):
    # TODO: Check match with json file
    # TODO: Net-> Encoder + Classifier
    # TODO: retrieve actual choices
    # TODO: Lowercase datasets, nets
    # TODO: (config file | other params)
    # TODO: Change config file to .json and hard code preTransforms

    parser_train.add_argument('--config_file',
                              metavar='NAME',
                              type=str,
                              help='Config file name. e.g. train_config if the config file is train_config.py',
                              required=True
                              )
    parser_train.add_argument('--datasets_dir',
                              metavar='PATH',
                              type=str,
                              help='PATH to datasets directory. e.g. /home/datasets/ or datasets'
                              )
    parser_train.add_argument('--output_dir',
                              metavar='PATH',
                              type=str,
                              help='PATH to save the outputs. e.g. /artifacts/output/ or output'
                              )
    parser_train.add_argument('--encoder',
                              metavar='ENC',
                              type=str,
                              choices=['mobilenet_v3_small'],
                              help='Select the encoder: %(choices)s e.g. mobilenet_v3_small'
                              )
    parser_train.add_argument('--encoder_pretrain',
                              metavar='MODEL',
                              type=str,
                              choices=['mobilenet_v3_small_1.pt'],
                              help='Select the encoder pretrained model: %(choices)s e.g. mobilenet_v3_small_1.pt'
                              )

    parser_train.add_argument('--features',
                              metavar='FEATURE',
                              nargs='+',
                              type=str,
                              choices=['age', 'gender'],
                              help='Select the features to learn: %(choices)s e.g. age gender'
                              )
    parser_train.add_argument('--datasets',
                              metavar='DS',
                              type=str,
                              nargs='+',
                              choices=['UTKFace', 'AgeDB'],
                              help='Select on dataset for each task: %(choices)s e.g. UTKFace AgeDB'
                              )
    parser_train.add_argument('--classifiers',
                              metavar='NET',
                              type=str,
                              nargs='+',
                              choices=['simClass'],
                              help='Select the network for each learning task. %(choices)s '
                                   'e.g. simClass simClass'
                              )
    parser_train.add_argument('--num_classes',
                              metavar='N',
                              type=int,
                              nargs='+',
                              help='Number of classes for each task. e.g. 120 2 (age,gender)'
                              )

    parser_train.add_argument('--preload',
                              action='store_true',
                              help='Set True to load the whole dataset to memory at beginning',
                              default=None
                              )

    parser_train.add_argument('--split_size',
                              metavar='SIZE',
                              nargs=3,
                              type=float,
                              help='Specify the train, validation and test size. e.g. 0.7 0.2 0.1'
                              )
    parser_train.add_argument('--batch_size',
                              type=int,
                              help='Set the batch size. e.g. 128'
                              )
    parser_train.add_argument('--epochs',
                              metavar='EPOCHS',
                              type=int,
                              help='Set the number of epochs. e.g. 40'
                              )
    parser_train.add_argument('--lr',
                              metavar='LR',
                              type=float,
                              help='Set the Learning Rate. e.g. 0.001'
                              )
    pass


def add_run(parser_run):
    # TODO: Check match with json file
    # TODO: modify help texts
    # TODO: Net-> Encoder + Classifier
    # TODO: retrieve actual choices
    # TODO: Lowercase datasets, nets
    # TODO: (config file | other params)
    # TODO: Change config file to .json and hard code preTransforms

    parser_run.add_argument('--config_file',
                            metavar='NAME',
                            type=str,
                            help='Config file name. e.g. train_config if the config file is train_config.py',
                            required= True
                            )
    parser_run.add_argument('--input_file',
                            metavar='PATH',
                            type=str,
                            help='PATH to input file. e.g. /home/input_samples/image.jpg'
                            )
    parser_run.add_argument('--output_dir',
                            metavar='PATH',
                            type=str,
                            help='PATH to save the outputs. e.g. /artifacts/output/ or output'
                            )
    parser_run.add_argument('--encoder',
                            metavar='ENC',
                            type=str,
                            choices=['mobilenet_v3_small'],
                            help='Select the encoder: %(choices)s e.g. mobilenet_v3_small'
                            )
    parser_run.add_argument('--encoder_pretrain',
                            metavar='MODEL',
                            type=str,
                            help='Select the encoder pretrained model. e.g. models/encoder/mobilenet_v3_small_1.pt'
                            )

    parser_run.add_argument('--features',
                            metavar='FEATURE',
                            nargs='+',
                            type=str,
                            choices=['age', 'gender'],
                            help='Select the features to learn: %(choices)s e.g. age gender'
                            )
    parser_run.add_argument('--classifiers',
                            metavar='NET',
                            type=str,
                            nargs='+',
                            choices=['simClass'],
                            help='Select the network for each learning task. %(choices)s e.g. simClass simClass'
                            )
    parser_run.add_argument('--classifier_pretrain',
                            metavar='MODEL',
                            type=str,
                            help='Select the classifier pretrained model. e.g. models/classifier/gender_model.pt'
                            )

    parser_run.add_argument('--num_classes',
                            metavar='N',
                            type=int,
                            nargs='+',
                            help='Number of classes for each task. e.g. 120 2 (age,gender)'
                            )

    pass

