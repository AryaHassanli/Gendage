import argparse
import json
import copy


def parse():
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='',
                                     fromfile_prefix_chars='@')

    subparsers = parser.add_subparsers(help='list of commands', dest='main_function')
    parser_train = subparsers.add_parser('train_classifier', help='use for train or test classifiers on a dataset',
                                         )
    add_train_classifier(parser_train)

    parser_run = subparsers.add_parser('run', help='a help')
    add_run(parser_run)

    args = parser.parse_args()

    function = args.main_function
    with open("src/defaults/"+str(function)+".json") as f:
        defaults = json.load(f)

    args_new = copy.deepcopy(args)
    for key, value in args.__dict__.items():
        if key == 'main_function' or key == 'config_file':
            args_new.__setattr__(key, {
                'value': value,
                'default': value
            })
            continue
        default = defaults[key]['default']
        args_new.__setattr__(key, {
            'value': value,
            'default': default
        })
    return args_new


def add_train_classifier(parser_train):
    # TODO: retrieve actual choices

    parser_train.add_argument('--config_file',
                              metavar='NAME',
                              type=str,
                              help='Config file name. e.g. train_config if the config file is train_config.py'
                              )
    with open("src/defaults/train_classifier.json") as f:
        defaults = json.load(f)
    types = {
        'int': int,
        'float': float,
        'str': str
    }
    for argument, details in defaults.items():
        name = details['name']
        metavar = details['metavar']
        type_ = types[details['type']]
        nargs = None if details['nargs'] == "None" else details['nargs']
        default = details['default']
        choices = None if details['choices'] == "None" else details['choices']
        help_ = details['help']
        parser_train.add_argument('--' + name,
                                  metavar=metavar,
                                  nargs=nargs,
                                  type=type_,
                                  choices=choices,
                                  default=None,
                                  help=help_ + (" (choices: %(choices)s)" if choices else "") + ", (default: " + str(
                                      default) + ")"
                                  )
    pass


def add_run(parser_run):
    # TODO: modify help texts
    # TODO: retrieve actual choices
    parser_run.add_argument('--config_file',
                            metavar='NAME',
                            type=str,
                            help='Config file name. e.g. train_config if the config file is train_config.py'
                            )
    with open("src/defaults/run.json") as f:
        defaults = json.load(f)
    types = {
        'int': int,
        'float': float,
        'str': str
    }
    for argument, details in defaults.items():
        name = details['name']
        metavar = details['metavar']
        type_ = types[details['type']]
        nargs = None if details['nargs'] == "None" else details['nargs']
        default = details['default']
        choices = None if details['choices'] == "None" else details['choices']
        help_ = details['help']
        parser_run.add_argument('--' + name,
                                metavar=metavar,
                                nargs=nargs,
                                type=type_,
                                choices=choices,
                                default=None,
                                help=help_ + (" (choices: %(choices)s)" if choices else "") + ", (default: " + str(
                                    default) + ")"
                                )
    pass
