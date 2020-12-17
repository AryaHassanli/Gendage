import argparse
import copy
import json


def parse():
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='',
                                     fromfile_prefix_chars='@')

    subparsers = parser.add_subparsers(help='list of commands', dest='main_function')

    parser_train = subparsers.add_parser('train_classifier', help='use for train or test classifiers on a dataset',
                                         )
    add_train_classifier(parser_train)

    parser_annotate = subparsers.add_parser('annotate', help='a help')
    add_annotate(parser_annotate)

    args = parser.parse_args()

    function = args.main_function
    with open("src/defaults/" + str(function) + ".json") as f:
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
        kwargs = {}

        name = details['name']
        default = details['default']
        help_ = details['help']
        if 'metavar' in details:
            kwargs['metavar'] = details['metavar']
        if 'type' in details:
            kwargs['type'] = types[details['type']]
        if 'nargs' in details:
            kwargs['nargs'] = details['nargs']
        if 'choices' in details:
            kwargs['choices'] = details['choices']
        if 'action' in details:
            kwargs['action'] = details['action']
        if 'const' in details:
            kwargs['const'] = details['const']

        parser_train.add_argument('--' + name,
                                  default=None,
                                  help=help_ + (
                                      " (choices: %(choices)s)" if 'choices' in kwargs else "") + ", (default: " + str(
                                      default) + ")",
                                  **kwargs
                                  )
    pass


def add_annotate(parser_annotate):
    # TODO: modify help texts
    # TODO: retrieve actual choices
    parser_annotate.add_argument('--config_file',
                                 metavar='NAME',
                                 type=str,
                                 help='Config file name. e.g. train_config if the config file is train_config.py'
                                 )
    with open("src/defaults/annotate.json") as f:
        defaults = json.load(f)
    types = {
        'int': int,
        'float': float,
        'str': str
    }
    for argument, details in defaults.items():
        kwargs = {}

        name = details['name']
        default = details['default']
        help_ = details['help']
        if 'metavar' in details:
            kwargs['metavar'] = details['metavar']
        if 'type' in details:
            kwargs['type'] = types[details['type']]
        if 'nargs' in details:
            kwargs['nargs'] = details['nargs']
        if 'choices' in details:
            kwargs['choices'] = details['choices']
        if 'action' in details:
            kwargs['action'] = details['action']
        if 'const' in details:
            kwargs['const'] = details['const']

        parser_annotate.add_argument('--' + name,
                                     default=None,
                                     help=help_ + (" (choices: %(choices)s)" if 'choices' in kwargs else "")
                                          + ", (default: " + str(default) + ")",
                                     **kwargs
                                     )
        pass
