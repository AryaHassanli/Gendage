from src.helpers.config import config
# noinspection PyUnresolvedReferences
from src.helpers import parseArguments
import json


# TODO: Add preprocess command


def main():
    cli_args = parseArguments.parse()
    cli_args = validate(cli_args)

    config_file = cli_args.config_file
    # TODO: detect if the file is in config or not
    # TODO: Validate inputs
    file_args = {}

    if config_file is not None:
        config_file = config_file['value']
        if config_file is not None:
            with open(config_file) as f:
                file_args = json.load(f)

    config.setup(file_args, cli_args)
    if config.main_function == 'train_classifier':
        from src import train_classifier
        train_classifier.main()
    elif config.main_function == 'run':
        from src import run
        run.main()
    pass


def validate(cli_args):
    return cli_args


if __name__ == '__main__':
    main()
