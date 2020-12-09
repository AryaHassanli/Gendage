from src.helpers.config import config
# noinspection PyUnresolvedReferences
from src.helpers import parseArguments
import os
import json


def main():
    cli_args = parseArguments.parse()
    config_file = cli_args.config_file

    # TODO: detect if the file is in config or not
    # TODO: Validate inputs

    with open(config_file) as f:
        file_args = json.load(f)

    cli_args = validate(cli_args)
    config.setup(cli_args, file_args)

    if cli_args.main_function == 'train':
        from src import train
        train.main()
    elif cli_args.main_function == 'run':
        from src import run
        run.main()
    pass


def validate(cli_args):
    return cli_args
    pass


if __name__ == '__main__':
    main()
