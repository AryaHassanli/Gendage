# noinspection PyUnresolvedReferences
import json
from argparse import Namespace

import torch

from src.helpers import parseArguments
from src.helpers.types import TrainParameters, AnnotateParameters, Features


# TODO: Add preprocess command


def main():
    cli_args = parseArguments.parse()
    cli_args = validate(cli_args)

    config_file = cli_args.config_file
    file_args = {}
    if config_file is not None:
        config_file = config_file['value']
        if config_file is not None:
            with open(config_file) as f:
                file_args = json.load(f)

    args = Namespace()
    for argument, value_set in cli_args.__dict__.items():
        if value_set['value'] is None:
            args.__setattr__(argument, value_set['default'])

    for key, value in file_args.items():
        if value is not None or value != "":
            args.__setattr__(key, value)

    for argument, value_set in cli_args.__dict__.items():
        if value_set['value'] is not None:
            args.__setattr__(argument, value_set['value'])

    # TODO: detect if the file is in config or not
    # TODO: Validate inputs

    if args.main_function == 'train_classifier':
        from src import train_classifier
        train_classifier.main(output_dir=args.output_dir,
                              features=Features(
                                  age=args.age,
                                  gender=args.gender,
                                  recognition=False
                              ),
                              parameters=TrainParameters(
                                  train_size=args.split_size[0],
                                  validate_size=args.split_size[1],
                                  test_size=args.split_size[2],
                                  batch_size=args.batch_size,
                                  lr=args.lr,
                                  epochs=args.epochs
                              ),
                              datasets_dir=args.datasets_dir,
                              dataset=args.dataset,
                              pretrained=args.pretrained,
                              pretrained_encoder=args.pretrained_encoder,
                              preload=args.preload,
                              use_preprocessed = args.use_preprocessed,
                              device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                              )

    elif args.main_function == 'annotate':
        from src.annotate import Annotate
        annotator = Annotate(
            features=Features(
                age=True,
                gender=True,
                recognition=True
            ),
            parameters=AnnotateParameters(
                max_recognize_dist=1.20,
                min_img_per_idx=25,
                max_img_per_idx=50,
                substitution_prob=0.05,

                process_every=2,
                clean_every=180
            ),
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        )
        annotator.annotate(
            input_file=args.input_file,
            save_path=args.save_path
        )
    pass


def validate(cli_args):
    return cli_args


if __name__ == '__main__':
    main()
