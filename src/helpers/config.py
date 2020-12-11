import os
import sys

import torch
import torchvision.transforms as transforms


class Config:
    def __init__(self):
        self.base_dir = os.path.dirname(sys.argv[0])
        self.datasets_dir = ""
        self.abs_datasets_dir = ""
        self.output_dir = ""
        self.abs_output_dir = ""
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.device != 'cpu':
            torch.backends.cudnn.benchmark = True
        self.pre_transforms = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.Pad(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        pass

    def setup(self, file_args, cli_args):
        for argument, value_set in cli_args.__dict__.items():
            if value_set['value'] is None:
                self.__setattr__(argument, value_set['default'])

        for key, value in file_args.items():
            if value is not None or value != "":
                self.__setattr__(key, value)

        for argument, value_set in cli_args.__dict__.items():
            if value_set['value'] is not None:
                self.__setattr__(argument, value_set['value'])

        if hasattr(self, 'classifier_pretrain'):
            for i, item in enumerate(self.classifier_pretrain):
                if item == "None":
                    self.classifier_pretrain[i] = None

        # output_sub_dir = datetime.datetime.now(pytz.utc).strftime("%Y-%m-%d_%H-%M-%S")
        output_sub_dir = ''
        # TODO: Think about subdir
        self.output_dir = os.path.join(self.output_dir, output_sub_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.abs_datasets_dir = self.datasets_dir if os.path.isabs(self.datasets_dir) \
            else os.path.join(self.base_dir, self.datasets_dir)
        self.abs_output_dir = self.output_dir if os.path.isabs(self.output_dir) else os.path.join(self.base_dir,
                                                                                                  self.output_dir)
        pass


config = Config()
