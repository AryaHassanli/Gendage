import torch.utils.data
import sys


# TODO: merge to get_loaders and create dataset


class DatasetHandler:
    def create_dataset(self, feature, transform, **kwargs):
        pass

    def get_loaders(self, train_size=0.7, validate_size=0.2, test_size=0.1, batch_size=15, **kwargs):
        if round(train_size + validate_size + test_size, 1) != 1.0:
            sys.exit("Sum of the percentages should be equal to 1. it's " + str(
                train_size + validate_size + test_size) + " now!")

        train_len = int(len(self.dataset) * train_size)
        validate_len = int(len(self.dataset) * validate_size)
        test_len = len(self.dataset) - train_len - validate_len

        self.trainDataset, self.validateDataset, self.testDataset = torch.utils.data.random_split(
            self.dataset, [train_len, validate_len, test_len])

        self.dataset = None

        train_loader = torch.utils.data.DataLoader(self.trainDataset, batch_size=batch_size)
        validate_loader = torch.utils.data.DataLoader(self.validateDataset, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(self.testDataset, batch_size=batch_size)

        return train_loader, validate_loader, test_loader

    def __prepare_on_disk(self):
        pass
