from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.brain_tumor_ops.configs.training import TrainingConfig


class DataLoaderFactory:
    def __init__(
        self,
        training_data: ImageFolder,
        test_data: ImageFolder,
        training_config: TrainingConfig,
    ) -> None:
        self.training_data = (training_data,)
        self.test_data = test_data
        self.training_config = training_config

    def _get_data_loader(
        self, dataset, batch_size, num_workers, pin_memory, shuffle
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
        )

    def get_training_data_loader(self):
        return self._get_data_loader(
            dataset=self.training_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

    def get_test_data_loader(self):
        return self._get_data_loader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )
