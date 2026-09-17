from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.brain_tumor_ops.configs.data_loader import DataLoaderConfig


class DataLoaderFactory:
    def __init__(
        self,
        training_data: ImageFolder,
        test_data: ImageFolder,
        training_config: DataLoaderConfig,
    ) -> None:
        self.training_data = training_data
        self.test_data = test_data
        self.training_config = training_config

    @staticmethod
    def _get_data_loader(
        dataset: ImageFolder,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
        )

    def get_training_data_loader(self) -> DataLoader:
        return self._get_data_loader(
            dataset=self.training_data,
            batch_size=self.training_config.batch_size,
            num_workers=self.training_config.num_workers,
            pin_memory=self.training_config.pin_memory,
            shuffle=self.training_config.training_shuffle,
        )

    def get_test_data_loader(self) -> DataLoader:
        return self._get_data_loader(
            dataset=self.test_data,
            batch_size=self.training_config.batch_size,
            num_workers=self.training_config.num_workers,
            pin_memory=self.training_config.pin_memory,
            shuffle=self.training_config.test_shuffle,
        )
