from pathlib import Path

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose


class DataCreator:
    def __init__(
        self, training_data_path: Path, test_data_path: Path, transformer: Compose
    ) -> None:
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path
        self.transformer = transformer

    def get_training_dataset(self, training_data_path, transformer) -> ImageFolder:
        return ImageFolder(root=training_data_path, transform=transformer)

    def get_test_dataset(self, test_data_path, transformer) -> ImageFolder:
        return ImageFolder(root=test_data_path, transform=transformer)
