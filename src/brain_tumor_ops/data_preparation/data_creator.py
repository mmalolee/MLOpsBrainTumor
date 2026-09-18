from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose


class DataCreator:
    def __init__(self, transformer: Compose) -> None:
        self.transformer = transformer

    def get_training_dataset(self, training_data_path) -> ImageFolder:
        return ImageFolder(root=training_data_path, transform=self.transformer)

    def get_test_dataset(self, test_data_path) -> ImageFolder:
        return ImageFolder(root=test_data_path, transform=self.transformer)
