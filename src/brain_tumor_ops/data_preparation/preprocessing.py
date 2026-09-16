from torchvision import transforms

from src.brain_tumor_ops.configs.preprocessing import DataPreprocessingConfig


class DataPreprocessor:
    def __init__(
        self,
        data_preprocessing_config: DataPreprocessingConfig,
    ) -> None:
        self.data_preprocessing_config = data_preprocessing_config

    @staticmethod
    def _grayscale() -> transforms.Grayscale:
        return transforms.Grayscale()

    def _resize(self) -> transforms.Resize:
        return transforms.Resize(
            size=(
                self.data_preprocessing_config.img_size,
                self.data_preprocessing_config.img_size,
            )
        )

    @staticmethod
    def _to_tensor() -> transforms.ToTensor:
        return transforms.ToTensor()

    def _normalize(self) -> transforms.Normalize:
        return transforms.Normalize(
            mean=self.data_preprocessing_config.mean,
            std=self.data_preprocessing_config.std,
        )

    def build_transforms(self) -> transforms.Compose:
        return transforms.Compose(
            [
                self._grayscale(),
                self._resize(),
                self._to_tensor(),
                self._normalize(),
            ]
        )
