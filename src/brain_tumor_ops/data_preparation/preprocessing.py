from torchvision import transforms

from src.brain_tumor_ops.configs.preprocessing import DataPreprocessingConfig


class DataPreprocessor:
    def __init__(
        self,
        data_preprocessing_config: DataPreprocessingConfig,
    ) -> None:
        self.data_preprocessing_config = data_preprocessing_config

    @staticmethod
    def _resize(img_size) -> transforms.Resize:
        return transforms.Resize(
            size=(
                img_size,
                img_size,
            )
        )

    @staticmethod
    def _to_tensor() -> transforms.ToTensor:
        return transforms.ToTensor()

    @staticmethod
    def _normalize(mean, std) -> transforms.Normalize:
        return transforms.Normalize(
            mean=mean,
            std=std,
        )

    def build_transforms(self) -> transforms.Compose:
        return transforms.Compose(
            [
                self._resize(self.data_preprocessing_config.img_size),
                self._to_tensor(),
                self._normalize(
                    self.data_preprocessing_config.mean,
                    self.data_preprocessing_config.std,
                ),
            ]
        )
