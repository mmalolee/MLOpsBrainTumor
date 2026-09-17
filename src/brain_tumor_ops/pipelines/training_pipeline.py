from src.brain_tumor_ops.configs.data_loader import DataLoaderConfig
from src.brain_tumor_ops.configs.model import ModelConfig
from src.brain_tumor_ops.configs.paths import PathsConfig
from src.brain_tumor_ops.configs.preprocessing import DataPreprocessingConfig
from src.brain_tumor_ops.configs.training import TrainingConfig


class TrainingPipeline:
    def __init__(
        self,
        paths_config: PathsConfig,
        data_preprocessing_config: DataPreprocessingConfig,
        data_loader_config: DataLoaderConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ):
        pass
