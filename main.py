from src.brain_tumor_ops.configs.data_loader import DataLoaderConfig
from src.brain_tumor_ops.configs.model import ModelConfig
from src.brain_tumor_ops.configs.paths import PathsConfig
from src.brain_tumor_ops.configs.preprocessing import DataPreprocessingConfig
from src.brain_tumor_ops.configs.training import TrainingConfig
from src.brain_tumor_ops.pipelines.training_pipeline import TrainingPipeline


def main() -> None:
    paths_config = PathsConfig()
    data_preprocessing_config = DataPreprocessingConfig()
    data_loader_config = DataLoaderConfig()
    model_config = ModelConfig()
    training_config = TrainingConfig()

    training_pipeline = TrainingPipeline(
        paths_config,
        data_preprocessing_config,
        data_loader_config,
        model_config,
        training_config,
    )

    training_pipeline.run()


if __name__ == "__main__":
    main()
