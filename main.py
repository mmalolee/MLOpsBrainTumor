from src.brain_tumor_ops.configs.data_loader import DataLoaderConfig
from src.brain_tumor_ops.configs.model import ModelConfig
from src.brain_tumor_ops.configs.paths import PathsConfig
from src.brain_tumor_ops.configs.preprocessing import DataPreprocessingConfig
from src.brain_tumor_ops.configs.training import TrainingConfig
from src.brain_tumor_ops.pipelines.training_pipeline import TrainingPipeline


def main() -> None:
    training_pipeline = TrainingPipeline(
        PathsConfig(),
        DataPreprocessingConfig(),
        DataLoaderConfig(),
        ModelConfig(),
        TrainingConfig(),
    )

    training_pipeline.run()


if __name__ == "__main__":
    main()
