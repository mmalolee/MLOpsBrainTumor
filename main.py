import argparse
from pathlib import Path

from src.brain_tumor_ops.configs.data_loader import DataLoaderConfig
from src.brain_tumor_ops.configs.model import ModelConfig
from src.brain_tumor_ops.configs.paths import PathsConfig
from src.brain_tumor_ops.configs.prediction import PredictionConfig
from src.brain_tumor_ops.configs.preprocessing import DataPreprocessingConfig
from src.brain_tumor_ops.configs.training import TrainingConfig
from src.brain_tumor_ops.pipelines.prediction_pipeline import PredictionPipeline
from src.brain_tumor_ops.pipelines.training_pipeline import TrainingPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("train")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--image", type=Path, required=True)

    args = parser.parse_args()

    if args.mode == "train":
        training_pipeline = TrainingPipeline(
            PathsConfig(),
            DataPreprocessingConfig(),
            DataLoaderConfig(),
            ModelConfig(),
            TrainingConfig(),
        )

        training_pipeline.run()

    elif args.mode == "predict":
        prediction_pipeline = PredictionPipeline(
            PredictionConfig(), PathsConfig(), ModelConfig(), DataPreprocessingConfig()
        )

        print(prediction_pipeline.run(args.image))


if __name__ == "__main__":
    main()
