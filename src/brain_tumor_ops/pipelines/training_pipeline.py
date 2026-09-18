from torch import nn, optim

from src.brain_tumor_ops.architectures.cnn import TumorClassifier
from src.brain_tumor_ops.configs.data_loader import DataLoaderConfig
from src.brain_tumor_ops.configs.model import ModelConfig
from src.brain_tumor_ops.configs.paths import PathsConfig
from src.brain_tumor_ops.configs.preprocessing import DataPreprocessingConfig
from src.brain_tumor_ops.configs.training import TrainingConfig
from src.brain_tumor_ops.data_preparation.data_creator import DataCreator
from src.brain_tumor_ops.data_preparation.data_loader import DataLoaderFactory
from src.brain_tumor_ops.data_preparation.data_preprocessing import DataPreprocessor
from src.brain_tumor_ops.training.trainer import Trainer


class TrainingPipeline:
    def __init__(
        self,
        paths_config: PathsConfig,
        data_preprocessing_config: DataPreprocessingConfig,
        data_loader_config: DataLoaderConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ) -> None:
        self.paths_config = paths_config
        self.data_loader_config = data_loader_config
        self.model_config = model_config
        self.training_config = training_config
        self.data_preprocessing_config = data_preprocessing_config

    def run(self) -> None:

        data_preprocessor = DataPreprocessor(self.data_preprocessing_config)
        data_transformer = data_preprocessor.build_transforms()

        data_creator = DataCreator(data_transformer)
        training_dataset = data_creator.get_training_dataset(
            self.paths_config.source_raw_training_data_dir
        )

        training_data_loader_factory = DataLoaderFactory(
            self.data_loader_config, training_dataset
        )
        training_data_loader = training_data_loader_factory.get_training_data_loader()

        model = TumorClassifier(self.model_config)
        model.to(self.training_config.device)

        optimizer = optim.Adam(
            model.parameters(), lr=self.training_config.learning_rate
        )
        cost_function = nn.CrossEntropyLoss()

        trainer = Trainer(self.training_config)

        trainer.fit(
            training_data_loader,
            model,
            optimizer,
            cost_function,
            self.paths_config.mvp_model_dir,
        )
