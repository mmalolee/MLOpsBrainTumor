from PIL import Image

from src.brain_tumor_ops.architectures.cnn import TumorClassifier
from src.brain_tumor_ops.configs.model import ModelConfig
from src.brain_tumor_ops.configs.paths import PathsConfig
from src.brain_tumor_ops.configs.prediction import PredictionConfig
from src.brain_tumor_ops.configs.preprocessing import DataPreprocessingConfig
from src.brain_tumor_ops.data_preparation.data_preprocessing import DataPreprocessor
from src.brain_tumor_ops.prediction.predictor import Predictor


class PredictionPipeline:
    def __init__(
        self,
        prediction_config: PredictionConfig,
        paths_config: PathsConfig,
        model_config: ModelConfig,
        data_preprocessing_config: DataPreprocessingConfig,
    ) -> None:
        self.prediction_config = prediction_config
        self.paths_config = paths_config
        self.model_config = model_config
        self.data_preprocessing_config = data_preprocessing_config

    def run(self, image_path) -> None:
        model = TumorClassifier(self.model_config)
        predictor = Predictor(self.prediction_config, self.paths_config)
        predictor.initialize_model_weight(model)
        model.to(self.prediction_config.device)

        preprocessor = DataPreprocessor(self.data_preprocessing_config)
        transformer = preprocessor.build_transforms()

        with Image.open(image_path) as image:
            image_tensor = transformer(image.convert("RGB"))

        return predictor.predict(model, image_tensor)
