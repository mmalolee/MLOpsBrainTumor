from pathlib import Path

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

        self.model = TumorClassifier(self.model_config)
        self.model.to(self.prediction_config.device)
        self.preprocessor = DataPreprocessor(self.data_preprocessing_config)
        self.predictor = Predictor(self.prediction_config, self.paths_config)
        self.transformer = self.preprocessor.build_transforms()
        self.predictor.initialize_model_weight(self.model)

    def predict_image(self, image: Image.Image) -> str:
        image_tensor = self.transformer(image.convert("RGB"))
        return self.predictor.predict(self.model, image_tensor)

    def run(self, image_path: Path) -> str:
        with Image.open(image_path) as image:
            return self.predict_image(image)
