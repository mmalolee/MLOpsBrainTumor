from typing import Any

import torch
from torch import nn

from src.brain_tumor_ops.configs.inference import InferenceConfig
from src.brain_tumor_ops.configs.paths import PathsConfig


class Predictor:
    def __init__(
        self, inference_config: InferenceConfig, paths_config: PathsConfig
    ) -> None:
        self.inference_config = inference_config
        self.paths_config = paths_config

    def _load_torch_file(self) -> Any:
        return torch.load(self.paths_config.mvp_model_dir, weights_only=True)

    def initialize_model_weight(self, model: nn.Module) -> None:
        model.load_state_dict(self._load_torch_file()["model_state_dict"])

    def predict(self, model: nn.Module, image_tensor: torch.Tensor) -> str:
        model.eval()

        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.inference_config.device)

        classes = self._load_torch_file()["classes"]

        with torch.inference_mode():
            logits = model(image_tensor)
            predicted_class = logits.argmax(dim=1)

        return classes[predicted_class.item()]
