import torch
from torch import nn

from src.brain_tumor_ops.configs.model import ModelConfig


class TumorClassifier(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

        self.features = nn.Sequential(
            self._conv_layer(
                model_config.input_channels,
                16,
                conv_kernel_size=self.model_config.conv_kernel_size,
                padding=self.model_config.padding,
            ),
            self._conv_layer(
                16,
                32,
                conv_kernel_size=self.model_config.conv_kernel_size,
                padding=self.model_config.padding,
            ),
            self._conv_layer(
                32,
                64,
                conv_kernel_size=self.model_config.conv_kernel_size,
                padding=self.model_config.padding,
            ),
        )

        self.classification = nn.Sequential(
            nn.Flatten(),
            self._linear_layer(64 * 31 * 31, 128),
            nn.ReLU(),
            self._linear_layer(128, model_config.num_classes),
        )

    @staticmethod
    def _conv_layer(
        input_channels: int,
        output_channels: int,
        conv_kernel_size: int,
        padding: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=conv_kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    @staticmethod
    def _linear_layer(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features=input_channels, out_features=output_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classification(x)
