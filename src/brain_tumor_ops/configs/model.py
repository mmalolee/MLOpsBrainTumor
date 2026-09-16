from dataclasses import dataclass


@dataclass
class ModelConfig:
    num_classes: int = 4
    input_channels: int = 3
    conv_kernel_size: int = 3
    padding: int = conv_kernel_size // 2
