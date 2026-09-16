from dataclasses import dataclass


@dataclass
class DataPreprocessingConfig:
    img_size: int = 250
    mean: float = 0.5
    std: float = 0.5
