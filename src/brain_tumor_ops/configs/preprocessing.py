from dataclasses import dataclass


@dataclass
class DataPreprocessingConfig:
    img_size: int = 250
    mean: tuple[float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float] = (0.5, 0.5, 0.5)
