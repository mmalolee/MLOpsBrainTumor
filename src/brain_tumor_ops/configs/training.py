from dataclasses import dataclass

import torch


@dataclass
class TrainingConfig:
    epochs: int = 5
    learning_rate: float = 0.001
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
