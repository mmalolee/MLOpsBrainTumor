from dataclasses import dataclass


@dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 0.001
    seed: int = 42
