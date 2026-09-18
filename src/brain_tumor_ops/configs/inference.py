from dataclasses import dataclass

import torch


@dataclass
class InferenceConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
