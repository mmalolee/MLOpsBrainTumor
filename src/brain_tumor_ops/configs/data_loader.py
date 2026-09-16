from dataclasses import dataclass


@dataclass
class DataLoaderConfig:
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
