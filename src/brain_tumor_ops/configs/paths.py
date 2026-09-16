from dataclasses import dataclass
from pathlib import Path


@dataclass
class PathsConfig:
    ROOT_DIR = Path(__file__).resolve().parents[3]

    @property
    def source_raw_data_dir(self):
        return self.ROOT_DIR / "data" / "raw"

    @property
    def source_raw_training_data_dir(self):
        return self.source_raw_data_dir / "Training"

    @property
    def source_raw_testing_data_dir(self):
        return self.source_raw_data_dir / "Testing"
