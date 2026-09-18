from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class PathsConfig:
    ROOT_DIR = Path(__file__).resolve().parents[3]

    @property
    def source_raw_data_dir(self) -> Path:
        return self.ROOT_DIR / "data" / "raw"

    @property
    def source_raw_training_data_dir(self) -> Path:
        return self.source_raw_data_dir / "Training"

    @property
    def source_raw_testing_data_dir(self) -> Path:
        return self.source_raw_data_dir / "Testing"

    @property
    def artifacts_dir(self) -> Path:
        return self.ROOT_DIR / "src" / "brain_tumor_ops" / "artifacts"

    @property
    def model_dir(self) -> Path:
        model_dir_timestamp = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"  # noqa: DTZ005
        return self.artifacts_dir / model_dir_timestamp
