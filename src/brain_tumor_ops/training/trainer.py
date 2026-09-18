from datetime import datetime
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.brain_tumor_ops.configs.training import TrainingConfig


class Trainer:
    def __init__(self, training_config: TrainingConfig) -> None:
        self.training_config = training_config

    def _train_one_epoch(
        self,
        training_data_loader: DataLoader,
        model: nn.Module,
        optimizer: optim.Adam,
        cost_function: nn.CrossEntropyLoss,
    ) -> float:
        model.train()
        running_loss = 0.0

        for images, labels in training_data_loader:
            images = images.to(self.training_config.device)
            labels = labels.to(self.training_config.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = cost_function(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        return running_loss / len(training_data_loader)

    @staticmethod
    def _save_model(
        model_dir: Path,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loss: float,
    ) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)

        model_timestamp = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"  # noqa: DTZ005
        model_path = model_dir / model_timestamp

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
            },
            model_path,
        )
        print(f"Zapisano checkpoint: {model_path.resolve()}")

    def fit(
        self,
        training_data_loader: DataLoader,
        model: nn.Sequential,
        optimizer: optim.Adam,
        cost_function: nn.CrossEntropyLoss,
        model_dir: Path,
    ):
        for _ in range(1, self.training_config.epochs + 1):
            avg_loss = self._train_one_epoch(
                training_data_loader, model, optimizer, cost_function
            )

        self._save_model(model_dir, model, optimizer, avg_loss)
