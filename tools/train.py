from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.datasets.vlmod_dataset import VLMODTrainDataset
from src.metrics.multilabel_f1 import binary_metrics, search_threshold
from src.models.vlmod_groundnet import VLMODGroundNet


SEED = 42
EPOCHS = 8
LEARNING_RATE = 1e-3
VALIDATION_RATIO = 0.15

TRAIN_DIR = Path(r"C:\Dataset\VLMOD\MonoMulti3D\train")
CHECKPOINT = Path("runs/checkpoints/vlmod_groundnet_best.pt")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(
    model: nn.Module,
    dataset: VLMODTrainDataset,
    indices: list[int],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, list[torch.Tensor], list[torch.Tensor]]:
    training = optimizer is not None
    model.train(training)

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    probabilities = []
    targets = []

    for index in indices:
        item = dataset[index]

        features = item["object_features"].to(device)
        labels = item["targets"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(features, item["descriptions"])
        loss = criterion(logits, labels)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.item())
        probabilities.append(torch.sigmoid(logits.detach()).cpu())
        targets.append(labels.detach().cpu())

    return total_loss / max(len(indices), 1), probabilities, targets


def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VLMODTrainDataset(TRAIN_DIR)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    validation_size = int(len(indices) * VALIDATION_RATIO)
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]

    model = VLMODGroundNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, _, _ = run_epoch(
            model,
            dataset,
            training_indices,
            optimizer,
            device,
        )

        with torch.no_grad():
            validation_loss, probabilities, targets = run_epoch(
                model,
                dataset,
                validation_indices,
                None,
                device,
            )

        threshold, metrics = search_threshold(
            probabilities,
            targets,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={validation_loss:.4f} | "
            f"threshold={threshold:.2f} | "
            f"F1={metrics['f1']:.4f} | "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "threshold": threshold,
                    "metrics": metrics,
                    "epoch": epoch,
                },
                CHECKPOINT,
            )

    print(f"Best checkpoint: {CHECKPOINT}")
    print(f"Best validation F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()