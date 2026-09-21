from __future__ import annotations

from typing import Iterable

import torch


def binary_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    truth = targets >= 0.5

    true_positive = (predictions & truth).sum().item()
    false_positive = (predictions & ~truth).sum().item()
    false_negative = (~predictions & truth).sum().item()

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = (
        true_positive / precision_denominator
        if precision_denominator
        else 0.0
    )

    recall = (
        true_positive / recall_denominator
        if recall_denominator
        else 0.0
    )

    f1_denominator = precision + recall

    f1 = (
        2.0 * precision * recall / f1_denominator
        if f1_denominator
        else 0.0
    )

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": float(true_positive),
        "fp": float(false_positive),
        "fn": float(false_negative),
    }


def search_threshold(
    probabilities: Iterable[torch.Tensor],
    targets: Iterable[torch.Tensor],
) -> tuple[float, dict[str, float]]:
    probability_values = list(probabilities)
    target_values = list(targets)

    best_threshold = 0.5
    best_metrics = {
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
    }

    for threshold in torch.arange(0.20, 0.81, 0.05):
        threshold_value = float(threshold)

        all_probabilities = torch.cat(probability_values)
        all_targets = torch.cat(target_values)

        metrics = binary_metrics(
            all_probabilities,
            all_targets,
            threshold_value,
        )

        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = threshold_value
            best_metrics = metrics

    return best_threshold, best_metrics