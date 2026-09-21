from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.datasets.object_features import object_to_feature
from src.datasets.object_parser import parse_object
from src.datasets.parse_annotations import object_key


class VLMODTrainDataset(Dataset):
    def __init__(self, annotation_dir: str | Path):
        self.annotation_dir = Path(annotation_dir)
        self.files = sorted(self.annotation_dir.glob("*.json"))

        if not self.files:
            raise FileNotFoundError(
                f"No JSON files found in {self.annotation_dir}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        annotations = sorted(
            data[0],
            key=lambda record: int(record["ann_id"]),
        )[:3]

        object_map: dict[tuple[str, ...], list[Any]] = {}

        for record in annotations:
            for raw_object in record.get("label_3", []):
                obj = parse_object(raw_object)
                object_map[object_key(obj)] = obj

        objects = list(object_map.values())

        features = [
            object_to_feature(obj)
            for obj in objects
        ]

        targets = []

        for obj in objects:
            key = object_key(obj)
            row = []

            for record in annotations:
                positive = {
                    object_key(parse_object(raw_object))
                    for raw_object in record.get("label_3", [])
                }
                row.append(float(key in positive))

            targets.append(row)

        return {
            "file_name": path.name,
            "descriptions": [
                record["public_description"]
                for record in annotations
            ],
            "object_features": torch.tensor(
                features,
                dtype=torch.float32,
            ),
            "targets": torch.tensor(
                targets,
                dtype=torch.float32,
            ),
        }


class VLMODTestDataset(Dataset):
    def __init__(self, annotation_dir: str | Path):
        self.annotation_dir = Path(annotation_dir)
        self.files = sorted(self.annotation_dir.glob("*.json"))

        if not self.files:
            raise FileNotFoundError(
                f"No JSON files found in {self.annotation_dir}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        objects = [
            parse_object(raw_object)
            for raw_object in data["test_data"]
        ]

        return {
            "file_name": path.name,
            "descriptions": data["public_description"],
            "object_features": torch.tensor(
                [object_to_feature(obj) for obj in objects],
                dtype=torch.float32,
            ),
            "objects": objects,
        }
