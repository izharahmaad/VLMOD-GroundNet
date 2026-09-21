from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.datasets.object_parser import parse_object


def object_key(obj: list[Any]) -> tuple[str, ...]:
    return tuple(str(item) for item in obj)


def parse_train_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Unexpected training format: {path}")

    annotations = data[0]

    if not isinstance(annotations, list):
        raise ValueError(f"Expected data[0] to be a list: {path}")

    annotations = sorted(
        annotations,
        key=lambda record: int(record["ann_id"]),
    )

    descriptions = [
        record["public_description"]
        for record in annotations[:3]
    ]

    object_map: dict[tuple[str, ...], list[Any]] = {}

    for record in annotations[:3]:
        for raw_object in record.get("label_3", []):
            obj = parse_object(raw_object)
            object_map[object_key(obj)] = obj

    objects = list(object_map.values())
    targets = []

    for obj in objects:
        key = object_key(obj)
        row = []

        for record in annotations[:3]:
            positive_keys = {
                object_key(parse_object(raw_object))
                for raw_object in record.get("label_3", [])
            }
            row.append(int(key in positive_keys))

        targets.append(row)

    return {
        "image_file_name": annotations[0].get("image_file_name"),
        "descriptions": descriptions,
        "objects": objects,
        "targets": targets,
    }
