from __future__ import annotations

import math
from typing import Any


CLASS_NAMES = {
    "car": 0,
    "van": 1,
    "truck": 2,
    "bus": 3,
    "pedestrian": 4,
    "cyclist": 5,
}

COLOR_NAMES = {
    "black": 0,
    "white": 1,
    "red": 2,
    "silver-grey": 3,
    "dark-brown": 4,
    "blue": 5,
    "yellow": 6,
}

NUM_CLASSES = len(CLASS_NAMES) + 1
NUM_COLORS = len(COLOR_NAMES) + 1


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def object_to_feature(obj: list[Any]) -> list[float]:
    if len(obj) < 16:
        raise ValueError(f"Unexpected object length: {len(obj)}")

    class_name = str(obj[0]).lower()
    color_name = str(obj[-1]).lower()

    class_id = CLASS_NAMES.get(class_name, len(CLASS_NAMES))
    color_id = COLOR_NAMES.get(color_name, len(COLOR_NAMES))

    numeric = [safe_float(value) for value in obj[1:-1]]

    if len(numeric) != 14:
        raise ValueError(
            f"Expected 14 numeric object fields, got {len(numeric)}"
        )

    x1, y1, x2, y2 = numeric[3:7]

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height

    normalized_numeric = [
        numeric[0] / 3.0,
        numeric[1] / 2.0,
        numeric[2] / 6.0,
        numeric[3] / 1920.0,
        numeric[4] / 1080.0,
        numeric[5] / 1920.0,
        numeric[6] / 1080.0,
        numeric[7] / 5.0,
        numeric[8] / 5.0,
        numeric[9] / 10.0,
        numeric[10] / 50.0,
        numeric[11] / 50.0,
        numeric[12] / 100.0,
        numeric[13] / 6.0,
    ]

    return [
        float(class_id) / NUM_CLASSES,
        float(color_id) / NUM_COLORS,
        *normalized_numeric,
        width / 1920.0,
        height / 1080.0,
        area / (1920.0 * 1080.0),
        math.atan2(height, max(width, 1.0)) / math.pi,
    ]