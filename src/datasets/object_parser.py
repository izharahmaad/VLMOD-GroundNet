from __future__ import annotations

import ast
from typing import Any


def parse_object(value: str | list[Any]) -> list[Any]:
    if isinstance(value, list):
        return value

    value = value.strip()

    if value.startswith("[") and value.endswith("]"):
        parsed = ast.literal_eval(value)

        if not isinstance(parsed, list):
            raise ValueError(
                f"Expected serialized list, got {type(parsed).__name__}"
            )

        return parsed

    parts = value.split()

    if len(parts) != 16:
        raise ValueError(
            f"Expected 16 space-separated object fields, got {len(parts)}: {value}"
        )

    parsed: list[Any] = [parts[0]]

    for token in parts[1:-1]:
        parsed.append(float(token))

    parsed.append(parts[-1])

    return parsed
