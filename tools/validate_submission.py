from __future__ import annotations

import json
from pathlib import Path


TEST_DIR = Path(r"C:\Dataset\VLMOD\MonoMulti3D\test")
PREDICTION_DIR = Path("runs/predictions")


def main() -> None:
    json_files = sorted(TEST_DIR.glob("*.json"))
    prediction_files = sorted(PREDICTION_DIR.glob("*.txt"))

    expected_names = {
        path.stem
        for path in json_files
    }

    actual_names = {
        path.stem
        for path in prediction_files
    }

    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)

    errors: list[str] = []

    if missing:
        errors.append(
            f"Missing prediction files: {missing[:10]}"
        )

    if extra:
        errors.append(
            f"Unexpected prediction files: {extra[:10]}"
        )

    total_rows = 0
    total_positive = 0

    for json_path in json_files:
        prediction_path = (
            PREDICTION_DIR / f"{json_path.stem}.txt"
        )

        if not prediction_path.exists():
            continue

        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        expected_rows = len(data["test_data"])

        lines = [
            line.strip()
            for line in prediction_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if len(lines) != expected_rows:
            errors.append(
                f"{prediction_path.name}: "
                f"expected {expected_rows} rows, got {len(lines)}"
            )

        for line_number, line in enumerate(lines, start=1):
            values = line.split()

            if len(values) != 3:
                errors.append(
                    f"{prediction_path.name}, line {line_number}: "
                    f"expected 3 values, got {len(values)}"
                )
                continue

            if any(value not in {"0", "1"} for value in values):
                errors.append(
                    f"{prediction_path.name}, line {line_number}: "
                    f"non-binary values: {line}"
                )
                continue

            total_positive += sum(
                int(value)
                for value in values
            )

        total_rows += len(lines)

    print("Expected JSON files:", len(json_files))
    print("Prediction TXT files:", len(prediction_files))
    print("Total prediction rows:", total_rows)
    print("Total positive labels:", total_positive)

    if errors:
        print("VALIDATION FAILED")
        for error in errors[:20]:
            print(error)
        raise SystemExit(1)

    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()