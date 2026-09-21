from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


SOURCE_DIR = Path("runs/predictions")
OUTPUT_ZIP = Path("submissions/VLMOD-GroundNet_submission_result.zip")


def main() -> None:
    files = sorted(SOURCE_DIR.glob("*.txt"))

    if len(files) != 300:
        raise ValueError(
            f"Expected 300 TXT files, found {len(files)}"
        )

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(
        OUTPUT_ZIP,
        "w",
        compression=ZIP_DEFLATED,
    ) as archive:
        for path in files:
            archive.write(
                path,
                arcname=f"result/{path.name}",
            )

    print("Created:", OUTPUT_ZIP)
    print("Files:", len(files))
    print("Top-level folder: result/")
    print("Size:", OUTPUT_ZIP.stat().st_size)


if __name__ == "__main__":
    main()