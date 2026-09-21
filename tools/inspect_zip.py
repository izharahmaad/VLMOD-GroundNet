from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile


ZIP_PATH = Path("submissions/VLMOD-GroundNet_submission_result.zip")


def main() -> None:
    with ZipFile(ZIP_PATH) as archive:
        names = archive.namelist()

    files = [
        name for name in names
        if name.endswith(".txt")
    ]

    invalid = [
        name for name in files
        if not name.startswith("result/")
        or name.count("/") != 1
    ]

    print("ZIP:", ZIP_PATH)
    print("TXT files:", len(files))
    print("First entries:", names[:3])
    print("Invalid paths:", len(invalid))

    if len(files) != 300 or invalid:
        raise SystemExit("ZIP validation failed")

    print("ZIP validation passed")


if __name__ == "__main__":
    main()