import json
from pathlib import Path

roots = [
    Path(r"C:\Dataset\VLMOD"),
    Path(r"C:\Users\Izhar Ahmad\VLMOD-GroundNet"),
]

json_files = []
for root in roots:
    if root.exists():
        json_files.extend(root.rglob("*.json"))

json_files = sorted(set(json_files))

print(f"Found JSON files: {len(json_files)}")

for path in json_files[:5]:
    print(f"\nFILE: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    print("Keys:", list(data.keys()))

    for key, value in data.items():
        if isinstance(value, list):
            print(f"{key}: list length={len(value)}")
            if value:
                print("First item:", value[0])
        else:
            print(f"{key}: {type(value).__name__}")
