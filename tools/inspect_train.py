import json
from pathlib import Path

train_dir = Path(r"C:\Dataset\VLMOD\MonoMulti3D\train")
train_files = sorted(train_dir.glob("*.json"))

print(f"Training JSON files: {len(train_files)}")

for path in train_files[:5]:
    print(f"\nFILE: {path.name}")
    data = json.loads(path.read_text(encoding="utf-8"))

    print("Top-level type:", type(data).__name__)

    if isinstance(data, dict):
        print("Keys:", list(data.keys()))

        for key, value in data.items():
            if isinstance(value, list):
                print(f"{key}: list length={len(value)}")
                if value:
                    print("First item:", value[0])
            else:
                print(f"{key}: {type(value).__name__}")

    elif isinstance(data, list):
        print("List length:", len(data))

        if not data:
            continue

        first = data[0]
        print("First item type:", type(first).__name__)

        if isinstance(first, dict):
            print("First item keys:", list(first.keys()))
            for key, value in first.items():
                if isinstance(value, list):
                    print(f"{key}: list length={len(value)}")
                    if value:
                        print("First nested item:", value[0])
                else:
                    print(f"{key}: {type(value).__name__}")
        else:
            print("First item:", first)
