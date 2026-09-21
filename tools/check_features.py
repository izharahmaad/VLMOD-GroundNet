from pathlib import Path

from src.datasets.object_features import object_to_feature
from src.datasets.parse_annotations import parse_train_file

train_dir = Path(r"C:\Dataset\VLMOD\MonoMulti3D\train")
sample = sorted(train_dir.glob("*.json"))[0]

parsed = parse_train_file(sample)
features = [object_to_feature(obj) for obj in parsed["objects"]]

print("Objects:", len(features))
print("Feature dimension:", len(features[0]))
print("First feature:", features[0])
