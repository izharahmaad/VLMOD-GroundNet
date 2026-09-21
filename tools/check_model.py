import torch

from src.datasets.vlmod_dataset import VLMODTrainDataset
from src.models.vlmod_groundnet import VLMODGroundNet

dataset = VLMODTrainDataset(
    r"C:\Dataset\VLMOD\MonoMulti3D\train"
)

item = dataset[0]

model = VLMODGroundNet()
logits = model(
    item["object_features"],
    item["descriptions"],
)

print("Input shape:", tuple(item["object_features"].shape))
print("Target shape:", tuple(item["targets"].shape))
print("Logit shape:", tuple(logits.shape))
print("Logits:", logits.detach())
