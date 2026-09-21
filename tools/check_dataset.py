from src.datasets.vlmod_dataset import VLMODTrainDataset, VLMODTestDataset

train = VLMODTrainDataset(
    r"C:\Dataset\VLMOD\MonoMulti3D\train"
)

test = VLMODTestDataset(
    r"C:\Dataset\VLMOD\MonoMulti3D\test"
)

train_item = train[0]
test_item = test[0]

print("Train scenes:", len(train))
print("Test scenes:", len(test))
print("Train features:", tuple(train_item["object_features"].shape))
print("Train targets:", tuple(train_item["targets"].shape))
print("Test features:", tuple(test_item["object_features"].shape))
print("Test descriptions:", len(test_item["descriptions"]))
