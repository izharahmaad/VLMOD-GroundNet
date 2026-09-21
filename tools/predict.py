from __future__ import annotations

from pathlib import Path

import torch

from src.datasets.vlmod_dataset import VLMODTestDataset
from src.models.vlmod_groundnet import VLMODGroundNet


TEST_DIR = Path(r"C:\Dataset\VLMOD\MonoMulti3D\test")
CHECKPOINT = Path("runs/checkpoints/vlmod_groundnet_best.pt")
OUTPUT_DIR = Path("runs/predictions")


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model = VLMODGroundNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    threshold = float(checkpoint["threshold"])

    dataset = VLMODTestDataset(TEST_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]

            features = item["object_features"].to(device)
            logits = model(
                features,
                item["descriptions"],
            )

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).to(torch.int64)

            output_path = OUTPUT_DIR / (
                Path(item["file_name"]).stem + ".txt"
            )

            lines = [
                " ".join(str(int(value)) for value in row)
                for row in predictions.cpu().tolist()
            ]

            output_path.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )

    print("Test scenes:", len(dataset))
    print("Prediction files:", len(list(OUTPUT_DIR.glob("*.txt"))))
    print("Threshold:", threshold)
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()