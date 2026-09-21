# VLMOD-GroundNet

Independent neural-network baseline for the VLMOD task:

**Understanding Multi-Object World from Monocular View**

VLMOD-GroundNet is an independently implemented and trained method for multi-object 3D visual grounding from monocular RGB scenes.

## Method

The model processes candidate-object records and three public descriptions for each scene.

The pipeline includes:

- Structured object features containing category, appearance, image geometry, 3D dimensions, 3D position, and orientation.
- Normalized numeric object representations.
- A learned recurrent description encoder.
- Gated fusion between object and language representations.
- Three independent grounding logits for every candidate object.
- Scene-level multi-label training with binary cross-entropy with logits.
- Validation-based threshold selection for final binary predictions.

The model outputs an \(N \times 3\) prediction matrix for each test scene, where \(N\) is the number of candidate objects and the three columns correspond to the three public descriptions.

## Repository structure

```text
src/
├── datasets/
│   ├── object_features.py
│   ├── object_parser.py
│   ├── parse_annotations.py
│   └── vlmod_dataset.py
├── metrics/
│   └── multilabel_f1.py
└── models/
    └── vlmod_groundnet.py

tools/
├── train.py
├── predict.py
├── validate_submission.py
└── make_result_zip.py
```

## Dataset

This repository does not include the VLMOD or Rope3D images, annotations, checkpoints, prediction files, or submission archives.

Configure the local dataset paths in the training and prediction scripts:

```text
MonoMulti3D/train
MonoMulti3D/test
img/train
img/test
```

The dataset must be obtained and used according to the official VLMOD and Rope3D terms.

## Training

Create a Python environment and install the required dependencies:

```bash
python -m pip install torch torchvision torchaudio
python -m pip install numpy pillow tqdm pyyaml scikit-learn
```

Run training from the repository root:

```bash
python tools/train.py
```

The best validation checkpoint is written locally under:

```text
runs/checkpoints/
```

Training outputs and checkpoints are excluded from version control.

## Prediction and submission

Generate predictions using the trained checkpoint:

```bash
python tools/predict.py
```

Validate the generated files:

```bash
python tools/validate_submission.py
```

The submission archive must contain:

```text
result/
├── one TXT file for each test JSON
└── ...
```

Each TXT file contains one row per test object and exactly three binary values:

```text
0 0 0
0 1 0
1 0 1
```

The dataset, trained weights, generated predictions, and ZIP archives are intentionally excluded from this repository.

## Current benchmark result

The submitted VLMOD-GroundNet baseline was evaluated with:

```text
F1:        63.3937
Precision: 48.5195
Recall:    91.4192
TP:        2802
FP:        2973
FN:        263
```

These values correspond to the submitted baseline and are included for reproducibility context.

## Relationship to the official project

This repository is an independent implementation for the VLMOD benchmark. It is based on the public task and repository context but does not claim to reproduce the original MonoMulti-3DVG implementation or any other submitted method.

## Citation

If you reference the underlying VLMOD task or paper, cite the original work:

```bibtex
@inproceedings{guo2025beyond,
  title={Beyond Human Perception: Understanding Multi-Object World from Monocular View},
  author={Guo, Keyu and Huang, Yongle and Sun, Keyu and Song, Xiangyu and Feng, Mingtao and Liu, Zedong and Song, Huansheng and Wang, Tiantian and Li, Jianxin and Akhtar, Naveed and others},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={3751--3760},
  year={2025}
}
```

## License

The original dataset and source repository remain subject to their respective licenses and terms. This implementation is provided for academic and research use.