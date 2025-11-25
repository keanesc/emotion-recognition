# Facial Expression Recognition

Classifies facial expressions into 7 categories: Happy, Sad, Anger, Surprise, Fear, Disgust, and Neutral.

## Model

MobileNetV2 backbone with a custom classifier (256 neurons, dropout 0.3-0.5). Uses data augmentation and learning rate scheduling.

## Installation

This project uses `pixi` for dependency management. Install dependencies:

```bash
pixi install
```

## Usage

### Training

```bash
pixi run python train.py
```

Saves the best model to `best_model.pth` and generates training plots.

### Prediction

```bash
pixi run python predict.py path/to/image.jpg
```

Show all probabilities:

```bash
pixi run python predict.py path/to/image.jpg --show-all
```

## Hyperparameters

- Batch size: 32
- Epochs: 30
- Learning rate: 0.001
- Image size: 224x224
- Optimizer: Adam

## Data Format

Organize images by emotion in subdirectories:

```
data/
├── train/
│   ├── angry/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── happy/
│   └── ...
└── test/
    ├── angry/
    └── ...
```
