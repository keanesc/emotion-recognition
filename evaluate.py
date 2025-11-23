from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)


class EmotionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []

        for label_idx, emotion in enumerate(EMOTIONS):
            emotion_dir = self.data_dir / emotion
            if emotion_dir.exists():
                for img_path in emotion_dir.glob("*.jpg"):
                    self.images.append(str(img_path))
                    self.labels.append(label_idx)

        print(f"Loaded {len(self.images)} images from {data_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super(EmotionCNN, self).__init__()

        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.model = models.mobilenet_v3_large(weights=weights)

        # Replace classifier with simpler head to reduce overfitting
        # MobileNetV3-Large has 960 input features to classifier
        self.model.classifier = nn.Sequential(
            nn.Linear(960, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def evaluate_model(model, dataloader, device):
    """Evaluate model and collect predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"acc": 100 * correct / total})

    accuracy = 100 * correct / total
    return all_preds, all_labels, accuracy


def plot_confusion_matrix(labels, preds, save_path="confusion_matrix.png"):
    """Generate and save confusion matrix visualization."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS,
        ax=ax,
    )
    ax.set(xlabel="Predicted Label", ylabel="True Label", title="Confusion Matrix")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_history(
    train_losses, val_losses, train_accs, val_accs, save_path="training_history.png"
):
    """Generate and save training history plots."""
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    sns.lineplot(
        x=range(len(train_losses)), y=train_losses, label="Train Loss", ax=ax1
    )
    sns.lineplot(x=range(len(val_losses)), y=val_losses, label="Val Loss", ax=ax1)
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Training and Validation Loss")

    sns.lineplot(x=range(len(train_accs)), y=train_accs, label="Train Acc", ax=ax2)
    sns.lineplot(x=range(len(val_accs)), y=val_accs, label="Val Acc", ax=ax2)
    ax2.set(
        xlabel="Epoch",
        ylabel="Accuracy (%)",
        title="Training and Validation Accuracy",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training history saved to {save_path}")


def main():
    """Load trained model and generate evaluation reports."""
    BATCH_SIZE = 64
    IMG_SIZE = 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    test_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = EmotionDataset("data/test", transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # Load model
    print("Loading trained model...")
    model = EmotionCNN(num_classes=NUM_CLASSES, pretrained=False).to(device)
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
    print(f"Model validation accuracy: {checkpoint['val_acc']:.2f}%")

    # Evaluate model
    print("\nEvaluating model on test set...")
    test_preds, test_labels, test_acc = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.2f}%")

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=EMOTIONS))

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(test_labels, test_preds)

    # Load and plot training history if available
    try:
        training_history = np.load("training_history.npz")
        plot_training_history(
            training_history["train_losses"],
            training_history["val_losses"],
            training_history["train_accs"],
            training_history["val_accs"],
        )
    except FileNotFoundError:
        print("Training history not found. Skipping training plots.")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
