from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
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

        # Replace classifier with enhanced head
        # MobileNetV3-Large has 960 input features to classifier
        self.model.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(),
            nn.Dropout(0.5),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"}
        )

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100 * correct / total:.2f}%",
                }
            )

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def main():
    torch.manual_seed(42)

    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    IMG_SIZE = 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    print("Loading datasets...")
    train_dataset = EmotionDataset("data/train", transform=train_transform)
    test_dataset = EmotionDataset("data/test", transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    print("Creating model...")
    print("Using MobileNetV3-Large with pretrained weights")
    model = EmotionCNN(num_classes=NUM_CLASSES, pretrained=True).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 50)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(model, test_loader, criterion, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                "best_model.pth",
            )
            print(f"Best model saved with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    # Save training history
    print("\nSaving training history...")
    import numpy as np

    np.savez(
        "training_history.npz",
        train_losses=train_losses,
        val_losses=val_losses,
        train_accs=train_accs,
        val_accs=val_accs,
    )
    print("Training history saved to training_history.npz")

    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("\nTo generate evaluation reports and visualizations, run:")
    print("  python evaluate.py")


if __name__ == "__main__":
    main()
