# %%
#!/usr/bin/env python
# coding: utf-8
"""PyTorch re‑implementation of the original Keras training script for event/noise classification.
Key changes:
* Uses torch, torchvision, and torchmetrics instead of TensorFlow/Keras.
* ResNet‑50 backbone adapted for single‑channel input.
* Manual LR schedule that mirrors the Keras LearningRateScheduler logic.
* Early Stopping & best‑model checkpointing implemented in the training loop.

"""

from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from util.build_model import build_model

# ---------------------------
# Torch Dataset & Model
# ---------------------------


class WaveformDataset(Dataset):
        """Dataset returning (1, H, W) float tensors and scalar labels."""

        def __init__(self, data: np.ndarray, labels: np.ndarray) -> None:
                self.data = data.astype(np.float32)
                self.labels = labels.astype(np.float32)

        def __len__(self) -> int:
                return len(self.data)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
                x = torch.from_numpy(self.data[idx]).unsqueeze(0)
                y = torch.tensor(self.labels[idx])
                return x, y


# ---------------------------
# Training utilities
# ---------------------------


def lr_schedule(epoch: int, base_lr: float = 1e-4) -> float:
        """Learning rate schedule used during training."""
        if epoch > 20:
                return base_lr * 0.1
        if epoch > 10:
                return base_lr * 0.5
        return base_lr


def train(
        model: nn.Module,
        loaders: tuple[DataLoader, DataLoader],
        device: torch.device,
        save_dir: Path,
        epochs: int = 100,
        patience: int = 20,
) -> None:
        """Train the model and save the best checkpoint."""
        train_loader, val_loader = loaders
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr_schedule(0))

        writer = SummaryWriter(log_dir=str(save_dir / 'tb'))

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
                # LR scheduling
                lr = lr_schedule(epoch)
                for pg in optimizer.param_groups:
                        pg['lr'] = lr
                writer.add_scalar('LR', lr, epoch)

                # ---------- Training ----------
                model.train()
                train_loss, train_correct = 0.0, 0
                for x, y in train_loader:
                        x, y = x.to(device), y.to(device).unsqueeze(1)
                        optimizer.zero_grad()
                        logits = model(x)
                        loss = criterion(logits, y)
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item() * x.size(0)
                        preds = torch.sigmoid(logits) >= 0.5
                        train_correct += (preds == y.bool()).sum().item()

                train_loss /= len(train_loader.dataset)
                train_acc = train_correct / len(train_loader.dataset)

                # ---------- Validation ----------
                model.eval()
                val_loss, val_correct = 0.0, 0
                with torch.no_grad():
                        for x, y in val_loader:
                                x, y = x.to(device), y.to(device).unsqueeze(1)
                                logits = model(x)
                                loss = criterion(logits, y)
                                val_loss += loss.item() * x.size(0)
                                preds = torch.sigmoid(logits) >= 0.5
                                val_correct += (preds == y.bool()).sum().item()

                val_loss /= len(val_loader.dataset)
                val_acc = val_correct / len(val_loader.dataset)

                # ---------- Logging ----------
                writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
                writer.add_scalars('Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch)

                print(
                        f'Epoch {epoch + 1:03d}/{epochs} – loss: {train_loss:.4f} – val_loss: {val_loss:.4f} – acc: {train_acc:.4f} – val_acc: {val_acc:.4f}'
                )

                # Early stopping & checkpoint
                if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        torch.save(model.state_dict(), save_dir / 'best_model.pth')
                else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                                print('Early stopping triggered.')
                                break

        writer.close()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
        """Evaluate a model on a data loader."""
        criterion = nn.BCEWithLogitsLoss()
        model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
                for x, y in loader:
                        x, y = x.to(device), y.to(device).unsqueeze(1)
                        logits = model(x)
                        loss += criterion(logits, y).item() * x.size(0)
                        preds = torch.sigmoid(logits) >= 0.5
                        correct += (preds == y.bool()).sum().item()
        loss /= len(loader.dataset)
        acc = correct / len(loader.dataset)
        return loss, acc


# ---------------------------
# Main script
# ---------------------------

# def main():
# Paths
data_dir = '/workspace/data'
event_file = data_dir + '/extract_event_2s.npy'
noise_file = data_dir + '/extract_noise_2s.npy'

event_data = np.load(event_file)
noise_data = np.load(noise_file)

# reverse the order of event new_event first
event_data = event_data[::-1]
noise_data = noise_data[::-1]
#
# scaling (trace normalization)
event_data = (event_data - np.mean(event_data, axis=2, keepdims=True)) / np.std(
        event_data, axis=2, keepdims=True
)
noise_data = (noise_data - np.mean(noise_data, axis=2, keepdims=True)) / np.std(
        noise_data, axis=2, keepdims=True
)

save_dir = Path('/workspace/output/train')

n = 0
test_range = (n, n + 130)  # 10%
valid_range = (n + 130, n + 130 + 193)  # 15%
train_range = (n + 130 + 193, len(event_data))  # 75%

x_train = np.vstack(
        (
                event_data[train_range[0] : train_range[1]],
                noise_data[train_range[0] : train_range[1]],
        )
)
x_test = np.vstack(
        (
                event_data[test_range[0] : test_range[1]],
                noise_data[test_range[0] : test_range[1]],
        )
)
x_valid = np.vstack(
        (
                event_data[valid_range[0] : valid_range[1]],
                noise_data[valid_range[0] : valid_range[1]],
        )
)

y_train = np.concatenate(
        (
                np.ones(len(event_data[train_range[0] : train_range[1]])),
                np.zeros(len(noise_data[train_range[0] : train_range[1]])),
        )
)
y_test = np.concatenate(
        (
                np.ones(len(event_data[test_range[0] : test_range[1]])),
                np.zeros(len(noise_data[test_range[0] : test_range[1]])),
        )
)
y_valid = np.concatenate(
        (
                np.ones(len(event_data[valid_range[0] : valid_range[1]])),
                np.zeros(len(noise_data[valid_range[0] : valid_range[1]])),
        )
)

# Build datasets & loaders
batch_size = 16
train_dataset = WaveformDataset(x_train, y_train)
valid_dataset = WaveformDataset(x_valid, y_valid)
test_dataset = WaveformDataset(x_test, y_test)

train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)
test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

# Model & device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = build_model().to(device)

# Train
train(model, (train_loader, valid_loader), device, save_dir)

# Load best model for evaluation
model.load_state_dict(torch.load(save_dir / 'best_model.pth', map_location=device))

for split_name, loader in [
        ('train', train_loader),
        ('test', test_loader),
        ('valid', valid_loader),
]:
        loss, acc = evaluate(model, loader, device)
        print(f'{split_name.capitalize()} – Loss: {loss:.4f} | Acc: {acc:.4f}')
