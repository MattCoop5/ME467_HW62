import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

data = np.load("shelf_images.npz")
images = data["images"]  # (900, 64, 64), float32 in [0, 1]
labels = data["labels"]  # (900,), int64 in {0, 1, 2}
class_names = list(data["class_names"])
# Shuffle and split: 70% train, 15% val, 15% test
rng = np.random.default_rng(0)
idx = rng.permutation(len(images))
n_test = int(0.15 * len(images))  # 135
n_val = int(0.15 * len(images))  # 135
n_train = len(images) - n_test - n_val  # 630
X_test = images[idx[:n_test]]
y_test = labels[idx[:n_test]]
X_val = images[idx[n_test : n_test + n_val]]
y_val = labels[idx[n_test : n_test + n_val]]
X_train = images[idx[n_test + n_val :]]
y_train = labels[idx[n_test + n_val :]]
# Convert to PyTorch tensors — add channel dimension: (N, 64, 64) -> (N, 1, 64, 64)
X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_t = torch.tensor(y_test, dtype=torch.long)
train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
print(f"Classes: {class_names}")


class ShelfCNN(nn.Module):
    """Conv(16) -> Pool -> Conv(32) -> Pool -> Conv(64) -> Pool -> FC(128) -> FC(3)."""

    def __init__(self):
        super().__init__()
        # We split the network into two parts following the torchvision
        # convention: `features` contains the convolutional layers (the
        # learned feature extractor) and `classifier` contains the fully
        # connected layers. This separation makes it easy to reuse the
        # feature extractor with a different classifier (transfer
        # learning, Section 6.6.13: Transfer Learning) or to extract intermediate
        # feature maps for visualization.
        self.features = nn.Sequential(
            # Block 1: 1x64x64 -> 16x64x64 -> 16x32x32
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 16x32x32 -> 32x32x32 -> 32x16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: 32x16x16 -> 64x16x16 -> 64x8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 64*8*8 = 4096
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 3 classes, no softmax (handled by loss)
        )

    def forward(self, x):
        x = self.features(x)  # convolutional feature extraction
        x = self.classifier(x)  # classification from extracted features
        return x


class ShelfFC(nn.Module):
    """Flatten -> FC(512) -> FC(256) -> FC(3)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),  # 64*64 = 4096
            nn.Linear(64 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, X_val_t, y_val_t, epochs=30, lr=1e-3):
    """Train and return per-epoch (train_loss, val_loss, val_acc)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses, val_accs = [], [], []
    # Evaluate before any training (epoch 0) so the curve starts
    # near chance level (~33% for 3 classes)
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t).item()
        val_acc = (val_pred.argmax(dim=1) == y_val_t).float().mean().item()
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    # No training loss at epoch 0; use val loss as placeholder
    train_losses.append(val_loss)
    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for X_batch, y_batch in train_loader:
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / n_batches)
        # Evaluate on validation set (not test — test is held out)
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            val_acc = (val_pred.argmax(dim=1) == y_val_t).float().mean().item()
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    return train_losses, val_losses, val_accs


torch.manual_seed(0)
cnn = ShelfCNN()
cnn_train, cnn_val, cnn_val_acc = train_model(cnn, train_loader, X_val_t, y_val_t)
torch.manual_seed(0)
fc = ShelfFC()
fc_train, fc_val, fc_val_acc = train_model(fc, train_loader, X_val_t, y_val_t)
# Parameter counts
cnn_params = sum(p.numel() for p in cnn.parameters())
fc_params = sum(p.numel() for p in fc.parameters())
print(f"CNN:  {cnn_params:,} parameters, val acc = {cnn_val_acc[-1]:.1%}")
print(f"FC:   {fc_params:,} parameters, val acc = {fc_val_acc[-1]:.1%}")

cnn.eval()
fc.eval()
with torch.no_grad():
    cnn_test_acc = (cnn(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()
    fc_test_acc = (fc(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()
print(f"CNN test accuracy: {cnn_test_acc:.1%}")
print(f"FC  test accuracy: {fc_test_acc:.1%}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
# Loss
ax = axes[0]
ax.plot(cnn_train, color="#4c78a8", ls="--", lw=1, label="CNN train")
ax.plot(cnn_val, color="#4c78a8", lw=2, label="CNN val")
ax.plot(fc_train, color="#e45756", ls="--", lw=1, label="FC train")
ax.plot(fc_val, color="#e45756", lw=2, label="FC val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("Loss")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Accuracy
ax = axes[1]
ax.plot(cnn_val_acc, color="#4c78a8", lw=2, label="CNN")
ax.plot(fc_val_acc, color="#e45756", lw=2, label="FC")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Accuracy")
ax.set_title("Accuracy")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
plt.show()

filters = cnn.features[0].weight.data.squeeze().numpy()  # (16, 3, 3)
fig, axes = plt.subplots(2, 8, figsize=(10, 2.8))
for i, ax in enumerate(axes.flat):
    ax.imshow(filters[i], cmap="RdBu_r", vmin=-filters.max(), vmax=filters.max())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"#{i + 1}", fontsize=8)
fig.suptitle("Learned first-layer filters (3×3)", fontsize=12)
fig.tight_layout()
plt.show()

cnn.eval()
fc.eval()
with torch.no_grad():
    cnn_preds = cnn(X_test_t).argmax(dim=1).numpy()
    fc_preds = fc(X_test_t).argmax(dim=1).numpy()
# Show 2 images from each class (6 total)
show_idx = []
for c in range(3):
    class_idx = np.where(y_test == c)[0]
    show_idx.extend(rng.choice(class_idx, 2, replace=False))
fig, axes = plt.subplots(
    3, 6, figsize=(13, 6.5), gridspec_kw={"height_ratios": [0.12, 1, 1]}
)
for col, i in enumerate(show_idx):
    true_name = class_names[y_test[i]]
    # Top row: true label only (no image)
    ax = axes[0, col]
    ax.set_axis_off()
    ax.text(
        0.5,
        0.0,
        true_name,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#333333",
    )
    # Middle row: CNN prediction
    # Bottom row: FC prediction
    for row_offset, (name, preds) in enumerate([("CNN", cnn_preds), ("FC", fc_preds)]):
        ax = axes[row_offset + 1, col]
        ax.imshow(X_test[i], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        pred_name = class_names[preds[i]]
        correct = preds[i] == y_test[i]
        color = "#006d00" if correct else "#cc0000"
        ax.set_title(pred_name, fontsize=10, fontweight="bold", color=color)
        if col == 0:
            ax.set_ylabel(name, fontsize=11, fontweight="bold")
# Column header for the true-label row
axes[0, 0].text(
    -0.35,
    0.0,
    "True:",
    transform=axes[0, 0].transAxes,
    ha="right",
    va="center",
    fontsize=10,
    fontweight="bold",
    color="#333333",
)
fig.subplots_adjust(top=0.93, hspace=0.35)
fig.suptitle("Predictions (green = correct, red = wrong)", fontsize=12)
plt.show()
