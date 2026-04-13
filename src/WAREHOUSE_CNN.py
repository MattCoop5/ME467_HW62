import numpy as np
from pathlib import Path
import time
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

DATA_PATH = Path(__file__).with_name("shelf_images.npz")
data = np.load(DATA_PATH)
images = data["images"]  # (900, 64, 64), float32 in [0, 1]
labels = data["labels"]  # (900,), int64 in {0, 1, 2}
class_names = list(data["class_names"])  # ["normal", "damaged", "overloaded"]


class ShelfDataset(Dataset):
    def __init__(self, image_array, label_array, transform=None):
        self.images = image_array.astype(np.float32)
        self.labels = label_array.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W), values in [0,1]
        label = int(self.labels[idx])
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).unsqueeze(0)
        return img, label


class WarehouseCNN(nn.Module):
    def __init__(self, num_classes=3, dropout_p=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 64x64 -> 32x32 -> 16x16 -> 8x8 after three pool ops
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc_out(x)
        return x


def make_splits(x, y, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)

    n_train = int(len(y) * train_ratio)
    n_val = int(len(y) * val_ratio)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    return (
        x[train_idx],
        y[train_idx],
        x[val_idx],
        y[val_idx],
        x[test_idx],
        y[test_idx],
    )


def make_dataloaders(batch_size=32):
    x_train, y_train, x_val, y_val, x_test, y_test = make_splits(images, labels)

    # Augmentations are applied ONLY to the training DataLoader.
    train_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),
            transforms.ToPILImage(mode="L"),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_ds = ShelfDataset(x_train, y_train, transform=train_transform)
    val_ds = ShelfDataset(x_val, y_val, transform=eval_transform)
    test_ds = ShelfDataset(x_test, y_test, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    if optimizer is None:
        # model.eval() disables dropout / batch norm updates for validation/test.
        model.eval()
    else:
        # model.train() enables dropout / batch norm updates for training.
        model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(optimizer is not None):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

    return running_loss / total, correct / total


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    patience=15,
    device="cpu",
    use_early_stopping=True,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer=optimizer, device=device
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer=None, device=device
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if use_early_stopping and wait >= patience:
                print(f"Early stopping triggered (patience={patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
        "epochs_ran": len(history["train_loss"]),
    }


def evaluate(model, loader, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    loss, acc = run_epoch(model, loader, criterion, optimizer=None, device=device)
    print(f"Test loss={loss:.4f}, test acc={acc:.3f}")
    return loss, acc


def collect_predictions(model, loader, device="cpu"):
    model.eval()
    all_images, all_true, all_pred = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            preds = logits.argmax(dim=1).cpu()
            all_images.append(xb.cpu())
            all_true.append(yb.cpu())
            all_pred.append(preds)

    images_t = torch.cat(all_images, dim=0)
    y_true = torch.cat(all_true, dim=0).numpy()
    y_pred = torch.cat(all_pred, dim=0).numpy()
    return images_t, y_true, y_pred


def compute_metrics(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    metrics = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        metrics.append((precision, recall, f1))

    acc = (y_true == y_pred).mean()
    return acc, cm, metrics


def plot_confusion_matrix(cm, class_names, show=True):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() * 0.5 else "black",
            )

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def show_prediction_examples(
    images_t, y_true, y_pred, class_names, max_per_group=5, show=True
):
    correct_idx = np.where(y_true == y_pred)[0][:max_per_group]
    wrong_idx = np.where(y_true != y_pred)[0][:max_per_group]

    n_cols = max(len(correct_idx), len(wrong_idx), 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.0 * n_cols, 6))
    axes = np.array(axes).reshape(2, n_cols)

    def draw_row(row_axes, idxs, row_title):
        for ax in row_axes:
            ax.axis("off")
        for col, idx in enumerate(idxs):
            img = images_t[idx, 0].numpy()
            img = np.clip(img * 0.5 + 0.5, 0, 1)  # unnormalize for display
            row_axes[col].imshow(img, cmap="gray", vmin=0, vmax=1)
            row_axes[col].set_title(
                f"pred={class_names[y_pred[idx]]}\ntrue={class_names[y_true[idx]]}",
                fontsize=9,
            )
            row_axes[col].axis("off")
        row_axes[0].set_ylabel(row_title, fontsize=11)

    draw_row(axes[0], correct_idx, "Correct")
    draw_row(axes[1], wrong_idx, "Misclassified")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def evaluate_detailed(model, loader, class_names, device="cpu"):
    images_t, y_true, y_pred = collect_predictions(model, loader, device=device)
    acc, cm, metrics = compute_metrics(y_true, y_pred, n_classes=len(class_names))

    print(f"\nHeld-out test accuracy: {acc:.4f}")
    print("Per-class metrics:")
    for i, (precision, recall, f1) in enumerate(metrics):
        print(
            f"  {class_names[i]:>10s} | "
            f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}"
        )

    plot_confusion_matrix(cm, class_names, show=True)
    show_prediction_examples(
        images_t,
        y_true,
        y_pred,
        class_names=class_names,
        max_per_group=5,
        show=True,
    )

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "metrics": metrics,
        "images_t": images_t,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def visualize_first_layer_filters(model, max_cols=8, show=True):
    weights = model.conv1.weight.detach().cpu().numpy()  # (16, 1, 3, 3)
    n_filters = weights.shape[0]
    n_cols = min(max_cols, n_filters)
    n_rows = int(np.ceil(n_filters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.8 * n_cols, 1.8 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i in range(n_rows * n_cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i < n_filters:
            filt = weights[i, 0]
            ax.imshow(filt, cmap="gray")
            ax.set_title(f"F{i}", fontsize=9)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def summarize_filter_patterns(model):
    weights = model.conv1.weight.detach().cpu().numpy()[:, 0, :, :]
    edge_like = 0
    texture_like = 0
    other = 0

    for filt in weights:
        gx = np.abs(np.diff(filt, axis=1)).mean()
        gy = np.abs(np.diff(filt, axis=0)).mean()
        grad_energy = gx + gy
        variance = np.var(filt)

        if grad_energy > 0.18:
            edge_like += 1
        elif variance > 0.015:
            texture_like += 1
        else:
            other += 1

    return {
        "edge_like": edge_like,
        "texture_like": texture_like,
        "other": other,
        "total": len(weights),
    }


def add_examples_panel(fig, images_t, y_true, y_pred, class_names, max_per_group=5):
    gs = fig.add_gridspec(
        2, max_per_group, left=0.07, right=0.98, bottom=0.08, top=0.63
    )
    correct_idx = np.where(y_true == y_pred)[0][:max_per_group]
    wrong_idx = np.where(y_true != y_pred)[0][:max_per_group]

    for row in range(2):
        idxs = correct_idx if row == 0 else wrong_idx
        row_title = "Correct" if row == 0 else "Misclassified"
        for col in range(max_per_group):
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")
            if col < len(idxs):
                idx = idxs[col]
                img = images_t[idx, 0].numpy()
                img = np.clip(img * 0.5 + 0.5, 0, 1)
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.set_title(
                    f"P:{class_names[y_pred[idx]]}\nT:{class_names[y_true[idx]]}",
                    fontsize=8,
                )
            if col == 0:
                ax.set_ylabel(row_title, fontsize=10)


def save_pdf_report(
    save_path,
    class_names,
    cnn_no_reg_info,
    cnn_reg_info,
    resnet_info,
    cnn_no_reg_test,
    cnn_reg_test,
    resnet_test,
    detailed_eval,
    cnn_reg_model,
):
    filter_summary = summarize_filter_patterns(cnn_reg_model)
    edge_like = filter_summary["edge_like"]
    texture_like = filter_summary["texture_like"]
    other = filter_summary["other"]
    total = filter_summary["total"]
    edge_pct = 100.0 * edge_like / max(total, 1)
    texture_pct = 100.0 * texture_like / max(total, 1)
    other_pct = 100.0 * other / max(total, 1)

    if edge_like >= max(texture_like, other):
        filter_answer = (
            f"Edge-like filters dominate ({edge_like}/{total}, {edge_pct:.1f}%). "
            f"Texture-like ({texture_like}/{total}, {texture_pct:.1f}%) and "
            f"other ({other}/{total}, {other_pct:.1f}%) filters still appear, "
            "which is expected: early CNN layers typically learn oriented edges first, "
            "then simple textures/blobs."
        )
    elif texture_like >= max(edge_like, other):
        filter_answer = (
            f"Texture-like filters dominate ({texture_like}/{total}, {texture_pct:.1f}%), "
            f"with edge-like ({edge_like}/{total}, {edge_pct:.1f}%) and "
            f"other ({other}/{total}, {other_pct:.1f}%) filters also present. "
            "This suggests the model relies more on local shelf texture patterns than "
            "pure edge geometry."
        )
    else:
        filter_answer = (
            f"Filters are mixed with intermediate/other patterns dominating "
            f"({other}/{total}, {other_pct:.1f}%). The learned bank still includes edge "
            "and texture detectors, but many kernels are hybrid responses."
        )

    acc_cnn = cnn_reg_test["acc"]
    acc_tl = resnet_test["acc"]
    t_cnn = cnn_reg_info["train_time_sec"]
    t_tl = resnet_info["train_time_sec"]
    e_cnn = max(cnn_reg_info["epochs_ran"], 1)
    e_tl = max(resnet_info["epochs_ran"], 1)

    acc_delta = acc_tl - acc_cnn
    time_delta = t_tl - t_cnn
    speedup = t_cnn / t_tl if t_tl > 0 else float("inf")
    cnn_eff = acc_cnn / t_cnn if t_cnn > 0 else 0.0
    tl_eff = acc_tl / t_tl if t_tl > 0 else 0.0

    if acc_delta > 0.01:
        verdict = "Yes—transfer learning helps on this synthetic dataset."
    elif acc_delta < -0.01:
        verdict = "Not in this run—the from-scratch CNN performs better."
    else:
        if time_delta < 0:
            verdict = "Accuracy is similar, but transfer learning is faster, so it still helps practically."
        else:
            verdict = "Accuracy is similar and transfer learning is not faster, so benefit is limited here."

    tl_answer = (
        f"From-scratch CNN (regularized): acc={acc_cnn:.3f}, "
        f"time={t_cnn:.1f}s over {e_cnn} epochs.\n"
        f"Transfer ResNet-18: acc={acc_tl:.3f}, "
        f"time={t_tl:.1f}s over {e_tl} epochs.\n"
        f"Accuracy delta (TL - CNN) = {acc_delta:+.3f}. "
        f"Time delta (TL - CNN) = {time_delta:+.1f}s "
        f"(speed ratio CNN/TL = {speedup:.2f}x).\n"
        f"Accuracy-per-second: CNN={cnn_eff:.4f}, TL={tl_eff:.4f}.\n"
        f"Conclusion: {verdict}"
    )
    filter_answer_wrapped = "\n".join(textwrap.wrap(filter_answer, width=105))
    tl_answer_wrapped = "\n".join(
        textwrap.wrap(tl_answer.replace("\n", " "), width=105)
    )

    with PdfPages(save_path) as pdf:
        # Page 1: Architecture comparison + training/validation curves.
        fig1 = plt.figure(figsize=(8.5, 11))
        fig1.suptitle("Warehouse Classification Report - Page 1", fontsize=14, y=0.98)

        ax_table = fig1.add_axes([0.06, 0.68, 0.88, 0.24])
        ax_table.axis("off")
        rows = [
            [
                "CNN (no reg)",
                f"{cnn_no_reg_info['params']:,}",
                f"{cnn_no_reg_info['epochs_ran']}",
                f"{cnn_no_reg_test['acc']:.3f}",
                f"{cnn_no_reg_info['train_time_sec']:.1f}",
            ],
            [
                "CNN (regularized)",
                f"{cnn_reg_info['params']:,}",
                f"{cnn_reg_info['epochs_ran']}",
                f"{cnn_reg_test['acc']:.3f}",
                f"{cnn_reg_info['train_time_sec']:.1f}",
            ],
            [
                "ResNet-18 (transfer)",
                f"{resnet_info['params']:,}",
                f"{resnet_info['epochs_ran']}",
                f"{resnet_test['acc']:.3f}",
                f"{resnet_info['train_time_sec']:.1f}",
            ],
        ]
        table = ax_table.table(
            cellText=rows,
            colLabels=[
                "Architecture",
                "Trainable Params",
                "Epochs Run",
                "Test Acc",
                "Train Time (s)",
            ],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.1, 1.6)
        ax_table.set_title("Architecture Comparison", fontsize=12, pad=10)

        ax_l = fig1.add_axes([0.08, 0.37, 0.38, 0.24])
        ax_r = fig1.add_axes([0.54, 0.37, 0.38, 0.24])
        h0 = cnn_no_reg_info["history"]
        h1 = cnn_reg_info["history"]
        e0 = np.arange(1, len(h0["train_loss"]) + 1)
        e1 = np.arange(1, len(h1["train_loss"]) + 1)

        ax_l.plot(e0, h0["train_loss"], label="Train loss")
        ax_l.plot(e0, h0["val_loss"], label="Val loss")
        ax_l.set_title("CNN Without Regularization")
        ax_l.set_xlabel("Epoch")
        ax_l.set_ylabel("Loss")
        ax_l.legend()

        ax_r.plot(e1, h1["train_loss"], label="Train loss")
        ax_r.plot(e1, h1["val_loss"], label="Val loss")
        ax_r.set_title("CNN With Regularization")
        ax_r.set_xlabel("Epoch")
        ax_r.set_ylabel("Loss")
        ax_r.legend()

        ax_l2 = fig1.add_axes([0.08, 0.08, 0.38, 0.22])
        ax_r2 = fig1.add_axes([0.54, 0.08, 0.38, 0.22])
        ax_l2.plot(e0, h0["train_acc"], label="Train acc")
        ax_l2.plot(e0, h0["val_acc"], label="Val acc")
        ax_l2.set_xlabel("Epoch")
        ax_l2.set_ylabel("Accuracy")
        ax_l2.set_ylim(0, 1.02)
        ax_l2.legend()

        ax_r2.plot(e1, h1["train_acc"], label="Train acc")
        ax_r2.plot(e1, h1["val_acc"], label="Val acc")
        ax_r2.set_xlabel("Epoch")
        ax_r2.set_ylabel("Accuracy")
        ax_r2.set_ylim(0, 1.02)
        ax_r2.legend()
        pdf.savefig(fig1)
        plt.close(fig1)

        # Page 2: Confusion matrix + example classifications.
        fig2 = plt.figure(figsize=(8.5, 11))
        fig2.suptitle("Warehouse Classification Report - Page 2", fontsize=14, y=0.98)

        cm = detailed_eval["confusion_matrix"]
        ax_cm = fig2.add_axes([0.18, 0.68, 0.64, 0.24])
        im = ax_cm.imshow(cm, cmap="Blues")
        ax_cm.set_xticks(np.arange(len(class_names)))
        ax_cm.set_yticks(np.arange(len(class_names)))
        ax_cm.set_xticklabels(class_names, rotation=30, ha="right")
        ax_cm.set_yticklabels(class_names)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        ax_cm.set_title("Confusion Matrix (Best Regularized Model)")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                )
        cax = fig2.add_axes([0.84, 0.68, 0.03, 0.24])
        plt.colorbar(im, cax=cax)

        fig2.text(
            0.07,
            0.64,
            "Example classifications (top: correct, bottom: misclassified)",
            fontsize=11,
        )
        add_examples_panel(
            fig2,
            detailed_eval["images_t"],
            detailed_eval["y_true"],
            detailed_eval["y_pred"],
            class_names=class_names,
            max_per_group=5,
        )
        pdf.savefig(fig2)
        plt.close(fig2)

        # Page 3: Filter visualization + comparison plots (visuals only).
        fig3 = plt.figure(figsize=(8.5, 11))
        fig3.suptitle("Warehouse Classification Report - Page 3", fontsize=14, y=0.98)

        weights = cnn_reg_model.conv1.weight.detach().cpu().numpy()[:, 0, :, :]
        n_filters = weights.shape[0]
        n_cols = 8
        n_rows = int(np.ceil(n_filters / n_cols))
        gs3 = fig3.add_gridspec(
            n_rows, n_cols, left=0.08, right=0.92, top=0.80, bottom=0.46
        )
        for i in range(n_rows * n_cols):
            ax = fig3.add_subplot(gs3[i // n_cols, i % n_cols])
            ax.axis("off")
            if i < n_filters:
                ax.imshow(weights[i], cmap="gray")
                ax.set_title(f"F{i}", fontsize=8)
        fig3.text(0.08, 0.82, "First-layer filters (regularized CNN)", fontsize=11)

        ax_bar = fig3.add_axes([0.10, 0.24, 0.35, 0.16])
        names = ["CNN no reg", "CNN reg", "ResNet-18"]
        accs = [cnn_no_reg_test["acc"], cnn_reg_test["acc"], resnet_test["acc"]]
        ax_bar.bar(names, accs, color=["#999999", "#4C78A8", "#72B7B2"])
        ax_bar.set_ylim(0, 1.0)
        ax_bar.set_ylabel("Test accuracy")
        ax_bar.set_title("Accuracy comparison")
        ax_bar.tick_params(axis="x", rotation=20)

        ax_bar2 = fig3.add_axes([0.55, 0.24, 0.35, 0.16])
        times = [
            cnn_no_reg_info["train_time_sec"],
            cnn_reg_info["train_time_sec"],
            resnet_info["train_time_sec"],
        ]
        ax_bar2.bar(names, times, color=["#999999", "#4C78A8", "#72B7B2"])
        ax_bar2.set_ylabel("Training time (s)")
        ax_bar2.set_title("Training time comparison")
        ax_bar2.tick_params(axis="x", rotation=20)
        fig3.text(
            0.08,
            0.16,
            "Detailed written analysis and per-class metrics are on Page 4.",
            fontsize=10,
            style="italic",
        )

        pdf.savefig(fig3)
        plt.close(fig3)

        # Page 4: Written analysis + per-class metrics table.
        fig4 = plt.figure(figsize=(8.5, 11))
        fig4.suptitle("Warehouse Classification Report - Page 4", fontsize=14, y=0.98)

        fig4.text(0.07, 0.91, "Q1: Filter interpretability", fontsize=12, weight="bold")
        fig4.text(0.07, 0.84, filter_answer_wrapped, fontsize=10, va="top")

        fig4.text(
            0.07,
            0.69,
            "Q2: Transfer learning vs from-scratch CNN",
            fontsize=12,
            weight="bold",
        )
        fig4.text(0.07, 0.62, tl_answer_wrapped, fontsize=10, va="top")

        ax_metrics = fig4.add_axes([0.07, 0.20, 0.86, 0.28])
        ax_metrics.axis("off")
        metric_rows = []
        for i, (p, r, f1) in enumerate(detailed_eval["metrics"]):
            metric_rows.append([class_names[i], f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"])
        metric_table = ax_metrics.table(
            cellText=metric_rows,
            colLabels=["Class", "Precision", "Recall", "F1-score"],
            loc="center",
            cellLoc="center",
        )
        metric_table.auto_set_font_size(False)
        metric_table.set_fontsize(10)
        metric_table.scale(1.0, 1.8)
        ax_metrics.set_title("Per-class held-out test metrics", fontsize=12, pad=10)

        fig4.text(
            0.07,
            0.14,
            (
                f"Best regularized model accuracy: {detailed_eval['accuracy']:.3f}. "
                "Use this table alongside the confusion matrix (Page 2) to identify class-level strengths/weaknesses."
            ),
            fontsize=9,
        )

        pdf.savefig(fig4)
        plt.close(fig4)


def build_resnet18_for_grayscale(num_classes=3):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Convert first conv from 3-channel RGB to 1-channel grayscale.
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
    model.conv1 = new_conv

    # Replace classifier head for 3 classes.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Classes: {class_names}")

    train_loader, val_loader, test_loader = make_dataloaders(batch_size=32)

    print("\n=== Training baseline CNN (no regularization) ===")
    cnn_no_reg = WarehouseCNN(num_classes=3, dropout_p=0.0).to(device)
    t0 = time.perf_counter()
    cnn_no_reg_info = train_with_early_stopping(
        cnn_no_reg,
        train_loader,
        val_loader,
        epochs=50,
        lr=1e-3,
        weight_decay=0.0,
        patience=50,
        device=device,
        use_early_stopping=False,
    )
    cnn_no_reg_info["train_time_sec"] = time.perf_counter() - t0
    cnn_no_reg_info["params"] = count_parameters(cnn_no_reg)
    _, cnn_no_reg_acc = evaluate(cnn_no_reg, test_loader, device=device)
    cnn_no_reg_test = {"acc": cnn_no_reg_acc}

    print("\n=== Training custom CNN ===")
    cnn = WarehouseCNN(num_classes=3, dropout_p=0.5).to(device)
    t0 = time.perf_counter()
    cnn_info = train_with_early_stopping(
        cnn,
        train_loader,
        val_loader,
        epochs=100,
        lr=1e-3,
        weight_decay=1e-4,
        patience=15,
        device=device,
    )
    cnn_info["train_time_sec"] = time.perf_counter() - t0
    cnn_info["params"] = count_parameters(cnn)
    _, cnn_acc = evaluate(cnn, test_loader, device=device)
    cnn_test = {"acc": cnn_acc}

    print("\nVisualizing first convolutional layer filters...")
    visualize_first_layer_filters(cnn, show=True)

    print("\n=== Fine-tuning pretrained ResNet-18 ===")
    resnet = build_resnet18_for_grayscale(num_classes=3).to(device)
    t0 = time.perf_counter()
    resnet_info = train_with_early_stopping(
        resnet,
        train_loader,
        val_loader,
        epochs=60,
        lr=1e-4,  # low learning rate for fine-tuning
        weight_decay=1e-4,
        patience=15,
        device=device,
    )
    resnet_info["train_time_sec"] = time.perf_counter() - t0
    resnet_info["params"] = count_parameters(resnet)
    _, resnet_acc = evaluate(resnet, test_loader, device=device)
    resnet_test = {"acc": resnet_acc}

    best_name, best_model, best_info = (
        ("CNN", cnn, cnn_info)
        if cnn_info["best_val_loss"] <= resnet_info["best_val_loss"]
        else ("ResNet-18", resnet, resnet_info)
    )

    print(
        f"\nUsing best regularized model based on validation loss: {best_name} "
        f"(best_val_loss={best_info['best_val_loss']:.4f} at epoch {best_info['best_epoch']})"
    )
    detailed_eval = evaluate_detailed(
        best_model, test_loader, class_names=class_names, device=device
    )

    report_path = Path(__file__).with_name("warehouse_cnn_report.pdf")
    save_pdf_report(
        save_path=report_path,
        class_names=class_names,
        cnn_no_reg_info=cnn_no_reg_info,
        cnn_reg_info=cnn_info,
        resnet_info=resnet_info,
        cnn_no_reg_test=cnn_no_reg_test,
        cnn_reg_test=cnn_test,
        resnet_test=resnet_test,
        detailed_eval=detailed_eval,
        cnn_reg_model=cnn,
    )
    print(f"Saved 3-page report to: {report_path}")


if __name__ == "__main__":
    main()
