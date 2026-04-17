import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


DATA_PATH = Path(__file__).with_name("shelf_images.npz")


class ShelfCNN(nn.Module):
    """Conv(16) -> Pool -> Conv(32) -> Pool -> Conv(64) -> Pool -> FC(128) -> FC(3)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ShelfFC(nn.Module):
    """Flatten -> FC(512) -> FC(256) -> FC(3)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, x_val_t, y_val_t, epochs=30, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses, val_accs = [], [], []

    model.eval()
    with torch.no_grad():
        val_pred = model(x_val_t)
        val_loss = criterion(val_pred, y_val_t).item()
        val_acc = (val_pred.argmax(dim=1) == y_val_t).float().mean().item()
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    train_losses.append(val_loss)

    for _ in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for x_batch, y_batch in train_loader:
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / n_batches)

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            val_acc = (val_pred.argmax(dim=1) == y_val_t).float().mean().item()
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    return train_losses, val_losses, val_accs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


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
    return cm, metrics


def add_prediction_panel(
    fig, x_test, y_test, class_names, cnn_preds, fc_preds, show_idx
):
    axes = fig.subplots(
        3,
        len(show_idx),
        gridspec_kw={"height_ratios": [0.12, 1, 1]},
        squeeze=False,
    )

    for col, i in enumerate(show_idx):
        true_name = class_names[y_test[i]]

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

        for row_offset, (name, preds) in enumerate(
            [("CNN", cnn_preds), ("FC", fc_preds)]
        ):
            ax = axes[row_offset + 1, col]
            ax.imshow(x_test[i], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            pred_name = class_names[preds[i]]
            correct = preds[i] == y_test[i]
            color = "#006d00" if correct else "#cc0000"
            ax.set_title(pred_name, fontsize=10, fontweight="bold", color=color)
            if col == 0:
                ax.set_ylabel(name, fontsize=11, fontweight="bold")

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


def save_pdf_report(
    save_path,
    class_names,
    cnn,
    fc,
    cnn_train,
    cnn_val,
    cnn_val_acc,
    fc_train,
    fc_val,
    fc_val_acc,
    cnn_test_acc,
    fc_test_acc,
    x_test,
    y_test,
    cnn_preds,
    fc_preds,
    show_idx,
):
    cnn_params = count_parameters(cnn)
    fc_params = count_parameters(fc)

    cm_cnn, metrics_cnn = compute_metrics(y_test, cnn_preds, n_classes=len(class_names))
    cm_fc, metrics_fc = compute_metrics(y_test, fc_preds, n_classes=len(class_names))

    filters = cnn.features[0].weight.detach().cpu().squeeze().numpy()
    fmax = float(np.max(np.abs(filters)))

    winner = "CNN" if cnn_test_acc >= fc_test_acc else "FC"
    delta = abs(cnn_test_acc - fc_test_acc)
    summary = (
        f"CNN test accuracy = {cnn_test_acc:.3f}, FC test accuracy = {fc_test_acc:.3f}. "
        f"Best model: {winner} (margin {delta:.3f})."
    )

    with PdfPages(save_path) as pdf:
        fig1 = plt.figure(figsize=(8.5, 11))
        fig1.suptitle("Shelf Image Classification Report - Page 1", fontsize=14, y=0.98)

        ax_table = fig1.add_axes([0.07, 0.73, 0.86, 0.18])
        ax_table.axis("off")
        table = ax_table.table(
            cellText=[
                [
                    "ShelfCNN",
                    f"{cnn_params:,}",
                    f"{cnn_val_acc[-1]:.3f}",
                    f"{cnn_test_acc:.3f}",
                ],
                [
                    "ShelfFC",
                    f"{fc_params:,}",
                    f"{fc_val_acc[-1]:.3f}",
                    f"{fc_test_acc:.3f}",
                ],
            ],
            colLabels=["Model", "Parameters", "Final Val Acc", "Test Acc"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.6)
        ax_table.set_title("Model Comparison", fontsize=12, pad=10)

        ax_loss = fig1.add_axes([0.08, 0.40, 0.84, 0.25])
        ax_loss.plot(cnn_train, color="#4c78a8", ls="--", lw=1, label="CNN train")
        ax_loss.plot(cnn_val, color="#4c78a8", lw=2, label="CNN val")
        ax_loss.plot(fc_train, color="#e45756", ls="--", lw=1, label="FC train")
        ax_loss.plot(fc_val, color="#e45756", lw=2, label="FC val")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Cross-Entropy Loss")
        ax_loss.set_title("Loss Curves")
        ax_loss.legend(fontsize=9)

        ax_acc = fig1.add_axes([0.08, 0.11, 0.84, 0.22])
        ax_acc.plot(cnn_val_acc, color="#4c78a8", lw=2, label="CNN val acc")
        ax_acc.plot(fc_val_acc, color="#e45756", lw=2, label="FC val acc")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Validation Accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.set_title("Validation Accuracy")
        ax_acc.legend(fontsize=9)

        pdf.savefig(fig1)
        plt.close(fig1)

        fig2 = plt.figure(figsize=(8.5, 11))
        fig2.suptitle("Shelf Image Classification Report - Page 2", fontsize=14, y=0.98)

        gs = fig2.add_gridspec(2, 8, left=0.08, right=0.92, top=0.90, bottom=0.55)
        for i in range(16):
            ax = fig2.add_subplot(gs[i // 8, i % 8])
            ax.imshow(filters[i], cmap="RdBu_r", vmin=-fmax, vmax=fmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"#{i + 1}", fontsize=8)
        fig2.text(0.08, 0.92, "Learned first-layer filters (3x3)", fontsize=11)

        ax_cm1 = fig2.add_axes([0.08, 0.12, 0.38, 0.30])
        ax_cm2 = fig2.add_axes([0.54, 0.12, 0.38, 0.30])
        for ax, cm, title in [
            (ax_cm1, cm_cnn, "CNN Confusion Matrix"),
            (ax_cm2, cm_fc, "FC Confusion Matrix"),
        ]:
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names, rotation=30, ha="right")
            ax.set_yticklabels(class_names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(title)
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    ax.text(
                        c,
                        r,
                        str(cm[r, c]),
                        ha="center",
                        va="center",
                        color="white" if cm[r, c] > cm.max() * 0.5 else "black",
                    )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        pdf.savefig(fig2)
        plt.close(fig2)

        fig3 = plt.figure(figsize=(11, 8.5))
        fig3.suptitle("Shelf Image Classification Report - Page 3", fontsize=14, y=0.98)
        add_prediction_panel(
            fig3,
            x_test,
            y_test,
            class_names,
            cnn_preds,
            fc_preds,
            show_idx,
        )
        fig3.subplots_adjust(top=0.90, hspace=0.35)
        fig3.text(
            0.08,
            0.03,
            "Predictions (green = correct, red = wrong). Same format as corrected.py.",
            fontsize=10,
        )
        pdf.savefig(fig3)
        plt.close(fig3)

        fig4 = plt.figure(figsize=(8.5, 11))
        fig4.suptitle("Shelf Image Classification Report - Page 4", fontsize=14, y=0.98)

        fig4.text(0.07, 0.91, "Summary", fontsize=12, weight="bold")
        fig4.text(
            0.07,
            0.84,
            "\n".join(textwrap.wrap(summary, width=95)),
            fontsize=10,
            va="top",
        )

        ax_metrics = fig4.add_axes([0.07, 0.20, 0.86, 0.56])
        ax_metrics.axis("off")
        metric_rows = []
        for i, name in enumerate(class_names):
            p1, r1, f1 = metrics_cnn[i]
            p2, r2, f2 = metrics_fc[i]
            metric_rows.append(
                [
                    name,
                    f"{p1:.3f}",
                    f"{r1:.3f}",
                    f"{f1:.3f}",
                    f"{p2:.3f}",
                    f"{r2:.3f}",
                    f"{f2:.3f}",
                ]
            )

        metric_table = ax_metrics.table(
            cellText=metric_rows,
            colLabels=[
                "Class",
                "CNN Precision",
                "CNN Recall",
                "CNN F1",
                "FC Precision",
                "FC Recall",
                "FC F1",
            ],
            loc="center",
            cellLoc="center",
        )
        metric_table.auto_set_font_size(False)
        metric_table.set_fontsize(10)
        metric_table.scale(1.0, 1.8)
        ax_metrics.set_title("Per-class held-out test metrics", fontsize=12, pad=10)

        pdf.savefig(fig4)
        plt.close(fig4)


def main():
    data = np.load(DATA_PATH)
    images = data["images"]
    labels = data["labels"]
    class_names = list(data["class_names"])

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(images))
    n_test = int(0.15 * len(images))
    n_val = int(0.15 * len(images))

    x_test = images[idx[:n_test]]
    y_test = labels[idx[:n_test]]
    x_val = images[idx[n_test : n_test + n_val]]
    y_val = labels[idx[n_test : n_test + n_val]]
    x_train = images[idx[n_test + n_val :]]
    y_train = labels[idx[n_test + n_val :]]

    x_train_t = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    torch.manual_seed(0)
    cnn = ShelfCNN()
    cnn_train, cnn_val, cnn_val_acc = train_model(cnn, train_loader, x_val_t, y_val_t)

    torch.manual_seed(0)
    fc = ShelfFC()
    fc_train, fc_val, fc_val_acc = train_model(fc, train_loader, x_val_t, y_val_t)

    cnn.eval()
    fc.eval()
    with torch.no_grad():
        cnn_preds = cnn(x_test_t).argmax(dim=1).cpu().numpy()
        fc_preds = fc(x_test_t).argmax(dim=1).cpu().numpy()

    cnn_test_acc = float((cnn_preds == y_test).mean())
    fc_test_acc = float((fc_preds == y_test).mean())

    show_idx = []
    for c in range(3):
        class_idx = np.where(y_test == c)[0]
        show_idx.extend(rng.choice(class_idx, 2, replace=False))

    report_path = Path(__file__).with_name("corrected_report.pdf")
    save_pdf_report(
        save_path=report_path,
        class_names=class_names,
        cnn=cnn,
        fc=fc,
        cnn_train=cnn_train,
        cnn_val=cnn_val,
        cnn_val_acc=cnn_val_acc,
        fc_train=fc_train,
        fc_val=fc_val,
        fc_val_acc=fc_val_acc,
        cnn_test_acc=cnn_test_acc,
        fc_test_acc=fc_test_acc,
        x_test=x_test,
        y_test=y_test,
        cnn_preds=cnn_preds,
        fc_preds=fc_preds,
        show_idx=show_idx,
    )
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
