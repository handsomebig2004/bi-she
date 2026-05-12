import os
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix


# Dataset
class NpyWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: [N, T] float32
        y: [N] int64 (0/1/2)
        """
        assert X.ndim == 2, f"Expected X shape [N,T], got {X.shape}"
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]              # [T]
        x = np.expand_dims(x, 0)     # [C=1, T]
        return torch.from_numpy(x), torch.tensor(self.y[idx])


# Model
class SimpleCNN1D(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()

        # A very small CNN; good baseline
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),

            # Global average pooling over time
            nn.AdaptiveAvgPool1d(1),  # -> [B, 64, 1]
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: [B, 1, T]
        h = self.net(x).squeeze(-1)  # [B, 64]
        return self.head(h)          # [B, n_classes]


# Train 
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return macro_f1, bal_acc, cm


def compute_class_weights(y: np.ndarray, n_classes=3):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    # inverse frequency
    w = 1.0 / np.maximum(counts, 1.0)
    w = w * (n_classes / w.sum())
    return torch.tensor(w, dtype=torch.float32)


def main():
    # ---- paths ----
    data_dir = Path(r".\data\UNIVERSE\windows_eda_60s30s")
    X_path = data_dir / "X.npy"
    y_path = data_dir / "y.npy"
    meta_path = data_dir / "meta.csv"

    assert X_path.exists() and y_path.exists() and meta_path.exists(), \
        f"Missing files in {data_dir}. Need X.npy, y.npy, meta.csv"

    X = np.load(X_path)   # [N, T]
    y = np.load(y_path)   # [N]
    meta = pd.read_csv(meta_path)
    groups = meta["subject"].astype(str).values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    tr_idx, val_idx = next(gss2.split(X[train_idx], y[train_idx], groups=groups[train_idx]))
    tr_idx = train_idx[tr_idx]
    val_idx = train_idx[val_idx]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    print("Shapes:", X_tr.shape, X_va.shape, X_te.shape)
    print("Train class counts:", dict(zip(*np.unique(y_tr, return_counts=True))))
    print("Val   class counts:", dict(zip(*np.unique(y_va, return_counts=True))))
    print("Test  class counts:", dict(zip(*np.unique(y_te, return_counts=True))))

    train_ds = NpyWindowDataset(X_tr, y_tr)
    val_ds   = NpyWindowDataset(X_va, y_va)
    test_ds  = NpyWindowDataset(X_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN1D(n_classes=3).to(device)

    class_w = compute_class_weights(y_tr, n_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0
        for x, yb in train_loader:
            x = x.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(yb)

        avg_loss = total_loss / len(train_ds)

        val_f1, val_ba, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | loss {avg_loss:.4f} | val macroF1 {val_f1:.3f} | val balAcc {val_ba:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ---- test best ----
    if best_state is not None:
        model.load_state_dict(best_state)

    test_f1, test_ba, cm = evaluate(model, test_loader, device)
    print("\nBEST VAL macroF1:", best_val_f1)
    print("TEST macroF1:", test_f1)
    print("TEST balAcc :", test_ba)
    print("Confusion matrix:\n", cm)


if __name__ == "__main__":
    main()
