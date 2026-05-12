import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix


class NpyWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]                  # [T]
        x = np.expand_dims(x, 0)         # [1, T]
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


class BVP_CNN1D(nn.Module):
    """
    A slightly deeper 1D CNN than the EDA version,
    because BVP windows are much longer (e.g. 3840 for 60s @ 64Hz).
    """
    def __init__(self, n_classes=3, dropout=0.2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),   # T -> T/2

            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),   # T -> T/4

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),   # T -> T/8

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),   # T -> T/16

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x


def compute_class_weights(y: np.ndarray, n_classes=3):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = 1.0 / np.maximum(counts, 1.0)
    w = w * (n_classes / w.sum())
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, n_classes=3):
    model.eval()
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true.append(y.numpy())
        y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    macro_f1 = f1_score(
        y_true, y_pred,
        average="macro",
        labels=list(range(n_classes)),
        zero_division=0
    )
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    return macro_f1, bal_acc, cm


def train_one_fold(
    X,
    y,
    groups,
    test_subject,
    seed=0,
    epochs=30,
    lr=1e-3,
    batch_size=64,
    n_classes=3,
):
    test_mask = (groups == test_subject)
    train_mask = ~test_mask

    X_train_all, y_train_all, g_train_all = X[train_mask], y[train_mask], groups[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # validation split inside training subjects
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(gss.split(X_train_all, y_train_all, groups=g_train_all))

    X_tr, y_tr = X_train_all[tr_idx], y_train_all[tr_idx]
    X_va, y_va = X_train_all[va_idx], y_train_all[va_idx]

    train_loader = DataLoader(NpyWindowDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(NpyWindowDataset(X_va, y_va), batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(NpyWindowDataset(X_test, y_test), batch_size=256, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = BVP_CNN1D(n_classes=n_classes).to(device)

    class_w = compute_class_weights(y_tr, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs

        train_loss = train_loss_sum / max(train_count, 1)
        val_f1, val_ba, _ = evaluate(model, val_loader, device, n_classes=n_classes)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"  Epoch {epoch:02d}/{epochs} | "
            f"loss {train_loss:.4f} | val macroF1 {val_f1:.3f} | val balAcc {val_ba:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_f1, test_ba, cm = evaluate(model, test_loader, device, n_classes=n_classes)
    return best_val_f1, test_f1, test_ba, cm, len(y_test), np.unique(y_test)


def main():
    # -------- config --------
    data_dir = Path(r".\data\UNIVERSE\windows_bvp_60s30s")
    min_test_windows = 50
    min_unique_classes = 3
    n_classes = 3

    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    meta = pd.read_csv(data_dir / "meta.csv")

    groups = meta["subject"].astype(str).values
    subjects = np.unique(groups)

    print("Total windows:", len(y), "T:", X.shape[1])
    print("Total subjects:", len(subjects))

    kept = []
    skipped = []
    cm_sum = np.zeros((n_classes, n_classes), dtype=np.int64)

    for i, sub in enumerate(subjects, 1):
        test_mask = (groups == sub)
        y_test = y[test_mask]
        n_test = len(y_test)
        uniq = np.unique(y_test)

        if n_test < min_test_windows:
            skipped.append({
                "subject": sub,
                "reason": f"n_test<{min_test_windows}",
                "n_test": int(n_test),
                "classes": uniq.tolist()
            })
            print(f"[{i:02d}/{len(subjects)}] subject={sub} SKIP (n_test={n_test} < {min_test_windows})")
            continue

        if len(uniq) < min_unique_classes:
            skipped.append({
                "subject": sub,
                "reason": f"unique_classes<{min_unique_classes}",
                "n_test": int(n_test),
                "classes": uniq.tolist()
            })
            print(f"[{i:02d}/{len(subjects)}] subject={sub} SKIP (classes={uniq.tolist()})")
            continue

        print(f"\n[{i:02d}/{len(subjects)}] subject={sub} training...")
        best_val_f1, test_f1, test_ba, cm, n_test2, uniq2 = train_one_fold(
            X, y, groups,
            test_subject=sub,
            seed=100 + i,
            epochs=30,
            lr=1e-3,
            batch_size=64,
            n_classes=n_classes,
        )

        kept.append({
            "subject": sub,
            "n_test": int(n_test2),
            "classes": uniq2.tolist(),
            "val_best_macroF1": float(best_val_f1),
            "test_macroF1": float(test_f1),
            "test_balAcc": float(test_ba),
        })
        cm_sum += cm

        print(
            f"[{i:02d}/{len(subjects)}] subject={sub} "
            f"testF1={test_f1:.3f} balAcc={test_ba:.3f} (n_test={n_test2})"
        )

    kept_df = pd.DataFrame(kept)
    skipped_df = pd.DataFrame(skipped)

    kept_df.to_csv(data_dir / "loso_results_kept.csv", index=False)
    skipped_df.to_csv(data_dir / "loso_results_skipped.csv", index=False)

    print("\n=== LOSO Summary (KEPT folds only) ===")
    print(f"Kept folds: {len(kept_df)} / {len(subjects)}")
    print(f"Skipped folds: {len(skipped_df)} / {len(subjects)}")

    if len(kept_df) == 0:
        print("No folds kept. Relax thresholds or fix label/data coverage.")
        return

    mean_f1 = kept_df["test_macroF1"].mean()
    std_f1 = kept_df["test_macroF1"].std(ddof=1) if len(kept_df) > 1 else 0.0
    mean_ba = kept_df["test_balAcc"].mean()
    std_ba = kept_df["test_balAcc"].std(ddof=1) if len(kept_df) > 1 else 0.0

    print(f"macroF1: {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"balAcc : {mean_ba:.3f} ± {std_ba:.3f}")
    print("Confusion sum (kept folds):\n", cm_sum)
    print("Saved:")
    print(" -", data_dir / "loso_results_kept.csv")
    print(" -", data_dir / "loso_results_skipped.csv")


if __name__ == "__main__":
    main()