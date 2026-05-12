import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix


# -------------- Dataset --------------
class NpyWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        x = np.expand_dims(x, 0)  # [1,T]
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


# -------------- Model (2-class) --------------
class SimpleCNN1D(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        h = self.net(x).squeeze(-1)
        return self.head(h)


def class_weights_binary(y):
    # inverse freq
    counts = np.bincount(y, minlength=2).astype(np.float32)
    w = 1.0 / np.maximum(counts, 1.0)
    w = w * (2 / w.sum())
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    yt, yp = [], []
    for x, y in loader:
        x = x.to(device)
        pred = torch.argmax(model(x), dim=1).cpu().numpy()
        yt.append(y.numpy())
        yp.append(pred)
    yt = np.concatenate(yt)
    yp = np.concatenate(yp)

    f1 = f1_score(yt, yp, average="binary", zero_division=0)
    ba = balanced_accuracy_score(yt, yp)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    return float(f1), float(ba), cm


def train_one_fold(X, y, groups, test_sub, seed=0, epochs=20, lr=1e-3, batch=64):
    test_mask = (groups == test_sub)
    train_mask = ~test_mask

    X_train_all, y_train_all, g_train_all = X[train_mask], y[train_mask], groups[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # val split inside train subjects
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(gss.split(X_train_all, y_train_all, groups=g_train_all))

    X_tr, y_tr = X_train_all[tr_idx], y_train_all[tr_idx]
    X_va, y_va = X_train_all[va_idx], y_train_all[va_idx]

    train_loader = DataLoader(NpyWindowDataset(X_tr, y_tr), batch_size=batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(NpyWindowDataset(X_va, y_va), batch_size=256, shuffle=False, num_workers=0)
    test_loader  = DataLoader(NpyWindowDataset(X_test, y_test), batch_size=256, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    model = SimpleCNN1D(n_classes=2).to(device)
    w = class_weights_binary(y_tr).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        val_f1, _, _ = evaluate(model, val_loader, device)
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_f1, test_ba, cm = evaluate(model, test_loader, device)
    return best_val, test_f1, test_ba, cm, int(len(y_test))


def main():
    universe_root = Path(r".\data\UNIVERSE")
    win_dir = universe_root / "windows_eda_60s30s"
    label_dir = win_dir / "fold_labels_bin"

    X = np.load(win_dir / "X.npy")
    meta = pd.read_csv(win_dir / "meta.csv")
    groups = meta["subject"].astype(str).values
    subjects = np.unique(groups)

    min_test_windows = 50
    require_classes = 2

    kept, skipped = [], []
    cm_sum = np.zeros((2,2), dtype=np.int64)

    print("Total windows:", len(meta), "T:", X.shape[1])
    print("Total subjects:", len(subjects))

    for i, test_sub in enumerate(subjects, 1):
        npz_path = label_dir / f"fold_{test_sub}_y.npz"
        if not npz_path.exists():
            skipped.append({"subject": test_sub, "reason": "missing_label_file"})
            print(f"[{i:02d}/{len(subjects)}] subject={test_sub} SKIP (missing label npz)")
            continue

        y_fold = np.load(npz_path, allow_pickle=True)["y"].astype(np.int64)

        # filter missing labels
        keep = (y_fold >= 0)
        X2 = X[keep]
        y2 = y_fold[keep]
        groups2 = groups[keep]

        test_mask = (groups2 == test_sub)
        n_test = int(np.sum(test_mask))
        uniq = np.unique(y2[test_mask]) if n_test > 0 else np.array([])

        if n_test < min_test_windows:
            skipped.append({"subject": test_sub, "reason": f"n_test<{min_test_windows}", "n_test": n_test, "classes": uniq.tolist()})
            print(f"[{i:02d}/{len(subjects)}] subject={test_sub} SKIP (n_test={n_test} < {min_test_windows})")
            continue

        if len(uniq) < require_classes:
            skipped.append({"subject": test_sub, "reason": "missing_class", "n_test": n_test, "classes": uniq.tolist()})
            print(f"[{i:02d}/{len(subjects)}] subject={test_sub} SKIP (classes={uniq.tolist()})")
            continue

        best_val, test_f1, test_ba, cm, n_test2 = train_one_fold(X2, y2, groups2, test_sub, seed=100+i)
        cm_sum += cm
        kept.append({
            "subject": test_sub,
            "n_test": n_test2,
            "val_best_f1": best_val,
            "test_f1": test_f1,
            "test_balAcc": test_ba,
        })
        print(f"[{i:02d}/{len(subjects)}] subject={test_sub} testF1={test_f1:.3f} balAcc={test_ba:.3f} (n_test={n_test2})")

    kept_df = pd.DataFrame(kept)
    skipped_df = pd.DataFrame(skipped)

    kept_df.to_csv(win_dir / "loso_bin_kept.csv", index=False)
    skipped_df.to_csv(win_dir / "loso_bin_skipped.csv", index=False)

    print("\n=== LOSO Binary Summary (kept only) ===")
    print(f"Kept folds: {len(kept_df)} / {len(subjects)}")
    print(f"Skipped folds: {len(skipped_df)} / {len(subjects)}")

    if len(kept_df) == 0:
        print("No folds kept.")
        return

    mean_f1 = kept_df["test_f1"].mean()
    std_f1 = kept_df["test_f1"].std(ddof=1) if len(kept_df) > 1 else 0.0
    mean_ba = kept_df["test_balAcc"].mean()
    std_ba = kept_df["test_balAcc"].std(ddof=1) if len(kept_df) > 1 else 0.0

    print(f"F1: {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"balAcc: {mean_ba:.3f} ± {std_ba:.3f}")
    print("Confusion sum:\n", cm_sum)
    print("Saved:", win_dir / "loso_bin_kept.csv", "and", win_dir / "loso_bin_skipped.csv")


if __name__ == "__main__":
    main()