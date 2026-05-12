import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix


def to_binary_keep_extremes(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame):
    keep = (y != 1)
    X2 = X[keep]
    y2 = y[keep].copy()
    y2[y2 == 2] = 1
    meta2 = meta.iloc[np.where(keep)[0]].reset_index(drop=True).copy()
    return X2, y2.astype(np.int64), meta2


def resample_1d_linear(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == target_len:
        return x.astype(np.float32)
    if len(x) <= 1:
        return np.zeros(target_len, dtype=np.float32)

    old_idx = np.linspace(0.0, 1.0, num=len(x), dtype=np.float32)
    new_idx = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(new_idx, old_idx, x).astype(np.float32)


def make_pair_key(df: pd.DataFrame) -> pd.Series:
    start_s = np.round(df["start_time_s"].astype(float).values, 6)
    end_s = np.round(df["end_time_s"].astype(float).values, 6)
    return (
        df["subject"].astype(str)
        + "||" + df["session"].astype(str)
        + "||" + df["task"].astype(str)
        + "||" + pd.Series(start_s).astype(str)
        + "||" + pd.Series(end_s).astype(str)
    )


class EarlyFusionDataset(Dataset):
    def __init__(self, X_eda: np.ndarray, X_bvp: np.ndarray, y: np.ndarray):
        assert len(X_eda) == len(X_bvp) == len(y)
        target_len = X_bvp.shape[1]
        X_eda_rs = np.stack([resample_1d_linear(x, target_len) for x in X_eda], axis=0)
        self.X = np.stack([X_eda_rs, X_bvp.astype(np.float32)], axis=1)  # [N, 2, T]
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)


class EarlyFusionCNN1D(nn.Module):
    def __init__(self, n_classes=2, dropout=0.2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

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


def compute_class_weights(y: np.ndarray, n_classes=2):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = 1.0 / np.maximum(counts, 1.0)
    w = w * (n_classes / w.sum())
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, n_classes=2):
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


def fit_standardizer(X: np.ndarray):
    mean = float(np.mean(X))
    std = float(np.std(X))
    if std < 1e-6:
        std = 1.0
    return mean, std


def apply_standardizer(X: np.ndarray, mean: float, std: float, clip: float | None = 5.0):
    Xn = (X - mean) / std
    if clip is not None:
        Xn = np.clip(Xn, -clip, clip)
    return Xn.astype(np.float32)


def train_one_fold(
    X_eda, X_bvp, y, groups, test_subject,
    seed=0, epochs=30, lr=1e-3, batch_size=64, n_classes=2
):
    test_mask = (groups == test_subject)
    train_mask = ~test_mask

    X_eda_train_all, X_bvp_train_all = X_eda[train_mask], X_bvp[train_mask]
    y_train_all, g_train_all = y[train_mask], groups[train_mask]

    X_eda_test, X_bvp_test = X_eda[test_mask], X_bvp[test_mask]
    y_test = y[test_mask]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(gss.split(X_eda_train_all, y_train_all, groups=g_train_all))

    X_eda_tr, X_bvp_tr, y_tr = X_eda_train_all[tr_idx], X_bvp_train_all[tr_idx], y_train_all[tr_idx]
    X_eda_va, X_bvp_va, y_va = X_eda_train_all[va_idx], X_bvp_train_all[va_idx], y_train_all[va_idx]

    eda_mean, eda_std = fit_standardizer(X_eda_tr)
    bvp_mean, bvp_std = fit_standardizer(X_bvp_tr)

    X_eda_tr = apply_standardizer(X_eda_tr, eda_mean, eda_std)
    X_eda_va = apply_standardizer(X_eda_va, eda_mean, eda_std)
    X_eda_test = apply_standardizer(X_eda_test, eda_mean, eda_std)

    X_bvp_tr = apply_standardizer(X_bvp_tr, bvp_mean, bvp_std)
    X_bvp_va = apply_standardizer(X_bvp_va, bvp_mean, bvp_std)
    X_bvp_test = apply_standardizer(X_bvp_test, bvp_mean, bvp_std)

    train_loader = DataLoader(EarlyFusionDataset(X_eda_tr, X_bvp_tr, y_tr), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(EarlyFusionDataset(X_eda_va, X_bvp_va, y_va), batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(EarlyFusionDataset(X_eda_test, X_bvp_test, y_test), batch_size=256, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = EarlyFusionCNN1D(n_classes=n_classes).to(device)
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


def load_and_pair_data(eda_dir: Path, bvp_dir: Path):
    X_eda = np.load(eda_dir / "X.npy")
    y_eda = np.load(eda_dir / "y.npy")
    meta_eda = pd.read_csv(eda_dir / "meta.csv").copy()

    X_bvp = np.load(bvp_dir / "X.npy")
    y_bvp = np.load(bvp_dir / "y.npy")
    meta_bvp = pd.read_csv(bvp_dir / "meta.csv").copy()

    meta_eda["pair_key"] = make_pair_key(meta_eda)
    meta_bvp["pair_key"] = make_pair_key(meta_bvp)

    eda_map = {k: i for i, k in enumerate(meta_eda["pair_key"].tolist())}
    bvp_map = {k: i for i, k in enumerate(meta_bvp["pair_key"].tolist())}

    common_keys = sorted(set(eda_map.keys()) & set(bvp_map.keys()))
    if len(common_keys) == 0:
        raise RuntimeError("EDA 和 BVP 没有配对上的窗口。")

    eda_idx = np.array([eda_map[k] for k in common_keys], dtype=np.int64)
    bvp_idx = np.array([bvp_map[k] for k in common_keys], dtype=np.int64)

    X_eda_pair = X_eda[eda_idx]
    X_bvp_pair = X_bvp[bvp_idx]
    y1 = y_eda[eda_idx]
    y2 = y_bvp[bvp_idx]

    if not np.array_equal(y1, y2):
        mismatch = np.where(y1 != y2)[0]
        raise RuntimeError(f"配对后标签不一致，mismatch count = {len(mismatch)}")

    meta_pair = meta_eda.iloc[eda_idx].reset_index(drop=True).copy()
    y_pair = y1.copy()

    X_eda_pair, y_pair, meta_pair = to_binary_keep_extremes(X_eda_pair, y_pair, meta_pair)
    X_bvp_pair = X_bvp_pair[(y1 != 1)]
    y_pair = y_pair.astype(np.int64)

    return X_eda_pair, X_bvp_pair, y_pair, meta_pair


def main():
    eda_dir = Path(r".\data\UNIVERSE\windows_eda_60s30s")
    bvp_dir = Path(r".\data\UNIVERSE\windows_bvp_60s30s")
    out_dir = Path(r".\data\UNIVERSE\windows_eda_bvp_earlyfusion_60s30s\binary_loso")
    out_dir.mkdir(parents=True, exist_ok=True)

    min_test_windows = 30
    min_unique_classes = 2
    n_classes = 2

    X_eda, X_bvp, y, meta = load_and_pair_data(eda_dir, bvp_dir)
    groups = meta["subject"].astype(str).values
    subjects = np.unique(groups)

    print("Binary paired windows:", len(y))
    print("EDA shape:", X_eda.shape)
    print("BVP shape:", X_bvp.shape)
    print("Total subjects:", len(subjects))
    print("Binary class counts:", dict(zip(*np.unique(y, return_counts=True))))

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
            X_eda, X_bvp, y, groups,
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

    kept_df.to_csv(out_dir / "loso_results_kept.csv", index=False)
    skipped_df.to_csv(out_dir / "loso_results_skipped.csv", index=False)

    print("\n=== Binary LOSO Summary (KEPT folds only) ===")
    print(f"Kept folds: {len(kept_df)} / {len(subjects)}")
    print(f"Skipped folds: {len(skipped_df)} / {len(subjects)}")

    if len(kept_df) == 0:
        print("No folds kept.")
        return

    mean_f1 = kept_df["test_macroF1"].mean()
    std_f1 = kept_df["test_macroF1"].std(ddof=1) if len(kept_df) > 1 else 0.0
    mean_ba = kept_df["test_balAcc"].mean()
    std_ba = kept_df["test_balAcc"].std(ddof=1) if len(kept_df) > 1 else 0.0

    print(f"macroF1: {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"balAcc : {mean_ba:.3f} ± {std_ba:.3f}")
    print("Confusion sum (kept folds):\n", cm_sum)

    
    print("Saved:")
    print(" -", out_dir / "loso_results_kept.csv")
    print(" -", out_dir / "loso_results_skipped.csv")


if __name__ == "__main__":
    main()