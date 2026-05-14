import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset


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


def to_binary_keep_extremes(
    X_eda: np.ndarray,
    X_bvp: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
):
    keep = y != 1
    y2 = y[keep].copy()
    y2[y2 == 2] = 1
    return (
        X_eda[keep],
        X_bvp[keep],
        y2.astype(np.int64),
        meta.iloc[np.where(keep)[0]].reset_index(drop=True).copy(),
    )


def load_and_pair_data(eda_dir: Path, bvp_dir: Path, binary: bool = False):
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
        raise RuntimeError("No paired EDA/BVP windows found. Check meta.csv keys.")

    eda_idx = np.array([eda_map[k] for k in common_keys], dtype=np.int64)
    bvp_idx = np.array([bvp_map[k] for k in common_keys], dtype=np.int64)

    X_eda_pair = X_eda[eda_idx]
    X_bvp_pair = X_bvp[bvp_idx]
    y_pair = y_eda[eda_idx]
    y_bvp_pair = y_bvp[bvp_idx]

    if not np.array_equal(y_pair, y_bvp_pair):
        mismatch = np.where(y_pair != y_bvp_pair)[0]
        raise RuntimeError(f"Paired labels differ, mismatch count = {len(mismatch)}")

    meta_pair = meta_eda.iloc[eda_idx].reset_index(drop=True).copy()
    if binary:
        X_eda_pair, X_bvp_pair, y_pair, meta_pair = to_binary_keep_extremes(
            X_eda_pair, X_bvp_pair, y_pair, meta_pair
        )

    return X_eda_pair, X_bvp_pair, y_pair.astype(np.int64), meta_pair


class LateFusionDataset(Dataset):
    def __init__(self, X_eda: np.ndarray, X_bvp: np.ndarray, y: np.ndarray):
        assert len(X_eda) == len(X_bvp) == len(y)
        self.X_eda = X_eda.astype(np.float32)
        self.X_bvp = X_bvp.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        eda = torch.from_numpy(self.X_eda[idx]).float().unsqueeze(0)
        bvp = torch.from_numpy(self.X_bvp[idx]).float().unsqueeze(0)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return eda, bvp, y


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.main(x) + self.shortcut(x))


class ResNetBranch1D(nn.Module):
    def __init__(
        self,
        stem_channels: int,
        stem_kernel: int,
        stages: tuple[tuple[int, int, int], ...],
        dropout: float = 0.2,
        stem_stride: int = 1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(
                1,
                stem_channels,
                kernel_size=stem_kernel,
                stride=stem_stride,
                padding=stem_kernel // 2,
                bias=False,
            ),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(),
        )

        blocks = []
        in_channels = stem_channels
        for out_channels, kernel_size, stride in stages:
            blocks.append(
                ResidualBlock1D(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = in_channels

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)


class LateFusionResNet1D(nn.Module):
    def __init__(self, n_classes: int = 3, dropout: float = 0.25):
        super().__init__()

        # EDA is slow and short, so use wider kernels and a smaller branch.
        self.eda_branch = ResNetBranch1D(
            stem_channels=32,
            stem_kernel=15,
            stages=((32, 15, 1), (64, 11, 2), (64, 7, 1)),
            dropout=dropout,
        )
        # BVP is long and high-rate, so use a deeper branch with progressive downsampling.
        self.bvp_branch = ResNetBranch1D(
            stem_channels=32,
            stem_kernel=9,
            stem_stride=2,
            stages=((32, 9, 1), (64, 9, 2), (128, 7, 2), (128, 5, 1)),
            dropout=dropout,
        )

        fusion_dim = self.eda_branch.out_dim + self.bvp_branch.out_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, eda, bvp):
        eda_feat = self.eda_branch(eda)
        bvp_feat = self.bvp_branch(bvp)
        fused = torch.cat([eda_feat, bvp_feat], dim=1)
        return self.head(fused)


def compute_class_weights(y: np.ndarray, n_classes: int):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights * (n_classes / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, n_classes: int):
    model.eval()
    y_true, y_pred = [], []

    for eda, bvp, y in loader:
        eda = eda.to(device)
        bvp = bvp.to(device)
        logits = model(eda, bvp)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true.append(y.numpy())
        y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    labels = list(range(n_classes))

    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
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
    X_eda,
    X_bvp,
    y,
    groups,
    test_subject,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    dropout: float,
    n_classes: int,
):
    test_mask = groups == test_subject
    train_mask = ~test_mask

    X_eda_train_all, X_bvp_train_all = X_eda[train_mask], X_bvp[train_mask]
    y_train_all, g_train_all = y[train_mask], groups[train_mask]
    X_eda_test, X_bvp_test, y_test = X_eda[test_mask], X_bvp[test_mask], y[test_mask]

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

    train_loader = DataLoader(
        LateFusionDataset(X_eda_tr, X_bvp_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        LateFusionDataset(X_eda_va, X_bvp_va, y_va),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        LateFusionDataset(X_eda_test, X_bvp_test, y_test),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LateFusionResNet1D(n_classes=n_classes, dropout=dropout).to(device)
    class_w = compute_class_weights(y_tr, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for eda, bvp, yb in train_loader:
            eda = eda.to(device)
            bvp = bvp.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(eda, bvp)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Late-fusion EDA+BVP ResNet1D LOSO on UNIVERSE.")
    parser.add_argument("--binary", action="store_true", help="Use low-vs-high labels and drop mid.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--min-test-windows", type=int, default=None)
    parser.add_argument("--eda-dir", type=Path, default=Path(r".\data\UNIVERSE\windows_eda_60s30s"))
    parser.add_argument("--bvp-dir", type=Path, default=Path(r".\data\UNIVERSE\windows_bvp_60s30s"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r".\data\UNIVERSE\windows_eda_bvp_latefusion_resnet_60s30s"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_classes = 2 if args.binary else 3
    min_unique_classes = n_classes
    min_test_windows = args.min_test_windows
    if min_test_windows is None:
        min_test_windows = 30 if args.binary else 50

    out_dir = args.out_dir / "binary_loso" if args.binary else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    X_eda, X_bvp, y, meta = load_and_pair_data(args.eda_dir, args.bvp_dir, binary=args.binary)
    groups = meta["subject"].astype(str).values
    subjects = np.unique(groups)

    task_name = "Binary" if args.binary else "3-class"
    print(f"{task_name} paired windows:", len(y))
    print("EDA shape:", X_eda.shape)
    print("BVP shape:", X_bvp.shape)
    print("Total subjects:", len(subjects))
    print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))

    kept = []
    skipped = []
    cm_sum = np.zeros((n_classes, n_classes), dtype=np.int64)

    for i, sub in enumerate(subjects, 1):
        test_mask = groups == sub
        y_test = y[test_mask]
        n_test = len(y_test)
        uniq = np.unique(y_test)

        if n_test < min_test_windows:
            skipped.append(
                {
                    "subject": sub,
                    "reason": f"n_test<{min_test_windows}",
                    "n_test": int(n_test),
                    "classes": uniq.tolist(),
                }
            )
            print(f"[{i:02d}/{len(subjects)}] subject={sub} SKIP (n_test={n_test} < {min_test_windows})")
            continue

        if len(uniq) < min_unique_classes:
            skipped.append(
                {
                    "subject": sub,
                    "reason": f"unique_classes<{min_unique_classes}",
                    "n_test": int(n_test),
                    "classes": uniq.tolist(),
                }
            )
            print(f"[{i:02d}/{len(subjects)}] subject={sub} SKIP (classes={uniq.tolist()})")
            continue

        print(f"\n[{i:02d}/{len(subjects)}] subject={sub} training...")
        best_val_f1, test_f1, test_ba, cm, n_test2, uniq2 = train_one_fold(
            X_eda,
            X_bvp,
            y,
            groups,
            test_subject=sub,
            seed=100 + i,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            dropout=args.dropout,
            n_classes=n_classes,
        )

        kept.append(
            {
                "subject": sub,
                "n_test": int(n_test2),
                "classes": uniq2.tolist(),
                "val_best_macroF1": float(best_val_f1),
                "test_macroF1": float(test_f1),
                "test_balAcc": float(test_ba),
            }
        )
        cm_sum += cm

        print(
            f"[{i:02d}/{len(subjects)}] subject={sub} "
            f"testF1={test_f1:.3f} balAcc={test_ba:.3f} (n_test={n_test2})"
        )

    kept_df = pd.DataFrame(kept)
    skipped_df = pd.DataFrame(skipped)
    kept_df.to_csv(out_dir / "loso_results_kept.csv", index=False)
    skipped_df.to_csv(out_dir / "loso_results_skipped.csv", index=False)

    print(f"\n=== Late Fusion ResNet {task_name} LOSO Summary (KEPT folds only) ===")
    print(f"Kept folds: {len(kept_df)} / {len(subjects)}")
    print(f"Skipped folds: {len(skipped_df)} / {len(subjects)}")

    if len(kept_df) == 0:
        print("No folds kept.")
        return

    mean_f1 = kept_df["test_macroF1"].mean()
    std_f1 = kept_df["test_macroF1"].std(ddof=1) if len(kept_df) > 1 else 0.0
    mean_ba = kept_df["test_balAcc"].mean()
    std_ba = kept_df["test_balAcc"].std(ddof=1) if len(kept_df) > 1 else 0.0

    print(f"macroF1: {mean_f1:.3f} +/- {std_f1:.3f}")
    print(f"balAcc : {mean_ba:.3f} +/- {std_ba:.3f}")
    print("Confusion sum (kept folds):\n", cm_sum)
    print("Saved:")
    print(" -", out_dir / "loso_results_kept.csv")
    print(" -", out_dir / "loso_results_skipped.csv")


if __name__ == "__main__":
    main()
