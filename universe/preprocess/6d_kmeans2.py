import os
import glob
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix


# =========================
# Config
# =========================
UNIVERSE_ROOT = Path(r".\data\UNIVERSE")
SESSIONS = ["Lab1", "Lab2"]        # 先只做 Lab，跑通后再考虑 Wild
FS = 4                             # UNIVERSE EDA 预处理脚本里 sf_eda=4
WIN_S = 60
HOP_S = 30
WIN = FS * WIN_S                   # 240
HOP = FS * HOP_S                   # 120
DROP_BASELINE_TASK = {"video_baseline"}  # 一般丢掉 baseline
RANDOM_STATE = 0

# Train
EPOCHS = 25
BATCH_SIZE = 64
LR = 1e-3
VAL_RATIO = 0.2


# =========================
# Utils: data reading
# =========================
def list_subjects(root: Path) -> List[str]:
    subs = sorted([p.name for p in root.glob("UN_*") if p.is_dir()])
    return subs

def find_task_labels_csv(subject_dir: Path, session: str) -> Path | None:
    """
    Try to find Task_Labels.csv under:
      - root/UN_101/Lab1/Task_Labels.csv  (common)
      - root/UN_101/Lab1/Raw/Task_Labels.csv (if someone kept it there)
      - root/UN_101/Lab1/**/Task_Labels.csv  (fallback)
    """
    session_dir = subject_dir / session
    cands = [
        session_dir / "Task_Labels.csv",
        session_dir / "Raw" / "Task_Labels.csv",
    ]
    for c in cands:
        if c.exists():
            return c
    hits = list(session_dir.glob("**/Task_Labels.csv"))
    return hits[0] if hits else None

def find_eda_pkl(subject_dir: Path, session: str, task: str) -> Path | None:
    """
    Find: root/UN_101/Lab1/Preprocessed/<task>/EDA_filtered.pickle
    """
    p = subject_dir / session / "Preprocessed" / task / "EDA_filtered.pickle"
    if p.exists():
        return p
    # fallback: glob
    hits = list((subject_dir / session / "Preprocessed").glob(f"**/{task}/EDA_filtered.pickle"))
    return hits[0] if hits else None

def load_eda_series(pkl_path: Path) -> np.ndarray:
    obj = pickle.load(open(pkl_path, "rb"))
    if hasattr(obj, "to_numpy"):
        x = obj.to_numpy(dtype=np.float32)
    else:
        x = np.asarray(obj, dtype=np.float32)
    return x

def sliding_windows_1d(x: np.ndarray, win: int, hop: int) -> Tuple[np.ndarray, np.ndarray]:
    T = len(x)
    if T < win:
        return np.empty((0, win), dtype=np.float32), np.empty((0,), dtype=np.int64)
    n = 1 + (T - win) // hop
    starts = np.arange(n, dtype=np.int64) * hop
    windows = np.stack([x[s:s+win] for s in starts], axis=0).astype(np.float32)
    return windows, starts

def is_bad_window(w: np.ndarray, nan_ratio_thr=0.10, std_thr=1e-6) -> bool:
    if np.isnan(w).mean() > nan_ratio_thr:
        return True
    ww = w.copy()
    if np.isnan(ww).any():
        med = np.nanmedian(ww)
        ww[np.isnan(ww)] = med if not np.isnan(med) else 0.0
    if float(np.std(ww)) < std_thr:
        return True
    return False


# =========================
# Fold-wise KMeans labeler (k=2) on Weighted NASA Score
# =========================
def load_task_scores(labels_csv: Path) -> pd.DataFrame:
    """
    Expect columns:
      - Task
      - Weighted Nasa Score
    """
    df = pd.read_csv(labels_csv)
    if "Task" not in df.columns:
        raise ValueError(f"{labels_csv} missing 'Task' column")
    # Weighted Nasa Score sometimes is str -> convert
    if "Weighted Nasa Score" not in df.columns:
        raise ValueError(f"{labels_csv} missing 'Weighted Nasa Score' column")
    df = df[["Task", "Weighted Nasa Score"]].copy()
    df["Task"] = df["Task"].astype(str).str.strip()
    df["Weighted Nasa Score"] = pd.to_numeric(df["Weighted Nasa Score"], errors="coerce")
    df = df.dropna(subset=["Weighted Nasa Score"])
    df = df[~df["Task"].isin(DROP_BASELINE_TASK)]
    return df

class KMeansBinaryLabeler:
    """
    Fit on TRAIN subjects' task scores (1D). Then assign labels to any task score.
    Label definition:
      cluster with lower center -> 0 (low)
      cluster with higher center -> 1 (high)
    """
    def __init__(self, random_state=0):
        self.scaler = StandardScaler()
        self.km = KMeans(n_clusters=2, n_init=50, random_state=random_state)
        self.centers_raw = None
        self.low_cluster = None
        self.high_cluster = None

    def fit(self, scores_1d: np.ndarray):
        scores_1d = np.asarray(scores_1d, dtype=float).reshape(-1, 1)
        z = self.scaler.fit_transform(scores_1d)
        self.km.fit(z)
        # centers in raw scale
        centers_raw = self.scaler.inverse_transform(self.km.cluster_centers_).ravel()
        order = np.argsort(centers_raw)  # low -> high
        self.low_cluster = int(order[0])
        self.high_cluster = int(order[1])
        self.centers_raw = centers_raw
        return self

    def predict(self, scores_1d: np.ndarray) -> np.ndarray:
        scores_1d = np.asarray(scores_1d, dtype=float).reshape(-1, 1)
        z = self.scaler.transform(scores_1d)
        c = self.km.predict(z).astype(int)
        y = np.zeros_like(c, dtype=np.int64)
        y[c == self.high_cluster] = 1
        return y


# =========================
# Dataset + Model
# =========================
class NpyWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]            # [T]
        x = np.expand_dims(x, 0)   # [1, T]
        return torch.from_numpy(x), torch.tensor(self.y[idx])

class SimpleCNN1D(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        h = self.feat(x).squeeze(-1)
        return self.head(h)

@torch.no_grad()
def eval_model(model, loader, device):
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
    f1 = f1_score(y_true, y_pred, average="macro")
    ba = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return f1, ba, cm

def class_weights_from_y(y: np.ndarray, n_classes=2) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = 1.0 / np.maximum(counts, 1.0)
    w = w * (n_classes / w.sum())
    return torch.tensor(w, dtype=torch.float32)

def subjectwise_zscore(X: np.ndarray, meta: pd.DataFrame, eps: float = 1e-6) -> np.ndarray:
    """
    X: [N, T]
    meta: DataFrame aligned with X rows, must contain 'subject'
    For each subject, compute mean/std over ALL points of ALL windows in this split,
    then z-score that subject's windows.
    """
    Xn = X.copy().astype(np.float32)
    subjects = meta["subject"].astype(str).values

    for sub in np.unique(subjects):
        idx = np.where(subjects == sub)[0]
        xs = Xn[idx]  # [n_sub, T]
        mu = float(xs.mean())
        sd = float(xs.std())
        Xn[idx] = (xs - mu) / (sd + eps)

    return Xn

# =========================
# Build windows for a set of subjects using fold-wise labeler
# =========================
def build_windows_for_subjects(
    subjects: List[str],
    labeler: KMeansBinaryLabeler,
    root: Path,
    sessions: List[str],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X_list, y_list = [], []
    meta_rows = []

    for sub in subjects:
        sub_dir = root / sub
        for sess in sessions:
            labels_csv = find_task_labels_csv(sub_dir, sess)
            if labels_csv is None:
                continue
            df_scores = load_task_scores(labels_csv)
            if df_scores.empty:
                continue

            # Predict y (0/1) for each task score using fold-wise labeler
            task_names = df_scores["Task"].astype(str).str.strip().tolist()
            scores = df_scores["Weighted Nasa Score"].to_numpy(dtype=float)
            task_y = labeler.predict(scores)  # [n_tasks]

            for task, y_task in zip(task_names, task_y):
                eda_pkl = find_eda_pkl(sub_dir, sess, task)
                if eda_pkl is None or not eda_pkl.exists():
                    continue
                x = load_eda_series(eda_pkl)
                windows, starts = sliding_windows_1d(x, WIN, HOP)
                if len(windows) == 0:
                    continue
                for w, s_idx in zip(windows, starts):
                    if is_bad_window(w):
                        continue
                    X_list.append(w)
                    y_list.append(int(y_task))
                    meta_rows.append({
                        "subject": sub,
                        "session": sess,
                        "task": task,
                        "pkl": str(eda_pkl),
                        "start_idx": int(s_idx),
                        "end_idx": int(s_idx + WIN),
                    })

    if not X_list:
        return np.empty((0, WIN), np.float32), np.empty((0,), np.int64), pd.DataFrame(meta_rows)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


# =========================
# LOSO main
# =========================
def main():
    subjects = list_subjects(UNIVERSE_ROOT)
    if not subjects:
        raise RuntimeError(f"No subjects found under {UNIVERSE_ROOT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for test_sub in subjects:
        train_subs = [s for s in subjects if s != test_sub]

        # ---------- fit fold-wise KMeans on TRAIN subjects' task scores ----------
        train_scores = []
        for sub in train_subs:
            sub_dir = UNIVERSE_ROOT / sub
            for sess in SESSIONS:
                labels_csv = find_task_labels_csv(sub_dir, sess)
                if labels_csv is None:
                    continue
                df_scores = load_task_scores(labels_csv)
                train_scores.extend(df_scores["Weighted Nasa Score"].to_list())

        train_scores = np.asarray(train_scores, dtype=float)
        if len(train_scores) < 10:
            print(f"[{test_sub}] SKIP: not enough train scores")
            continue

        labeler = KMeansBinaryLabeler(random_state=RANDOM_STATE).fit(train_scores)

        # ---------- build windows ----------
        X_tr_all, y_tr_all, meta_tr_all = build_windows_for_subjects(
            train_subs, labeler, UNIVERSE_ROOT, SESSIONS
        )
        X_te, y_te, meta_te = build_windows_for_subjects(
            [test_sub], labeler, UNIVERSE_ROOT, SESSIONS
        )

        if len(y_tr_all) == 0 or len(y_te) == 0:
            print(f"[{test_sub}] SKIP: no windows (train={len(y_tr_all)}, test={len(y_te)})")
            continue

        # ---------- split train -> train/val by subject (still no leakage) ----------
        groups = meta_tr_all["subject"].astype(str).values
        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=RANDOM_STATE)
        tr_idx, va_idx = next(gss.split(X_tr_all, y_tr_all, groups=groups))

        X_tr, y_tr = X_tr_all[tr_idx], y_tr_all[tr_idx]
        X_va, y_va = X_tr_all[va_idx], y_tr_all[va_idx]

        meta_tr = meta_tr_all.iloc[tr_idx].reset_index(drop=True)
        meta_va = meta_tr_all.iloc[va_idx].reset_index(drop=True)
        meta_te = meta_te.reset_index(drop=True)

        # ---------- subject-wise normalization ----------
        X_tr = subjectwise_zscore(X_tr, meta_tr)
        X_va = subjectwise_zscore(X_va, meta_va)
        X_te = subjectwise_zscore(X_te, meta_te)
        
        train_loader = DataLoader(NpyWindowDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(NpyWindowDataset(X_va, y_va), batch_size=256, shuffle=False)
        test_loader  = DataLoader(NpyWindowDataset(X_te, y_te), batch_size=256, shuffle=False)

        # ---------- model ----------
        model = SimpleCNN1D(n_classes=2).to(device)
        w = class_weights_from_y(y_tr, n_classes=2).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=w)
        optim = torch.optim.Adam(model.parameters(), lr=LR)

        best_val_f1 = -1.0
        best_state = None

        # ---------- train ----------
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optim.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optim.step()
                total += float(loss.item()) * len(yb)

            val_f1, val_ba, _ = eval_model(model, val_loader, device)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        test_f1, test_ba, cm = eval_model(model, test_loader, device)

        print(f"[{test_sub}] "
              f"centers_raw={np.sort(labeler.centers_raw)} "
              f"valF1={best_val_f1:.3f} "
              f"testF1={test_f1:.3f} "
              f"balAcc={test_ba:.3f} "
              f"(n_test={len(y_te)})")

        results.append({
            "subject": test_sub,
            "valF1": best_val_f1,
            "testF1": test_f1,
            "balAcc": test_ba,
            "n_test": int(len(y_te)),
            "cm00": int(cm[0,0]), "cm01": int(cm[0,1]),
            "cm10": int(cm[1,0]), "cm11": int(cm[1,1]),
            "low_center": float(np.sort(labeler.centers_raw)[0]),
            "high_center": float(np.sort(labeler.centers_raw)[1]),
        })

    if not results:
        raise RuntimeError("No folds produced results. Check paths / Task_Labels.csv / EDA_filtered.pickle.")

    df = pd.DataFrame(results)
    print("\n==== LOSO Summary ====")
    print(df[["testF1", "balAcc", "n_test"]].describe())
    out_path = UNIVERSE_ROOT / "results_loso_kmeans2_eda_cnn.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()