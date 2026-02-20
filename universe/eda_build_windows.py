import os
import glob
import pickle
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

LABEL_MAP = {"low": 0, "mid": 1, "high": 2}

def load_eda_series(pkl_path: Path) -> np.ndarray:
    """Load EDA_filtered.pickle -> np.ndarray float32, shape [T]."""
    obj = pickle.load(open(pkl_path, "rb"))
    # Typically pandas Series
    if hasattr(obj, "to_numpy"):
        x = obj.to_numpy(dtype=np.float32)
    else:
        x = np.asarray(obj, dtype=np.float32)
    return x

def sliding_windows_1d(x: np.ndarray, win: int, hop: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (windows, starts):
      windows: [N, win]
      starts:  [N] start indices
    """
    T = len(x)
    if T < win:
        return np.empty((0, win), dtype=np.float32), np.empty((0,), dtype=np.int64)
    n = 1 + (T - win) // hop
    starts = np.arange(n, dtype=np.int64) * hop
    windows = np.stack([x[s:s+win] for s in starts], axis=0).astype(np.float32)
    return windows, starts

def is_bad_window(w: np.ndarray, nan_ratio_thr: float = 0.10, std_thr: float = 1e-6) -> bool:
    """Filter windows with too many NaNs or near-constant values."""
    if np.isnan(w).mean() > nan_ratio_thr:
        return True
    # After NaN handling, check variance
    ww = w.copy()
    if np.isnan(ww).any():
        # simple impute with window median
        med = np.nanmedian(ww)
        ww[np.isnan(ww)] = med if not np.isnan(med) else 0.0
    if float(np.std(ww)) < std_thr:
        return True
    return False

def find_label_csv(session_root: Path, clustered_labels_dir: Path | None = None) -> Path | None:
    """
    Find Task_Labels_clustered_6d.csv for a given session in clustered_labels_dir/UN_101/Lab1/Task_Labels_clustered_6d.csv
    """

    if clustered_labels_dir is None or not clustered_labels_dir.exists():
        return None

    subject = session_root.parent.name  # UN_101
    session = session_root.name         # Lab1 / Lab2 / Wild

    p2 = clustered_labels_dir / subject / session / "Task_Labels_clustered_6d.csv"
    if p2.exists():
        return p2
    return None

def load_task_to_label(label_csv: Path) -> Dict[str, int]:
    df = pd.read_csv(label_csv)
    if "Task" not in df.columns or "tlx6_level" not in df.columns:
        raise ValueError(f"{label_csv} must contain columns: Task, tlx6_level")
    mapping = {}
    for _, r in df.iterrows():
        task = str(r["Task"])
        lvl = str(r["tlx6_level"]).strip().lower()
        if lvl not in LABEL_MAP:
            continue
        mapping[task] = LABEL_MAP[lvl]
    return mapping

def compute_minmax_from_baseline(session_root: Path,
                                baseline_task: str = "video_baseline",
                                robust: bool = False,
                                q_low: float = 0.01,
                                q_high: float = 0.99,
                                eps: float = 1e-6) -> Tuple[float, float] | None:
    """
    session_root: .../UN_101/Lab1
    baseline_task folder expected at: session_root/Preprocessed/video_baseline/EDA_filtered.pickle
    robust=False: use min/max
    robust=True : use quantiles (q_low/q_high) to reduce spike influence
    """
    baseline_pkl = session_root / "Preprocessed" / baseline_task / "EDA_filtered.pickle"
    if not baseline_pkl.exists():
        return None

    xb = load_eda_series(baseline_pkl)
    xb = xb[~np.isnan(xb)]
    if xb.size < 10:
        return None

    if robust:
        xmin = float(np.quantile(xb, q_low))
        xmax = float(np.quantile(xb, q_high))
    else:
        xmin = float(np.min(xb))
        xmax = float(np.max(xb))

    if (xmax - xmin) < eps:
        return None
    return xmin, xmax


def user_minmax_normalize(x: np.ndarray,
                          xmin: float,
                          xmax: float,
                          eps: float = 1e-6,
                          clip: bool = True) -> np.ndarray:
    x_norm = (x - xmin) / (xmax - xmin + eps)
    if clip:
        x_norm = np.clip(x_norm, 0.0, 1.0)
    return x_norm.astype(np.float32)


def main():

    universe_root = Path(r"./data/UNIVERSE")
    out_dir = universe_root / "windows_eda_60s30s"
    out_dir.mkdir(parents=True, exist_ok=True)

    clustered_labels_dir = universe_root / "clustered_labels_6d"

    fs = 4
    win_s = 60
    hop_s = 30
    win = fs * win_s   # 240
    hop = fs * hop_s   # 120

    sessions = ["Lab1", "Lab2"]

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta_rows: List[dict] = []

    label_cache: Dict[Tuple[str, str], Dict[str, int]] = {}

    minmax_cache: Dict[Tuple[str, str], Tuple[float, float] | None] = {}

    for session in sessions:
        pattern = str(universe_root / "UN_*" / session / "Preprocessed" / "*" / "EDA_filtered.pickle")
        for p in glob.glob(pattern):
            pkl_path = Path(p)
            task_name = pkl_path.parent.name
            session_root = pkl_path.parents[2]  # .../UN_101/Lab1
            subject_id = session_root.parent.name
            session_name = session_root.name

            key = (subject_id, session_name)
            if key not in label_cache:
                label_csv = find_label_csv(session_root, clustered_labels_dir=clustered_labels_dir)
                if label_csv is None:
                    print(f"{subject_id}/{session_name}没有找到对应的标签 CSV，后续这个 session 的所有任务都会被跳过。")
                    label_cache[key] = {}
                else:
                    label_cache[key] = load_task_to_label(label_csv)

            task2label = label_cache[key]
            if task_name not in task2label:
                print(f"{subject_id}/{session_name}的任务 {task_name} 没有找到对应的标签，跳过这个任务。")
                continue

            y_task = task2label[task_name]

            # ---- baseline min-max (user-specific, computed once per subject/session) ----
            if key not in minmax_cache:
                # robust=True 会更稳（避免 baseline 尖峰把 min/max 拉爆）
                minmax_cache[key] = compute_minmax_from_baseline(
                    session_root=session_root,
                    baseline_task="relaxation_video",
                    robust=True,      # 想严格复刻 Prajod 可改成 False
                    q_low=0.01,
                    q_high=0.99
                )

            mm = minmax_cache[key]
            if mm is None:
                print(f"{subject_id}/{session_name} 找不到可用 baseline(relaxation_video) 来算 min/max，跳过该 session。")
                continue

            xmin, xmax = mm
            x = load_eda_series(pkl_path)
            x = user_minmax_normalize(x, xmin, xmax, clip=True)

            windows, starts = sliding_windows_1d(x, win=win, hop=hop)

            if len(windows) == 0:
                print(f"{subject_id}/{session_name}/{task_name} 的 EDA 数据长度 {len(x)} 小于窗口大小 {win}，没有生成任何窗口，跳过这个任务。")
                continue

            for w, s_idx in zip(windows, starts):
                if is_bad_window(w):
                    continue
                X_list.append(w)
                y_list.append(int(y_task))
                meta_rows.append({
                    "subject": subject_id,
                    "session": session_name,
                    "task": task_name,
                    "pkl_path": str(pkl_path),
                    "start_idx": int(s_idx),
                    "end_idx": int(s_idx + win),
                    "fs": fs,
                    "win_s": win_s,
                    "hop_s": hop_s,
                })

    if not X_list:
        raise RuntimeError("没有找到任何有效的窗口，检查数据和配置。")

    X = np.stack(X_list, axis=0).astype(np.float32)  # [N, 240]
    y = np.asarray(y_list, dtype=np.int64)           # [N]
    meta = pd.DataFrame(meta_rows)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    meta.to_csv(out_dir / "meta.csv", index=False)

    print("Saved to:", out_dir)
    print("X shape:", X.shape, "y shape:", y.shape)
    print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))

if __name__ == "__main__":
    main()
