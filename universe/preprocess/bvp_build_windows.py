import argparse
import glob
import pickle
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

LABEL_MAP = {"low": 0, "mid": 1, "high": 2}

# 文件夹任务名 -> 标签表任务名
TASK_NAME_ALIASES = {
    "relaxation_video": "video_baseline",
}


def normalize_task_name_for_label(task_name: str) -> str:
    return TASK_NAME_ALIASES.get(task_name, task_name)


def load_bvp_series(pkl_path: Path) -> np.ndarray:
    """Load BVP_filtered.pickle -> np.ndarray float32, shape [T]."""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if hasattr(obj, "to_numpy"):
        x = obj.to_numpy(dtype=np.float32)
    else:
        x = np.asarray(obj, dtype=np.float32)

    x = np.squeeze(x).astype(np.float32)
    if x.ndim != 1:
        raise ValueError(f"{pkl_path} 读取后不是 1D 序列，shape={x.shape}")
    return x


def sliding_windows_1d(x: np.ndarray, win: int, hop: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      windows: [N, win]
      starts:  [N]
    """
    T = len(x)
    if T < win:
        return np.empty((0, win), dtype=np.float32), np.empty((0,), dtype=np.int64)

    n = 1 + (T - win) // hop
    starts = np.arange(n, dtype=np.int64) * hop
    windows = np.stack([x[s:s + win] for s in starts], axis=0).astype(np.float32)
    return windows, starts


def is_bad_window(
    w: np.ndarray,
    nan_ratio_thr: float = 0.10,
    std_thr: float = 1e-6,
    abs_max_thr: float = 1e6,
) -> bool:
    """Filter windows with too many NaNs / near-constant / exploding values."""
    if np.isnan(w).mean() > nan_ratio_thr:
        return True

    ww = w.copy()
    if np.isnan(ww).any():
        med = np.nanmedian(ww)
        ww[np.isnan(ww)] = med if not np.isnan(med) else 0.0

    if float(np.std(ww)) < std_thr:
        return True

    if np.max(np.abs(ww)) > abs_max_thr:
        return True

    return False


def find_label_csv(session_root: Path, clustered_labels_dir: Path | None = None) -> Path | None:
    """
    Find Task_Labels_clustered_6d.csv for a given session:
      clustered_labels_dir / UN_101 / Lab1 / Task_Labels_clustered_6d.csv
    """
    if clustered_labels_dir is None or not clustered_labels_dir.exists():
        return None

    subject = session_root.parent.name   # UN_101
    session = session_root.name          # Lab1 / Lab2 / Wild

    p = clustered_labels_dir / subject / session / "Task_Labels_clustered_6d.csv"
    if p.exists():
        return p
    return None


def load_task_to_label(label_csv: Path) -> Dict[str, int]:
    df = pd.read_csv(label_csv)
    if "Task" not in df.columns or "tlx6_level" not in df.columns:
        raise ValueError(f"{label_csv} must contain columns: Task, tlx6_level")

    mapping = {}
    for _, r in df.iterrows():
        task = str(r["Task"]).strip()
        lvl = str(r["tlx6_level"]).strip().lower()
        if lvl not in LABEL_MAP:
            continue
        mapping[task] = LABEL_MAP[lvl]
    return mapping


def compute_zscore_from_baseline(
    session_root: Path,
    baseline_task: str = "relaxation_video",
    eps: float = 1e-6,
) -> Tuple[float, float] | None:
    """
    Use baseline BVP to compute (mean, std) for subject/session normalization.

    注意：
    这里 baseline_task 用的是真实文件夹名，不做 alias 映射。
    """
    baseline_pkl = session_root / "Preprocessed" / baseline_task / "BVP_filtered.pickle"
    if not baseline_pkl.exists():
        return None

    xb = load_bvp_series(baseline_pkl)
    xb = xb[~np.isnan(xb)]
    if xb.size < 10:
        return None

    mean = float(np.mean(xb))
    std = float(np.std(xb))
    if std < eps:
        return None

    return mean, std


def user_zscore_normalize(
    x: np.ndarray,
    mean: float,
    std: float,
    eps: float = 1e-6,
    clip: float | None = 5.0,
) -> np.ndarray:
    x_norm = (x - mean) / (std + eps)
    if clip is not None:
        x_norm = np.clip(x_norm, -clip, clip)
    return x_norm.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Build UNIVERSE BVP windows.")
    parser.add_argument("--universe-root", type=Path, default=Path(r"./data/UNIVERSE"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--win-s", type=int, default=60)
    parser.add_argument("--hop-s", type=int, default=30)
    parser.add_argument("--label-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    universe_root = args.universe_root
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = universe_root / f"windows_bvp_{args.win_s}s{args.hop_s}s"
    out_dir.mkdir(parents=True, exist_ok=True)

    clustered_labels_dir = args.label_dir or (universe_root / "clustered_labels_6d")

    # BVP sampling rate
    fs = 64
    win_s = args.win_s
    hop_s = args.hop_s
    win = fs * win_s
    hop = fs * hop_s

    sessions = ["Lab1", "Lab2"]

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta_rows: List[dict] = []

    label_cache: Dict[Tuple[str, str], Dict[str, int]] = {}
    zscore_cache: Dict[Tuple[str, str], Tuple[float, float] | None] = {}

    for session in sessions:
        pattern = str(universe_root / "UN_*" / session / "Preprocessed" / "*" / "BVP_filtered.pickle")

        for p in glob.glob(pattern):
            pkl_path = Path(p)
            task_name = pkl_path.parent.name
            label_task_name = normalize_task_name_for_label(task_name)

            session_root = pkl_path.parents[2]   # .../UN_101/Lab1
            subject_id = session_root.parent.name
            session_name = session_root.name

            key = (subject_id, session_name)

            if key not in label_cache:
                label_csv = find_label_csv(session_root, clustered_labels_dir=clustered_labels_dir)
                if label_csv is None:
                    print(f"{subject_id}/{session_name} 没有找到对应标签 CSV，后续该 session 跳过。")
                    label_cache[key] = {}
                else:
                    label_cache[key] = load_task_to_label(label_csv)

            task2label = label_cache[key]
            if label_task_name not in task2label:
                print(
                    f"{subject_id}/{session_name} 的任务 {task_name} "
                    f"(label key={label_task_name}) 没有标签，跳过。"
                )
                continue

            y_task = task2label[label_task_name]

            if key not in zscore_cache:
                zscore_cache[key] = compute_zscore_from_baseline(
                    session_root=session_root,
                    baseline_task="relaxation_video",
                )

            zs = zscore_cache[key]
            if zs is None:
                print(f"{subject_id}/{session_name} 找不到可用 BVP baseline(relaxation_video)，跳过该 session。")
                continue

            mean, std = zs

            x = load_bvp_series(pkl_path)
            x = user_zscore_normalize(x, mean, std, clip=5.0)

            windows, starts = sliding_windows_1d(x, win=win, hop=hop)

            if len(windows) == 0:
                print(
                    f"{subject_id}/{session_name}/{task_name} 的 BVP 数据长度 {len(x)} "
                    f"小于窗口大小 {win}，没有生成任何窗口，跳过该任务。"
                )
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
                    "label_task": label_task_name,
                    "pkl_path": str(pkl_path),
                    "start_idx": int(s_idx),
                    "end_idx": int(s_idx + win),
                    "start_time_s": float(s_idx / fs),
                    "end_time_s": float((s_idx + win) / fs),
                    "fs": fs,
                    "win_s": win_s,
                    "hop_s": hop_s,
                    "modality": "BVP",
                })

    if not X_list:
        raise RuntimeError("没有找到任何有效的 BVP 窗口，请检查路径、标签和 baseline 配置。")

    X = np.stack(X_list, axis=0).astype(np.float32)   # [N, 3840]
    y = np.asarray(y_list, dtype=np.int64)            # [N]
    meta = pd.DataFrame(meta_rows)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    meta.to_csv(out_dir / "meta.csv", index=False)

    print("Saved to:", out_dir)
    print("X shape:", X.shape, "y shape:", y.shape)
    print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))


if __name__ == "__main__":
    main()
