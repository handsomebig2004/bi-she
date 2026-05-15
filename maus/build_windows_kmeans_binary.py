import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def parse_trial_level(col: str) -> str:
    text = col.lower().replace("_", "")
    if "0back" in text or "0 back" in text:
        return "0_back"
    if "2back" in text or "2 back" in text:
        return "2_back"
    if "3back" in text or "3 back" in text:
        return "3_back"
    return col.strip()


def resample_1d_linear(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == target_len:
        return x.astype(np.float32)
    if len(x) <= 1:
        return np.zeros(target_len, dtype=np.float32)

    old_idx = np.linspace(0.0, 1.0, num=len(x), dtype=np.float32)
    new_idx = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(new_idx, old_idx, x).astype(np.float32)


def zscore_clip(x: np.ndarray, clip: float = 5.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = float(np.nanmean(x))
    std = float(np.nanstd(x))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    y = (x - mean) / std
    y = np.nan_to_num(y, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(y, -clip, clip).astype(np.float32)


def sliding_starts(n: int, win: int, hop: int) -> np.ndarray:
    if n < win:
        return np.empty((0,), dtype=np.int64)
    return np.arange(1 + (n - win) // hop, dtype=np.int64) * hop


def is_bad_window(w: np.ndarray, nan_ratio_thr: float = 0.10, std_thr: float = 1e-6) -> bool:
    if np.isnan(w).mean() > nan_ratio_thr:
        return True
    ww = np.nan_to_num(w, nan=float(np.nanmedian(w)) if np.isfinite(np.nanmedian(w)) else 0.0)
    return float(np.std(ww)) < std_thr


def load_nasa_scores(nasa_path: Path) -> list[dict]:
    df = pd.read_csv(nasa_path)
    trial_cols = [c for c in df.columns if c.lower().startswith("trial")]
    if len(trial_cols) == 0:
        raise ValueError(f"No trial columns found in {nasa_path}")

    # MAUS NASA_TLX.csv stores weighted/adjusted overall scores in the last row.
    score_row = df.iloc[-1]
    rows = []
    for trial_idx, col in enumerate(trial_cols):
        y_cont = pd.to_numeric(score_row[col], errors="coerce")
        if pd.isna(y_cont):
            continue
        rows.append(
            {
                "trial_idx": trial_idx,
                "trial_col": col,
                "trial_level": parse_trial_level(col),
                "y_cont": float(y_cont),
            }
        )
    return rows


def collect_trials(raw_root: Path, rating_root: Path) -> pd.DataFrame:
    rows = []
    for subject_dir in sorted(raw_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name
        gsr_path = subject_dir / "inf_gsr.csv"
        ppg_path = subject_dir / "inf_ppg.csv"
        nasa_path = rating_root / subject / "NASA_TLX.csv"
        if not (gsr_path.exists() and ppg_path.exists() and nasa_path.exists()):
            continue

        try:
            for row in load_nasa_scores(nasa_path):
                row.update(
                    {
                        "subject": subject,
                        "gsr_path": str(gsr_path),
                        "ppg_path": str(ppg_path),
                        "nasa_path": str(nasa_path),
                    }
                )
                rows.append(row)
        except Exception as exc:
            print(f"[WARN] Skip {subject}: {type(exc).__name__}: {exc}")

    if not rows:
        raise RuntimeError("No MAUS trial scores found. Check --raw-root and --rating-root.")
    return pd.DataFrame(rows)


def add_kmeans_binary_labels(trials: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    scores = trials["y_cont"].to_numpy(dtype=np.float32).reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=50, random_state=random_state)
    clusters = km.fit_predict(scores)
    centers = km.cluster_centers_.reshape(-1)
    order = np.argsort(centers)
    cluster_to_label = {int(order[0]): 0, int(order[1]): 1}

    out = trials.copy()
    out["kmeans_cluster"] = clusters.astype(np.int64)
    out["y_binary"] = np.asarray([cluster_to_label[int(c)] for c in clusters], dtype=np.int64)
    out["kmeans_center"] = np.asarray([centers[int(c)] for c in clusters], dtype=np.float32)
    return out, centers


def build_windows(
    trials: pd.DataFrame,
    out_dir: Path,
    raw_fs: int,
    win_s: int,
    hop_s: int,
    gsr_target_fs: int,
    ppg_target_fs: int,
):
    win = raw_fs * win_s
    hop = raw_fs * hop_s
    gsr_target_len = gsr_target_fs * win_s
    ppg_target_len = ppg_target_fs * win_s

    X_gsr, X_ppg, y, meta_rows = [], [], [], []
    grouped = trials.groupby("subject", sort=True)

    for subject, sub_trials in grouped:
        gsr_path = Path(sub_trials["gsr_path"].iloc[0])
        ppg_path = Path(sub_trials["ppg_path"].iloc[0])
        print(f"[{subject}] reading signals")
        gsr_df = pd.read_csv(gsr_path)
        ppg_df = pd.read_csv(ppg_path)

        for _, tr in sub_trials.iterrows():
            trial_idx = int(tr["trial_idx"])
            if trial_idx >= gsr_df.shape[1] or trial_idx >= ppg_df.shape[1]:
                print(f"[WARN] {subject} trial {trial_idx} missing signal column")
                continue

            gsr_raw = pd.to_numeric(gsr_df.iloc[:, trial_idx], errors="coerce").to_numpy(dtype=np.float32)
            ppg_raw = pd.to_numeric(ppg_df.iloc[:, trial_idx], errors="coerce").to_numpy(dtype=np.float32)
            n = min(len(gsr_raw), len(ppg_raw))
            gsr_raw = zscore_clip(gsr_raw[:n])
            ppg_raw = zscore_clip(ppg_raw[:n])

            starts = sliding_starts(n, win, hop)
            if len(starts) == 0:
                print(f"[WARN] {subject} trial {trial_idx} shorter than {win_s}s")
                continue

            for s_idx in starts:
                e_idx = int(s_idx + win)
                gsr_w = gsr_raw[s_idx:e_idx]
                ppg_w = ppg_raw[s_idx:e_idx]
                if is_bad_window(gsr_w) or is_bad_window(ppg_w):
                    continue

                X_gsr.append(resample_1d_linear(gsr_w, gsr_target_len))
                X_ppg.append(resample_1d_linear(ppg_w, ppg_target_len))
                y.append(int(tr["y_binary"]))
                meta_rows.append(
                    {
                        "subject": subject,
                        "trial_idx": trial_idx,
                        "trial_col": tr["trial_col"],
                        "trial_level": tr["trial_level"],
                        "y_cont": float(tr["y_cont"]),
                        "y_binary": int(tr["y_binary"]),
                        "kmeans_cluster": int(tr["kmeans_cluster"]),
                        "kmeans_center": float(tr["kmeans_center"]),
                        "start_idx_raw": int(s_idx),
                        "end_idx_raw": e_idx,
                        "start_time_s": float(s_idx / raw_fs),
                        "end_time_s": float(e_idx / raw_fs),
                        "raw_fs": raw_fs,
                        "win_s": win_s,
                        "hop_s": hop_s,
                        "gsr_target_fs": gsr_target_fs,
                        "ppg_target_fs": ppg_target_fs,
                    }
                )

    if not X_gsr:
        raise RuntimeError("No valid MAUS windows produced.")

    out_dir.mkdir(parents=True, exist_ok=True)
    X_gsr_arr = np.stack(X_gsr, axis=0).astype(np.float32)
    X_ppg_arr = np.stack(X_ppg, axis=0).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    meta = pd.DataFrame(meta_rows)

    np.save(out_dir / "X_gsr.npy", X_gsr_arr)
    np.save(out_dir / "X_ppg.npy", X_ppg_arr)
    np.save(out_dir / "y.npy", y_arr)
    meta.to_csv(out_dir / "meta.csv", index=False)

    print("Saved:", out_dir)
    print("X_gsr:", X_gsr_arr.shape)
    print("X_ppg:", X_ppg_arr.shape)
    print("y:", y_arr.shape)
    print("Subjects:", meta["subject"].nunique())
    print("Class counts:", dict(zip(*np.unique(y_arr, return_counts=True))))


def parse_args():
    parser = argparse.ArgumentParser(description="Build MAUS GSR/PPG windows with KMeans-2 NASA labels.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/MAUS/Data/Raw_data"))
    parser.add_argument("--rating-root", type=Path, default=Path("data/MAUS/Subjective_rating"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/MAUS/windows_gsr_ppg_30s15s_kmeans2"))
    parser.add_argument("--raw-fs", type=int, default=256)
    parser.add_argument("--win-s", type=int, default=30)
    parser.add_argument("--hop-s", type=int, default=15)
    parser.add_argument("--gsr-target-fs", type=int, default=4)
    parser.add_argument("--ppg-target-fs", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    trials = collect_trials(args.raw_root, args.rating_root)
    trials, centers = add_kmeans_binary_labels(trials, random_state=args.random_state)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.out_dir / "trial_labels_kmeans2.csv", index=False)
    pd.DataFrame(
        {
            "cluster": [0, 1],
            "center": centers,
            "binary_label": [0 if c == int(np.argmin(centers)) else 1 for c in range(2)],
        }
    ).to_csv(args.out_dir / "kmeans2_centers.csv", index=False)

    print("Trial labels:", len(trials), "subjects:", trials["subject"].nunique())
    print("KMeans centers:", centers.tolist())
    print("Trial class counts:", trials["y_binary"].value_counts().sort_index().to_dict())

    build_windows(
        trials=trials,
        out_dir=args.out_dir,
        raw_fs=args.raw_fs,
        win_s=args.win_s,
        hop_s=args.hop_s,
        gsr_target_fs=args.gsr_target_fs,
        ppg_target_fs=args.ppg_target_fs,
    )


if __name__ == "__main__":
    main()
