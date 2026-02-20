import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


def find_trial_roots(base_path: str):
    roots = []
    for root, _, _ in os.walk(base_path):
        ecg = os.path.join(root, "inf_ecg.csv")
        gsr = os.path.join(root, "inf_gsr.csv")
        ppg = os.path.join(root, "inf_ppg.csv")
        if os.path.exists(ecg) and os.path.exists(gsr) and os.path.exists(ppg):
            roots.append(root.replace("\\", "/"))
    return sorted(roots)


def infer_subject_id(root: str):
    # 你之前用 root[-3:]，这里保留兼容逻辑
    tail_digits = "".join([c for c in root if c.isdigit()])
    return tail_digits[-3:] if len(tail_digits) >= 3 else root.split("/")[-1]


def read_weighted_score_per_trial(tlx_csv_path: str):
    """
    读取每个 trial 的 weighted score（Adjusted rate(Weight x Raw) 下一行）
    返回：
      trial_cols: ['Trial 1: 0_back', ...]
      y_per_trial: np.ndarray shape (n_trials,)
    """
    df = pd.read_csv(tlx_csv_path)

    # trial 列：以 Trial 开头的列
    trial_cols = [c for c in df.columns if str(c).strip().lower().startswith("trial")]
    if len(trial_cols) == 0:
        raise ValueError(f"No Trial columns found in {tlx_csv_path}")

    # 找到包含 'Adjusted rate' 的那一行，然后取下一行作为 adjusted values
    # 你的文件里它在 df.iloc[6,1]，值在 df.iloc[7, trial_cols]
    hit_row = None
    for r in range(len(df)):
        row_str = " ".join([str(x).lower() for x in df.iloc[r].tolist()])
        if "adjusted" in row_str and "weight" in row_str:
            hit_row = r
            break

    if hit_row is None or hit_row + 1 >= len(df):
        # 兜底：直接取最后一行当作 adjusted values
        adjusted_row = df.iloc[-1]
    else:
        adjusted_row = df.iloc[hit_row + 1]

    y = pd.to_numeric(adjusted_row[trial_cols], errors="coerce").to_numpy(dtype=np.float32)

    # 清理 NaN
    if np.any(np.isnan(y)):
        raise ValueError(
            f"Found NaN in weighted scores. Check {tlx_csv_path}. "
            f"Parsed values: {y}"
        )

    return trial_cols, y


def build_manifest(base_path: str, rating_path: str):
    """
    每行一个 trial：
      subject_id, root, trial_id, trial_name, y_cont
    """
    rows = []
    roots = find_trial_roots(base_path)

    for root in roots:
        subj = infer_subject_id(root)

        tlx_path = os.path.join(rating_path, subj, "NASA_TLX.csv").replace("\\", "/")
        if not os.path.exists(tlx_path):
            # 没标签就跳过
            continue

        trial_cols, y_trials = read_weighted_score_per_trial(tlx_path)

        # 这个 root 里每个 csv 的 trial 数（转置后的行数）
        ecg_trials = pd.read_csv(os.path.join(root, "inf_ecg.csv")).to_numpy().T
        gsr_trials = pd.read_csv(os.path.join(root, "inf_gsr.csv")).to_numpy().T
        ppg_trials = pd.read_csv(os.path.join(root, "inf_ppg.csv")).to_numpy().T
        n_trials_sig = min(len(ecg_trials), len(gsr_trials), len(ppg_trials))
        n_trials_lbl = len(y_trials)
        n = min(n_trials_sig, n_trials_lbl)

        for t in range(n):
            rows.append({
                "subject_id": subj,
                "root": root,
                "trial_id": int(t),
                "trial_name": str(trial_cols[t]),
                "y_cont": float(y_trials[t]),
            })

    if not rows:
        raise RuntimeError("No samples found. Check base_path/rating_path or label paths.")
    return pd.DataFrame(rows)


def fit_kmeans_on_train_y(y_train, random_state=0, n_init=50):
    """
    只在训练集 y 上 fit KMeans(3)，并把簇按中心从小到大映射为：
      0=low, 1=mid, 2=high
    """
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    km = KMeans(n_clusters=3, n_init=n_init, random_state=random_state)
    km.fit(y_train)

    centers = km.cluster_centers_.reshape(-1)
    order = np.argsort(centers)  # 小->大
    cluster2class = {int(cluster_id): int(rank) for rank, cluster_id in enumerate(order)}
    return km, centers, cluster2class


def apply_kmeans_labels(km: KMeans, cluster2class: dict, y):
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    clusters = km.predict(y).astype(int)
    cls = np.vectorize(cluster2class.get)(clusters).astype(np.int64)
    return cls


def split_subjectwise(df, train_ratio=0.7, val_ratio=0.15, random_state=0):
    subjects = df["subject_id"].unique()
    train_subj, rest_subj = train_test_split(
        subjects, test_size=(1 - train_ratio), random_state=random_state
    )
    val_size = val_ratio / (1 - train_ratio)
    val_subj, test_subj = train_test_split(
        rest_subj, test_size=(1 - val_size), random_state=random_state
    )

    train_df = df[df["subject_id"].isin(train_subj)].copy()
    val_df   = df[df["subject_id"].isin(val_subj)].copy()
    test_df  = df[df["subject_id"].isin(test_subj)].copy()
    return train_df, val_df, test_df


def split_trialwise(df, train_ratio=0.7, val_ratio=0.15, random_state=0):
    train_df, rest_df = train_test_split(df, test_size=(1 - train_ratio), random_state=random_state)
    val_size = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(rest_df, test_size=(1 - val_size), random_state=random_state)
    return train_df.copy(), val_df.copy(), test_df.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_path", type=str, default="data/MAUS/Data/Normalized_data/")
    ap.add_argument("--rating_path", type=str, default="data/MAUS/Subjective_rating/")
    ap.add_argument("--out_dir", type=str, default="splits_maus")
    ap.add_argument("--split_mode", type=str, choices=["subject", "trial"], default="subject")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = build_manifest(args.base_path, args.rating_path)
    print("Total trials:", len(df), "| Subjects:", df["subject_id"].nunique())
    print("y_cont range:", float(df["y_cont"].min()), "->", float(df["y_cont"].max()))

    if args.split_mode == "subject":
        train_df, val_df, test_df = split_subjectwise(df, args.train_ratio, args.val_ratio, args.seed)
    else:
        train_df, val_df, test_df = split_trialwise(df, args.train_ratio, args.val_ratio, args.seed)

    print("Split sizes:", len(train_df), len(val_df), len(test_df))

    # 只在训练集 y 上 fit KMeans
    km, centers, cluster2class = fit_kmeans_on_train_y(train_df["y_cont"].to_numpy(), random_state=args.seed)

    train_df["y_cls"] = apply_kmeans_labels(km, cluster2class, train_df["y_cont"].to_numpy())
    val_df["y_cls"]   = apply_kmeans_labels(km, cluster2class, val_df["y_cont"].to_numpy())
    test_df["y_cls"]  = apply_kmeans_labels(km, cluster2class, test_df["y_cont"].to_numpy())

    train_df.to_csv(os.path.join(args.out_dir, "splits_train.csv"), index=False)
    val_df.to_csv(os.path.join(args.out_dir, "splits_val.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "splits_test.csv"), index=False)

    mapping = {
        "split_mode": args.split_mode,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "centers_raw": centers.tolist(),
        "cluster2class": {str(k): int(v) for k, v in cluster2class.items()},
        "class_names": {"0": "low", "1": "mid", "2": "high"},
    }
    with open(os.path.join(args.out_dir, "kmeans_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    def show_counts(name, d):
        c = d["y_cls"].value_counts().sort_index()
        print(f"{name} class counts:", {int(k): int(v) for k, v in c.items()})

    print("KMeans centers:", centers, " (raw, unordered)")
    print("cluster2class:", cluster2class, " -> 0/1/2 = low/mid/high")
    show_counts("train", train_df)
    show_counts("val", val_df)
    show_counts("test", test_df)

    print("\nSaved to:", args.out_dir)
    print("CSV columns: subject_id, root, trial_id, trial_name, y_cont, y_cls")


if __name__ == "__main__":
    main()
