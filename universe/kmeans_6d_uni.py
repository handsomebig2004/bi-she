import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ========= 配置区 =========
BASE_DIR = os.path.normpath("./data/UNIVERSE")
SESSIONS = ["Lab1", "Lab2", "Wild"]
SUBJECT_RANGE = range(101, 125)  # 101..124
INPUT_NAME = "Task_Labels.csv"

OUT_ROOT = os.path.join(BASE_DIR, "clustered_labels_6d")
OUT_NAME = "Task_Labels_clustered_6d.csv"

# 6D 维度（按你文件的列名）
TLX_6D_COLS = [
    "Mental Demand",
    "Physical Demand",
    "Temporal Demand",
    "Performance",
    "Effort",
    "Frustration",
]

# 用于给簇排序的参考列（建议用加权总分）
WEIGHTED_COL = "Weighted Nasa Score"

N_CLUSTERS = 3
RANDOM_STATE = 0
# =========================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_numeric_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def assign_levels_by_weighted_mean(df_valid: pd.DataFrame, clusters: np.ndarray) -> dict[int, str]:
    """
    用每个簇内 Weighted Nasa Score 的均值来排序簇 -> low/mid/high
    返回 cluster_id -> level 的映射
    """
    if WEIGHTED_COL not in df_valid.columns:
        # 如果没有 weighted 列，就退化为按簇中心在 6D 空间的范数排序（不如 weighted 直观）
        # 这里先抛错更安全
        raise ValueError(f"Missing column '{WEIGHTED_COL}' for ordering clusters.")

    w = pd.to_numeric(df_valid[WEIGHTED_COL], errors="coerce")
    # 若 weighted 缺失很多，也可以改成用 6D 的某种指标排序
    if w.isna().all():
        raise ValueError(f"Column '{WEIGHTED_COL}' is all NaN; cannot order clusters.")

    tmp = df_valid.copy()
    tmp["_cluster"] = clusters
    tmp["_weighted"] = w.values
    mean_w = tmp.groupby("_cluster")["_weighted"].mean()

    # 按均值从小到大排序簇 id
    ordered = mean_w.sort_values().index.tolist()  # low->high
    mapping = {int(ordered[0]): "low", int(ordered[1]): "mid", int(ordered[2]): "high"}
    return mapping


def process_one_csv(in_path: str, out_path: str) -> dict:
    df = pd.read_csv(in_path)

    missing = [c for c in TLX_6D_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing TLX columns: {missing}")

    df_num = to_numeric_df(df, TLX_6D_COLS + ([WEIGHTED_COL] if WEIGHTED_COL in df.columns else []))

    # 6D 全部非空才参与聚类
    valid_mask = df_num[TLX_6D_COLS].notna().all(axis=1)
    n_valid = int(valid_mask.sum())

    if n_valid < N_CLUSTERS:
        # 不足以聚类：写出空标签
        df["kmeans6_cluster"] = np.nan
        df["tlx6_level"] = np.nan
        ensure_dir(os.path.dirname(out_path))
        df.to_csv(out_path, index=False)
        return {"status": "skipped_insufficient_samples", "n_total": len(df), "n_valid": n_valid}

    X = df_num.loc[valid_mask, TLX_6D_COLS].to_numpy(dtype=float)

    # 标准化 + KMeans
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
    clusters = km.fit_predict(Xz)

    # 用 weighted 均值排序簇 -> low/mid/high
    df_valid = df_num.loc[valid_mask].copy()
    mapping = assign_levels_by_weighted_mean(df_valid, clusters)
    levels = np.array([mapping[int(c)] for c in clusters], dtype=object)

    # 写回原 df
    df["kmeans6_cluster"] = np.nan
    df["tlx6_level"] = np.nan
    df.loc[valid_mask, "kmeans6_cluster"] = clusters
    df.loc[valid_mask, "tlx6_level"] = levels

    ensure_dir(os.path.dirname(out_path))
    df.to_csv(out_path, index=False)

    # 给日志：各簇 weighted 均值
    if WEIGHTED_COL in df_num.columns:
        w = df_valid[WEIGHTED_COL]
        tmp = pd.DataFrame({"cluster": clusters, "weighted": w.values})
        mean_w = tmp.groupby("cluster")["weighted"].mean().to_dict()
    else:
        mean_w = None

    return {
        "status": "ok",
        "n_total": len(df),
        "n_valid": n_valid,
        "cluster_to_level": mapping,
        "mean_weighted_by_cluster": mean_w,
    }


def main():
    n_ok, n_skip, n_missing, n_err = 0, 0, 0, 0

    for sid in SUBJECT_RANGE:
        subj = f"UN_{sid}"
        for sess in SESSIONS:
            in_path = os.path.join(BASE_DIR, subj, sess, INPUT_NAME)
            if not os.path.isfile(in_path):
                n_missing += 1
                continue

            out_dir = os.path.join(OUT_ROOT, subj, sess)
            out_path = os.path.join(out_dir, OUT_NAME)

            try:
                info = process_one_csv(in_path, out_path)
                if info["status"] == "ok":
                    n_ok += 1
                    mapping = info["cluster_to_level"]
                    mean_w = info["mean_weighted_by_cluster"]
                    if mean_w is not None:
                        mw_str = ", ".join([f"{k}:{v:.2f}" for k, v in sorted(mean_w.items())])
                    else:
                        mw_str = "N/A"
                    print(f"[OK]  {subj}/{sess} -> mean weighted by cluster: {mw_str} -> {out_path}")
                else:
                    n_skip += 1
                    print(f"[SKIP] {subj}/{sess} (valid={info['n_valid']}) -> {out_path}")
            except Exception as e:
                n_err += 1
                print(f"[ERR] {subj}/{sess}: {e}")

    print("\n==== Summary ====")
    print(f"OK: {n_ok}")
    print(f"SKIP (insufficient samples): {n_skip}")
    print(f"MISSING Task_Labels.csv: {n_missing}")
    print(f"ERROR: {n_err}")
    print(f"Output root: {OUT_ROOT}")


if __name__ == "__main__":
    main()
