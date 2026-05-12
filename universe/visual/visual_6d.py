import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ========= 配置区 =========
BASE_DIR = os.path.normpath("./data/UNIVERSE")
CLUSTERED_ROOT = os.path.join(BASE_DIR, "clustered_labels_6d")

SESSIONS = ["Lab1", "Lab2", "Wild"]
SUBJECT_RANGE = range(101, 125)

IN_NAME = "Task_Labels_clustered_6d.csv"

TLX_6D_COLS = [
    "Mental Demand",
    "Physical Demand",
    "Temporal Demand",
    "Performance",
    "Effort",
    "Frustration",
]
WEIGHTED_COL = "Weighted Nasa Score"

LEVEL_COL = "tlx6_level"
CLUSTER_COL = "kmeans6_cluster"

OUT_PLOTS_ROOT = os.path.join(CLUSTERED_ROOT, "plots")
DPI = 170
# =========================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_numeric_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def plot_pca_scatter(df: pd.DataFrame, title: str, out_path: str) -> None:
    # 必要列检查
    missing = [c for c in TLX_6D_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing TLX columns: {missing}")
    if LEVEL_COL not in df.columns or CLUSTER_COL not in df.columns:
        raise ValueError(f"Missing clustering cols '{LEVEL_COL}' or '{CLUSTER_COL}'")

    df_num = to_numeric_df(df, TLX_6D_COLS + [WEIGHTED_COL, CLUSTER_COL])

    # 只画 6D 完整且有 level 的点
    valid_mask = df_num[TLX_6D_COLS].notna().all(axis=1) & df_num[LEVEL_COL].notna()
    d = df_num.loc[valid_mask].copy()
    if len(d) == 0:
        return

    X = d[TLX_6D_COLS].to_numpy(dtype=float)

    # 标准化 + PCA
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xz)

    # 计算每个 cluster 在 6D 标准化空间的中心，再投影到 PCA
    clusters = d[CLUSTER_COL].astype(int).to_numpy()
    unique_clusters = np.unique(clusters)
    centers_pca = {}
    for c in unique_clusters:
        idx = (clusters == c)
        center = Xz[idx].mean(axis=0)
        center2d = pca.transform(center.reshape(1, -1))[0]
        centers_pca[int(c)] = center2d

    # 汇总统计：各类数量 + weighted均值（可选）
    counts = d[LEVEL_COL].value_counts().to_dict()
    low_n = int(counts.get("low", 0))
    mid_n = int(counts.get("mid", 0))
    high_n = int(counts.get("high", 0))

    stats_line = f"low={low_n}, mid={mid_n}, high={high_n}"

    weighted_line = ""
    if WEIGHTED_COL in d.columns and d[WEIGHTED_COL].notna().any():
        mw = d.groupby(LEVEL_COL)[WEIGHTED_COL].mean().to_dict()
        # 只展示存在的
        parts = []
        for k in ["low", "mid", "high"]:
            if k in mw and pd.notna(mw[k]):
                parts.append(f"{k}:{mw[k]:.2f}")
        if parts:
            weighted_line = " | mean weighted: " + ", ".join(parts)

    # 画图：按 level 分组 scatter（不指定颜色，让 matplotlib 自动分配）
    plt.figure(figsize=(7.2, 5.2))

    for level in ["low", "mid", "high"]:
        idx = (d[LEVEL_COL].values == level)
        if idx.any():
            plt.scatter(Z[idx, 0], Z[idx, 1], label=level, alpha=0.85)

    # 画中心（X）
    for cid, (cx, cy) in centers_pca.items():
        plt.scatter([cx], [cy], marker="x", s=160, linewidths=3)
        plt.text(cx, cy, f" c{cid}", fontsize=10, va="bottom")

    var = pca.explained_variance_ratio_
    plt.title(f"{title}\n{stats_line}{weighted_line}\nPCA var: PC1={var[0]:.2f}, PC2={var[1]:.2f}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=DPI)
    plt.close()


def main():
    n_ok, n_missing, n_skip, n_err = 0, 0, 0, 0

    for sid in SUBJECT_RANGE:
        subj = f"UN_{sid}"
        for sess in SESSIONS:
            in_path = os.path.join(CLUSTERED_ROOT, subj, sess, IN_NAME)
            if not os.path.isfile(in_path):
                n_missing += 1
                continue

            out_dir = os.path.join(OUT_PLOTS_ROOT, subj)
            out_path = os.path.join(out_dir, f"{sess}_pca_kmeans6d.png")

            try:
                df = pd.read_csv(in_path)
                title = f"UNIVERSE {subj}/{sess} - 6D KMeans (PCA view)"
                plot_pca_scatter(df, title, out_path)
                print(f"[OK]  {subj}/{sess} -> {out_path}")
                n_ok += 1
            except ValueError as e:
                n_skip += 1
                print(f"[SKIP] {subj}/{sess}: {e}")
            except Exception as e:
                n_err += 1
                print(f"[ERR] {subj}/{sess}: {e}")

    print("\n==== Summary ====")
    print(f"OK: {n_ok}")
    print(f"MISSING clustered csv: {n_missing}")
    print(f"SKIP: {n_skip}")
    print(f"ERROR: {n_err}")
    print(f"Plots root: {OUT_PLOTS_ROOT}")


if __name__ == "__main__":
    main()
