import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans

BASE_DIR = os.path.normpath("./data/UNIVERSE")

# 1D
CL1_ROOT = os.path.join(BASE_DIR, "clustered_labels")
CL1_NAME = "Task_Labels_clustered.csv"
SCORE_COL = "Weighted Nasa Score"
LEVEL1_COL = "tlx_level"

# 6D
CL6_ROOT = os.path.join(BASE_DIR, "clustered_labels_6d")
CL6_NAME = "Task_Labels_clustered_6d.csv"
TLX_6D_COLS = ["Mental Demand","Physical Demand","Temporal Demand","Performance","Effort","Frustration"]
LEVEL6_COL = "tlx6_level"
CLUSTER6_COL = "kmeans6_cluster"

SESSIONS = ["Lab1", "Lab2", "Wild"]
SUBJECT_RANGE = range(101, 125)

OUT_DIR = os.path.join(BASE_DIR, "cluster_quality")
OUT_1D_CSV = os.path.join(OUT_DIR, "quality_1d.csv")
OUT_6D_CSV = os.path.join(OUT_DIR, "quality_6d.csv")
OUT_PLOT_DIR = os.path.join(OUT_DIR, "plots")

N_CLUSTERS = 3
DPI = 170


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def compute_unsup_metrics(X: np.ndarray, labels: np.ndarray):
    # 需要至少2个簇且每簇至少1个点；silhouette 还要求样本数 > 簇数
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(X) <= len(uniq):
        return dict(sil=np.nan, ch=np.nan, db=np.nan)

    return dict(
        sil=float(silhouette_score(X, labels)),
        ch=float(calinski_harabasz_score(X, labels)),
        db=float(davies_bouldin_score(X, labels)),
    )


def check_weighted_monotonic(df: pd.DataFrame, level_col: str):
    if SCORE_COL not in df.columns or level_col not in df.columns:
        return dict(monotonic=np.nan, mean_low=np.nan, mean_mid=np.nan, mean_high=np.nan)

    d = df.copy()
    d[SCORE_COL] = _safe_num(d[SCORE_COL])
    d = d.dropna(subset=[SCORE_COL, level_col])

    if len(d) == 0:
        return dict(monotonic=np.nan, mean_low=np.nan, mean_mid=np.nan, mean_high=np.nan)

    means = d.groupby(level_col)[SCORE_COL].mean().to_dict()
    ml = means.get("low", np.nan)
    mm = means.get("mid", np.nan)
    mh = means.get("high", np.nan)

    monotonic = (ml < mm < mh) if (np.isfinite(ml) and np.isfinite(mm) and np.isfinite(mh)) else np.nan
    return dict(monotonic=bool(monotonic) if monotonic is not np.nan else np.nan,
                mean_low=float(ml) if np.isfinite(ml) else np.nan,
                mean_mid=float(mm) if np.isfinite(mm) else np.nan,
                mean_high=float(mh) if np.isfinite(mh) else np.nan)


def plot_weighted_box(df: pd.DataFrame, title: str, out_path: str, level_col: str):
    d = df.copy()
    d[SCORE_COL] = _safe_num(d[SCORE_COL])
    d = d.dropna(subset=[SCORE_COL, level_col])
    if len(d) == 0:
        return

    order = ["low", "mid", "high"]
    data = [d.loc[d[level_col] == k, SCORE_COL].values for k in order]

    plt.figure(figsize=(6.8, 4.6))
    plt.boxplot(data, labels=order, showmeans=True)
    plt.title(title)
    plt.ylabel(SCORE_COL)
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=DPI)
    plt.close()


def stability_6d(df: pd.DataFrame, n_runs=10, seeds=None):
    """
    在 6D 标准化空间上重复跑 KMeans，比较聚类结果一致性（ARI/NMI 的平均值）。
    这一步不依赖你保存的 cluster id，而是重新聚类评估“可重复性”。
    """
    if seeds is None:
        seeds = list(range(n_runs))

    df_num = df.copy()
    for c in TLX_6D_COLS:
        df_num[c] = _safe_num(df_num[c])
    mask = df_num[TLX_6D_COLS].notna().all(axis=1)
    d = df_num.loc[mask].copy()
    if len(d) < N_CLUSTERS:
        return dict(ari_mean=np.nan, nmi_mean=np.nan)

    X = d[TLX_6D_COLS].to_numpy(dtype=float)
    Xz = StandardScaler().fit_transform(X)

    labels_list = []
    for sd in seeds:
        km = KMeans(n_clusters=N_CLUSTERS, random_state=sd, n_init="auto")
        labels_list.append(km.fit_predict(Xz))

    # 两两比较
    aris, nmis = [], []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            aris.append(adjusted_rand_score(labels_list[i], labels_list[j]))
            nmis.append(normalized_mutual_info_score(labels_list[i], labels_list[j]))

    return dict(ari_mean=float(np.mean(aris)) if aris else np.nan,
                nmi_mean=float(np.mean(nmis)) if nmis else np.nan)


def main():
    ensure_dir(OUT_DIR)
    ensure_dir(OUT_PLOT_DIR)

    # ---------- 1D ----------
    rows_1d = []
    for sid in SUBJECT_RANGE:
        subj = f"UN_{sid}"
        for sess in SESSIONS:
            path = os.path.join(CL1_ROOT, subj, sess, CL1_NAME)
            if not os.path.isfile(path):
                continue
            df = pd.read_csv(path)

            df[SCORE_COL] = _safe_num(df.get(SCORE_COL))
            df = df.dropna(subset=[SCORE_COL, LEVEL1_COL])

            if len(df) < N_CLUSTERS:
                continue

            # 1D 空间
            X = df[[SCORE_COL]].to_numpy(dtype=float)
            # 为了计算指标：把 level 转成整数标签
            level_to_id = {"low": 0, "mid": 1, "high": 2}
            y = df[LEVEL1_COL].map(level_to_id).to_numpy()

            metrics = compute_unsup_metrics(X, y)
            mono = check_weighted_monotonic(df, LEVEL1_COL)

            counts = df[LEVEL1_COL].value_counts().to_dict()
            rows_1d.append({
                "subject": subj,
                "session": sess,
                "n": len(df),
                "n_low": int(counts.get("low", 0)),
                "n_mid": int(counts.get("mid", 0)),
                "n_high": int(counts.get("high", 0)),
                **metrics,
                **mono,
            })

            # 画箱线图
            out_plot = os.path.join(OUT_PLOT_DIR, subj, f"{sess}_1d_weighted_box.png")
            plot_weighted_box(df, f"{subj}/{sess} - 1D levels vs Weighted", out_plot, LEVEL1_COL)

    pd.DataFrame(rows_1d).to_csv(OUT_1D_CSV, index=False)

    # ---------- 6D ----------
    rows_6d = []
    for sid in SUBJECT_RANGE:
        subj = f"UN_{sid}"
        for sess in SESSIONS:
            path = os.path.join(CL6_ROOT, subj, sess, CL6_NAME)
            if not os.path.isfile(path):
                continue
            df = pd.read_csv(path)

            # 6D valid
            df_num = df.copy()
            for c in TLX_6D_COLS:
                df_num[c] = _safe_num(df_num[c])
            df_num[SCORE_COL] = _safe_num(df_num.get(SCORE_COL))

            mask = df_num[TLX_6D_COLS].notna().all(axis=1) & df_num[LEVEL6_COL].notna()
            d = df_num.loc[mask].copy()
            if len(d) < N_CLUSTERS:
                continue

            # 6D 标准化空间算指标
            X = d[TLX_6D_COLS].to_numpy(dtype=float)
            Xz = StandardScaler().fit_transform(X)

            level_to_id = {"low": 0, "mid": 1, "high": 2}
            y = d[LEVEL6_COL].map(level_to_id).to_numpy()

            metrics = compute_unsup_metrics(Xz, y)
            mono = check_weighted_monotonic(d, LEVEL6_COL)
            stab = stability_6d(df)  # 用原 df 重新聚类做稳定性

            counts = d[LEVEL6_COL].value_counts().to_dict()
            rows_6d.append({
                "subject": subj,
                "session": sess,
                "n": len(d),
                "n_low": int(counts.get("low", 0)),
                "n_mid": int(counts.get("mid", 0)),
                "n_high": int(counts.get("high", 0)),
                **metrics,
                **mono,
                **stab,
            })

            out_plot = os.path.join(OUT_PLOT_DIR, subj, f"{sess}_6d_weighted_box.png")
            plot_weighted_box(d, f"{subj}/{sess} - 6D levels vs Weighted", out_plot, LEVEL6_COL)

    pd.DataFrame(rows_6d).to_csv(OUT_6D_CSV, index=False)

    print("Saved:")
    print(" -", OUT_1D_CSV)
    print(" -", OUT_6D_CSV)
    print(" - plots under:", OUT_PLOT_DIR)


if __name__ == "__main__":
    main()
