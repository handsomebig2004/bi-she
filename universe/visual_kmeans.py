import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 配置区 =========
BASE_DIR = os.path.normpath("./data/UNIVERSE")
CLUSTERED_ROOT = os.path.join(BASE_DIR, "clustered_labels")

SESSIONS = ["Lab1", "Lab2", "Wild"]
SUBJECT_RANGE = range(101, 125)  # 101..124

# 你的聚类输出文件名规则（和前面脚本一致）
IN_NAME = "Task_Labels_clustered.csv"

SCORE_COL = "Weighted Nasa Score"
LEVEL_COL = "tlx_level"
CENTER_COL = "kmeans_center"

OUT_PLOTS_ROOT = os.path.join(CLUSTERED_ROOT, "plots")
BINS = 10  # 直方图 bins，可改 15/20
DPI = 160
# =========================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def plot_one(df: pd.DataFrame, title: str, out_path: str) -> None:
    scores = safe_numeric(df[SCORE_COL])
    scores = scores.dropna()

    if len(scores) == 0:
        return

    # centers：优先从 kmeans_center 列恢复（会重复），取 unique 后排序
    centers = None
    if CENTER_COL in df.columns:
        c = safe_numeric(df[CENTER_COL]).dropna().unique()
        if len(c) > 0:
            centers = np.sort(c)

    # 统计各类数量（如果有 tlx_level）
    counts_text = ""
    if LEVEL_COL in df.columns:
        counts = df[LEVEL_COL].value_counts(dropna=False).to_dict()
        low_n = int(counts.get("low", 0))
        mid_n = int(counts.get("mid", 0))
        high_n = int(counts.get("high", 0))
        nan_n = int(counts.get(np.nan, 0))  # 可能抓不到
        counts_text = f" | low={low_n}, mid={mid_n}, high={high_n}"

    # 画图
    plt.figure(figsize=(7.2, 4.4))
    plt.hist(scores.values, bins=BINS)

    # 画中心线
    if centers is not None and len(centers) > 0:
        for c in centers:
            plt.axvline(float(c), linewidth=2)
        centers_str = ", ".join([f"{float(c):.2f}" for c in centers])
        subtitle = f"centers: [{centers_str}]"
    else:
        subtitle = "centers: (missing)"

    plt.title(f"{title}{counts_text}\n{subtitle}")
    plt.xlabel(SCORE_COL)
    plt.ylabel("count")
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

            try:
                df = pd.read_csv(in_path)
                if SCORE_COL not in df.columns:
                    n_skip += 1
                    print(f"[SKIP] {subj}/{sess}: missing column '{SCORE_COL}'")
                    continue

                title = f"UNIVERSE {subj} / {sess} - 1D KMeans TLX"
                out_path = os.path.join(OUT_PLOTS_ROOT, subj, f"{sess}_tlx_kmeans.png")
                plot_one(df, title, out_path)
                print(f"[OK]  {subj}/{sess} -> {out_path}")
                n_ok += 1
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
