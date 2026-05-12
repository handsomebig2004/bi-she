import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ========= 路径配置 =========
BASE_DIR = os.path.normpath("./data/UNIVERSE")

QUALITY_DIR = os.path.join(BASE_DIR, "cluster_quality")
Q1_PATH = os.path.join(QUALITY_DIR, "quality_1d.csv")
Q6_PATH = os.path.join(QUALITY_DIR, "quality_6d.csv")

CL1_ROOT = os.path.join(BASE_DIR, "clustered_labels")
CL6_ROOT = os.path.join(BASE_DIR, "clustered_labels_6d")

CL1_NAME = "Task_Labels_clustered.csv"
CL6_NAME = "Task_Labels_clustered_6d.csv"

OUT_SUMMARY = os.path.join(QUALITY_DIR, "comparison_summary.csv")

SESSIONS = ["Lab1", "Lab2", "Wild"]
SUBJECT_RANGE = range(101, 125)

# ========= 列名配置 =========
SCORE_COL = "Weighted Nasa Score"
MONO_COL = "monotonic"
MEAN_LOW = "mean_low"
MEAN_MID = "mean_mid"
MEAN_HIGH = "mean_high"

# 6D 稳定性列（你之前脚本产出的）
ARI_COL = "ari_mean"
NMI_COL = "nmi_mean"

# ========= 稳定性参数 =========
N_CLUSTERS = 3
N_RUNS_STAB = 15
# ============================


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _safe_num(x):
    return pd.to_numeric(x, errors="coerce")


def compute_gap(df: pd.DataFrame):
    """
    gap = min(mean_mid-mean_low, mean_high-mean_mid)
    """
    ml = df.get(MEAN_LOW)
    mm = df.get(MEAN_MID)
    mh = df.get(MEAN_HIGH)
    ml = _safe_num(ml)
    mm = _safe_num(mm)
    mh = _safe_num(mh)
    gap1 = mm - ml
    gap2 = mh - mm
    return np.minimum(gap1, gap2)

def summarize_method(df: pd.DataFrame, method_name: str,
                     ari_col: str | None = None, nmi_col: str | None = None):
    """
    df 是 quality 表
    """
    d = df.copy()

    # monotonic True 比例
    if MONO_COL in d.columns:
        mono = d[MONO_COL]
        # 兼容字符串/布尔
        mono_bool = mono.map(lambda x: True if str(x).lower() == "true" else (False if str(x).lower() == "false" else np.nan))
        mono_rate = float(np.nanmean(mono_bool.astype(float))) if mono_bool.notna().any() else np.nan
    else:
        mono_rate = np.nan

    # gap
    if all(c in d.columns for c in [MEAN_LOW, MEAN_MID, MEAN_HIGH]):
        gap = compute_gap(d)
        gap_med = float(np.nanmedian(gap)) if np.isfinite(gap).any() else np.nan
        gap_mean = float(np.nanmean(gap)) if np.isfinite(gap).any() else np.nan
    else:
        gap_med, gap_mean = np.nan, np.nan

    # unsup metrics
    sil_med = float(np.nanmedian(_safe_num(d.get("sil")))) if "sil" in d.columns else np.nan
    db_med = float(np.nanmedian(_safe_num(d.get("db")))) if "db" in d.columns else np.nan
    ch_med = float(np.nanmedian(_safe_num(d.get("ch")))) if "ch" in d.columns else np.nan

    # stability
    ari_med = float(np.nanmedian(_safe_num(d.get(ari_col)))) if (ari_col and ari_col in d.columns) else np.nan
    nmi_med = float(np.nanmedian(_safe_num(d.get(nmi_col)))) if (nmi_col and nmi_col in d.columns) else np.nan

    # coverage
    n_med = float(np.nanmedian(_safe_num(d.get("n")))) if "n" in d.columns else np.nan
    n_rows = int(len(d))

    return {
        "method": method_name,
        "rows": n_rows,
        "n_median": n_med,
        "monotonic_rate": mono_rate,
        "gap_median": gap_med,
        "gap_mean": gap_mean,
        "sil_median": sil_med,
        "db_median": db_med,
        "ch_median": ch_med,
        "ari_median": ari_med,
        "nmi_median": nmi_med,
    }

def main():
    ensure_dir(QUALITY_DIR)

    if not os.path.isfile(Q1_PATH):
        raise FileNotFoundError(f"Missing {Q1_PATH}")
    if not os.path.isfile(Q6_PATH):
        raise FileNotFoundError(f"Missing {Q6_PATH}")

    q1 = pd.read_csv(Q1_PATH)
    q6 = pd.read_csv(Q6_PATH)

    # 汇总
    sum1 = summarize_method(q1, "1D", ari_col=ARI_COL, nmi_col=NMI_COL)
    sum6 = summarize_method(q6, "6D", ari_col=ARI_COL, nmi_col=NMI_COL)

    summary_df = pd.DataFrame([sum1, sum6])
    summary_df.to_csv(OUT_SUMMARY, index=False)

    # better = decide_better(sum1, sum6)

    print("Saved:")
    print(" -", OUT_SUMMARY)
    print("\n--- Summary ---")
    print(summary_df.to_string(index=False))
    # print(f"\n推荐: {better}")


if __name__ == "__main__":
    main()
