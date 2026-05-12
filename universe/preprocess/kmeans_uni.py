import os
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ========= 配置区 =========
BASE_DIR = os.path.normpath("./data/UNIVERSE")
SESSIONS = ["Lab1", "Lab2", "Wild"]
SUBJECT_RANGE = range(101, 125)  # 101..124
INPUT_NAME = "Task_Labels.csv"

OUT_ROOT = os.path.join(BASE_DIR, "clustered_labels")
OUT_NAME_SUFFIX = "_clustered.csv"

SCORE_COL = "Weighted Nasa Score"  # 你这个 CSV 的列名
N_CLUSTERS = 3
RANDOM_STATE = 0
# =========================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def kmeans_1d_to_levels(scores: np.ndarray, n_clusters=3, random_state=0):
    """
    scores: shape (n,)
    returns:
      cluster_id (n,), centers (k,), level (n,) where level in {"low","mid","high"}
    """
    X = scores.reshape(-1, 1)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster = km.fit_predict(X)
    centers = km.cluster_centers_.flatten()

    order = np.argsort(centers)  # smallest->largest
    mapping = {int(order[0]): "low", int(order[1]): "mid", int(order[2]): "high"}
    levels = np.array([mapping[int(c)] for c in cluster], dtype=object)
    return cluster, centers, levels


def process_one_csv(in_path: str, out_path: str) -> dict:
    """
    读取一个 Task_Labels.csv，按 Weighted Nasa Score 做 1D KMeans(3)，写出带标签的新文件。
    返回一些统计信息，便于打印日志。
    """
    df = pd.read_csv(in_path)

    if SCORE_COL not in df.columns:
        raise ValueError(f"Missing column '{SCORE_COL}' in {in_path}")

    # 转 float，去掉无法解析的
    scores = pd.to_numeric(df[SCORE_COL], errors="coerce")
    valid_mask = scores.notna()

    if valid_mask.sum() < N_CLUSTERS:
        # 样本数不足以聚 N 类：直接跳过或退化处理
        # 这里选择：写出文件，但 tlx_level 全部设为 NaN，并记录原因
        df["kmeans_cluster"] = np.nan
        df["kmeans_center"] = np.nan
        df["tlx_level"] = np.nan
        ensure_dir(os.path.dirname(out_path))
        df.to_csv(out_path, index=False)
        return {
            "status": "skipped_insufficient_samples",
            "n_total": len(df),
            "n_valid": int(valid_mask.sum()),
            "centers": None,
        }

    # 只对有效行做聚类，然后再填回去
    valid_scores = scores[valid_mask].to_numpy(dtype=float)
    cluster, centers, levels = kmeans_1d_to_levels(
        valid_scores, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE
    )

    df["kmeans_cluster"] = np.nan
    df["kmeans_center"] = np.nan
    df["tlx_level"] = np.nan

    # 填回有效位置
    df.loc[valid_mask, "kmeans_cluster"] = cluster
    # 每行对应的簇中心值也写进去，方便你看边界
    df.loc[valid_mask, "kmeans_center"] = [centers[int(c)] for c in cluster]
    df.loc[valid_mask, "tlx_level"] = levels

    ensure_dir(os.path.dirname(out_path))
    df.to_csv(out_path, index=False)

    return {
        "status": "ok",
        "n_total": len(df),
        "n_valid": int(valid_mask.sum()),
        "centers": centers.tolist(),
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
            out_path = os.path.join(out_dir, INPUT_NAME.replace(".csv", OUT_NAME_SUFFIX))

            try:
                info = process_one_csv(in_path, out_path)
                if info["status"] == "ok":
                    n_ok += 1
                    centers_str = ", ".join([f"{c:.2f}" for c in info["centers"]])
                    print(f"[OK] {subj}/{sess} -> centers: [{centers_str}] -> {out_path}")
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
