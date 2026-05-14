from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


OUT_DIR = Path("results/cnn_collected")
SUMMARY_OUT = OUT_DIR / "cnn_summary_all.csv"
PER_SUBJECT_OUT = OUT_DIR / "cnn_per_subject_all.csv"
SKIPPED_OUT = OUT_DIR / "cnn_skipped_all.csv"


EXPERIMENTS = [
    {
        "experiment": "eda_cnn_3class",
        "family": "single_modality",
        "model": "cnn",
        "backbone": "plain",
        "fusion": "none",
        "modality": "eda",
        "task": "3class",
        "window": "60s30s",
        "kept": "data/UNIVERSE/windows_eda_60s30s/loso_results_kept.csv",
        "skipped": "data/UNIVERSE/windows_eda_60s30s/loso_results_skipped.csv",
    },
    {
        "experiment": "eda_cnn_foldwise_kmeans_3class",
        "family": "single_modality",
        "model": "cnn",
        "backbone": "plain",
        "fusion": "none",
        "modality": "eda",
        "task": "3class",
        "window": "60s30s",
        "labeling": "foldwise_kmeans_weighted",
        "kept": "data/UNIVERSE/windows_eda_60s30s/loso_kmeans_foldwise_kept_weighted.csv",
        "skipped": "data/UNIVERSE/windows_eda_60s30s/loso_kmeans_foldwise_skipped_weighted.csv",
    },
    {
        "experiment": "eda_cnn_kmeans2_binary",
        "family": "single_modality",
        "model": "cnn",
        "backbone": "plain",
        "fusion": "none",
        "modality": "eda",
        "task": "binary",
        "window": "60s30s",
        "labeling": "foldwise_kmeans2_weighted",
        "kept": "data/UNIVERSE/results_loso_kmeans2_eda_cnn.csv",
    },
    {
        "experiment": "bvp_cnn_3class",
        "family": "single_modality",
        "model": "cnn",
        "backbone": "plain",
        "fusion": "none",
        "modality": "bvp",
        "task": "3class",
        "window": "60s30s",
        "kept": "data/UNIVERSE/windows_bvp_60s30s/loso_results_kept.csv",
        "skipped": "data/UNIVERSE/windows_bvp_60s30s/loso_results_skipped.csv",
    },
    {
        "experiment": "bvp_cnn_binary",
        "family": "single_modality",
        "model": "cnn",
        "backbone": "plain",
        "fusion": "none",
        "modality": "bvp",
        "task": "binary",
        "window": "60s30s",
        "kept": "data/UNIVERSE/windows_bvp_60s30s/binary_loso/loso_results_kept.csv",
        "skipped": "data/UNIVERSE/windows_bvp_60s30s/binary_loso/loso_results_skipped.csv",
    },
    {
        "experiment": "eda_bvp_earlyfusion_cnn_3class",
        "family": "fusion",
        "model": "cnn",
        "backbone": "plain",
        "fusion": "early",
        "modality": "eda_bvp",
        "task": "3class",
        "window": "60s30s",
        "kept": "data/UNIVERSE/windows_eda_bvp_earlyfusion_60s30s/loso_results_kept.csv",
        "skipped": "data/UNIVERSE/windows_eda_bvp_earlyfusion_60s30s/loso_results_skipped.csv",
    },
    {
        "experiment": "eda_bvp_earlyfusion_cnn_binary",
        "family": "fusion",
        "model": "cnn",
        "backbone": "plain",
        "fusion": "early",
        "modality": "eda_bvp",
        "task": "binary",
        "window": "60s30s",
        "kept": "data/UNIVERSE/windows_eda_bvp_earlyfusion_60s30s/binary_loso/loso_results_kept.csv",
        "skipped": "data/UNIVERSE/windows_eda_bvp_earlyfusion_60s30s/binary_loso/loso_results_skipped.csv",
    },
    {
        "experiment": "eda_bvp_latefusion_cnn_3class",
        "family": "fusion",
        "model": "cnn",
        "backbone": "plain",
        "fusion": "late",
        "modality": "eda_bvp",
        "task": "3class",
        "window": "60s30s",
        "kept": "results/cnn_latefusion/latefusion_cnn/loso_results_kept.csv",
        "skipped": "results/cnn_latefusion/latefusion_cnn/loso_results_skipped.csv",
    },
    {
        "experiment": "eda_bvp_latefusion_cnn_binary",
        "family": "fusion",
        "model": "cnn",
        "backbone": "plain",
        "fusion": "late",
        "modality": "eda_bvp",
        "task": "binary",
        "window": "60s30s",
        "kept": "results/cnn_latefusion/latefusion_cnn/binary_loso/loso_results_kept.csv",
        "skipped": "results/cnn_latefusion/latefusion_cnn/binary_loso/loso_results_skipped.csv",
    },
    {
        "experiment": "eda_bvp_latefusion_resnet_3class",
        "family": "fusion",
        "model": "resnet",
        "backbone": "resnet1d",
        "fusion": "late",
        "modality": "eda_bvp",
        "task": "3class",
        "window": "60s30s",
        "kept": "results/cnn_latefusion/latefusion_resnet/loso_results_kept.csv",
        "skipped": "results/cnn_latefusion/latefusion_resnet/loso_results_skipped.csv",
    },
    {
        "experiment": "eda_bvp_latefusion_resnet_binary",
        "family": "fusion",
        "model": "resnet",
        "backbone": "resnet1d",
        "fusion": "late",
        "modality": "eda_bvp",
        "task": "binary",
        "window": "60s30s",
        "kept": "results/cnn_latefusion/latefusion_resnet/binary_loso/loso_results_kept.csv",
        "skipped": "results/cnn_latefusion/latefusion_resnet/binary_loso/loso_results_skipped.csv",
    },
]


for window in ("30s15s", "60s30s", "90s30s", "120s60s"):
    for model, backbone in (("cnn", "plain"), ("resnet", "resnet1d")):
        folder = "latefusion_cnn" if model == "cnn" else "latefusion_resnet"
        if window == "30s15s":
            EXPERIMENTS.append(
                {
                    "experiment": f"window_ablation_{window}_latefusion_{model}_3class",
                    "family": "window_ablation",
                    "model": model,
                    "backbone": backbone,
                    "fusion": "late",
                    "modality": "eda_bvp",
                    "task": "3class",
                    "window": window,
                    "kept": f"results/cnn_window_ablation/{window}/{folder}/loso_results_kept.csv",
                    "skipped": f"results/cnn_window_ablation/{window}/{folder}/loso_results_skipped.csv",
                }
            )
        EXPERIMENTS.append(
            {
                "experiment": f"window_ablation_{window}_latefusion_{model}_binary",
                "family": "window_ablation",
                "model": model,
                "backbone": backbone,
                "fusion": "late",
                "modality": "eda_bvp",
                "task": "binary",
                "window": window,
                "kept": f"results/cnn_window_ablation/{window}/{folder}/binary_loso/loso_results_kept.csv",
                "skipped": f"results/cnn_window_ablation/{window}/{folder}/binary_loso/loso_results_skipped.csv",
            }
        )


def metric_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def add_metadata(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    out = df.copy()
    for key, value in meta.items():
        if key not in {"kept", "skipped"}:
            out[key] = value
    return out


def summarize_experiment(meta: dict):
    kept_path = Path(meta["kept"])
    skipped_path = Path(meta["skipped"]) if meta.get("skipped") else None

    row = {k: v for k, v in meta.items() if k not in {"kept", "skipped"}}
    row["kept_csv"] = str(kept_path)
    row["skipped_csv"] = str(skipped_path) if skipped_path else ""
    row["exists"] = kept_path.exists()

    if not kept_path.exists():
        row.update(
            {
                "kept_folds": 0,
                "skipped_folds": 0,
                "macroF1_mean": pd.NA,
                "macroF1_std": pd.NA,
                "balAcc_mean": pd.NA,
                "balAcc_std": pd.NA,
                "val_macroF1_mean": pd.NA,
                "n_test_sum": 0,
            }
        )
        return row, None, None

    kept = safe_read_csv(kept_path)
    skipped = safe_read_csv(skipped_path) if skipped_path and skipped_path.exists() else pd.DataFrame()

    f1_col = metric_col(kept, ("test_macroF1", "testF1", "test_f1"))
    ba_col = metric_col(kept, ("test_balAcc", "balAcc", "test_ba"))
    val_col = metric_col(kept, ("val_best_macroF1", "val_best_f1", "valF1"))

    row["kept_folds"] = int(len(kept))
    row["skipped_folds"] = int(len(skipped))
    row["macroF1_col"] = f1_col or ""
    row["balAcc_col"] = ba_col or ""
    row["macroF1_mean"] = kept[f1_col].mean() if f1_col else pd.NA
    row["macroF1_std"] = kept[f1_col].std(ddof=1) if f1_col and len(kept) > 1 else 0.0
    row["balAcc_mean"] = kept[ba_col].mean() if ba_col else pd.NA
    row["balAcc_std"] = kept[ba_col].std(ddof=1) if ba_col and len(kept) > 1 else 0.0
    row["val_macroF1_mean"] = kept[val_col].mean() if val_col else pd.NA
    row["n_test_sum"] = int(kept["n_test"].sum()) if "n_test" in kept.columns else 0

    kept_meta = add_metadata(kept, row)
    if f1_col:
        kept_meta["test_macroF1_norm"] = kept_meta[f1_col]
    if ba_col:
        kept_meta["test_balAcc_norm"] = kept_meta[ba_col]
    if val_col:
        kept_meta["val_macroF1_norm"] = kept_meta[val_col]

    skipped_meta = add_metadata(skipped, row) if len(skipped) else None
    return row, kept_meta, skipped_meta


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    subject_rows = []
    skipped_rows = []

    for meta in EXPERIMENTS:
        row, kept, skipped = summarize_experiment(meta)
        summary_rows.append(row)
        if kept is not None:
            subject_rows.append(kept)
        if skipped is not None:
            skipped_rows.append(skipped)

    summary = pd.DataFrame(summary_rows)
    per_subject = pd.concat(subject_rows, ignore_index=True, sort=False) if subject_rows else pd.DataFrame()
    skipped_all = pd.concat(skipped_rows, ignore_index=True, sort=False) if skipped_rows else pd.DataFrame()

    summary.to_csv(SUMMARY_OUT, index=False)
    per_subject.to_csv(PER_SUBJECT_OUT, index=False)
    skipped_all.to_csv(SKIPPED_OUT, index=False)

    print("Saved:")
    print(" -", SUMMARY_OUT)
    print(" -", PER_SUBJECT_OUT)
    print(" -", SKIPPED_OUT)
    print()
    cols = ["experiment", "task", "window", "model", "fusion", "kept_folds", "macroF1_mean", "balAcc_mean"]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
