import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def summarize(run_dir: Path):
    kept_path = run_dir / "loso_results_kept.csv"
    skipped_path = run_dir / "loso_results_skipped.csv"
    if not kept_path.exists():
        return None, None, None

    kept = safe_read_csv(kept_path)
    skipped = safe_read_csv(skipped_path) if skipped_path.exists() else pd.DataFrame()
    if len(kept) == 0:
        return None, None, None

    row = {
        "config_name": kept["config_name"].iloc[0] if "config_name" in kept.columns else run_dir.parent.name,
        "run_seed": int(kept["run_seed"].iloc[0]) if "run_seed" in kept.columns else -1,
        "pool": kept["pool"].iloc[0] if "pool" in kept.columns else "",
        "run_dir": str(run_dir),
        "kept_folds": int(len(kept)),
        "skipped_folds": int(len(skipped)),
        "macroF1_mean": kept["test_macroF1"].mean(),
        "macroF1_std": kept["test_macroF1"].std(ddof=1) if len(kept) > 1 else 0.0,
        "balAcc_mean": kept["test_balAcc"].mean(),
        "balAcc_std": kept["test_balAcc"].std(ddof=1) if len(kept) > 1 else 0.0,
        "n_test_sum": int(kept["n_test"].sum()) if "n_test" in kept.columns else 0,
    }

    for col in [
        "lr",
        "weight_decay",
        "dropout",
        "loss",
        "focal_gamma",
        "scheduler",
        "early_stop_patience",
        "balanced_sampler",
        "augment",
    ]:
        if col in kept.columns:
            row[col] = kept[col].iloc[0]

    kept = kept.copy()
    kept["run_dir"] = str(run_dir)
    kept["test_macroF1_norm"] = kept["test_macroF1"]
    kept["test_balAcc_norm"] = kept["test_balAcc"]

    skipped = skipped.copy()
    if len(skipped):
        skipped["run_dir"] = str(run_dir)

    return row, kept, skipped


def parse_args():
    parser = argparse.ArgumentParser(description="Collect attention pooling binary results.")
    parser.add_argument("--root", type=Path, default=Path("results/cnn_attention_binary"))
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    per_subject = []
    skipped_rows = []

    for kept_path in sorted(args.root.glob("*/*/loso_results_kept.csv")):
        row, kept, skipped = summarize(kept_path.parent)
        if row is None:
            continue
        rows.append(row)
        per_subject.append(kept)
        if skipped is not None and len(skipped):
            skipped_rows.append(skipped)

    args.root.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    per_subject_df = pd.concat(per_subject, ignore_index=True, sort=False) if per_subject else pd.DataFrame()
    skipped_df = pd.concat(skipped_rows, ignore_index=True, sort=False) if skipped_rows else pd.DataFrame()

    summary.to_csv(args.root / "summary_all.csv", index=False)
    per_subject_df.to_csv(args.root / "per_subject_all.csv", index=False)
    skipped_df.to_csv(args.root / "skipped_all.csv", index=False)

    print("Saved:")
    print(" -", args.root / "summary_all.csv")
    print(" -", args.root / "per_subject_all.csv")
    print(" -", args.root / "skipped_all.csv")
    if len(summary):
        print()
        print(
            summary.sort_values("macroF1_mean", ascending=False)[
                ["config_name", "pool", "run_seed", "kept_folds", "macroF1_mean", "balAcc_mean"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
