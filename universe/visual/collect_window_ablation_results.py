import argparse
from pathlib import Path

import pandas as pd


WINDOWS = ("30s15s", "60s30s", "90s30s", "120s60s")
MODELS = ("latefusion_cnn", "latefusion_resnet")


def summarize_result(root: Path, window: str, model: str, task: str):
    result_dir = root / window / model
    if task == "binary":
        result_dir = result_dir / "binary_loso"

    kept_path = result_dir / "loso_results_kept.csv"
    skipped_path = result_dir / "loso_results_skipped.csv"

    if not kept_path.exists():
        return None

    kept = pd.read_csv(kept_path)
    skipped = pd.read_csv(skipped_path) if skipped_path.exists() else pd.DataFrame()

    row = {
        "window": window,
        "model": model,
        "task": task,
        "kept_folds": int(len(kept)),
        "skipped_folds": int(len(skipped)),
        "kept_csv": str(kept_path),
        "skipped_csv": str(skipped_path) if skipped_path.exists() else "",
    }

    if len(kept) == 0:
        row.update(
            {
                "macroF1_mean": float("nan"),
                "macroF1_std": float("nan"),
                "balAcc_mean": float("nan"),
                "balAcc_std": float("nan"),
                "n_test_sum": 0,
            }
        )
        return row

    row.update(
        {
            "macroF1_mean": kept["test_macroF1"].mean(),
            "macroF1_std": kept["test_macroF1"].std(ddof=1) if len(kept) > 1 else 0.0,
            "balAcc_mean": kept["test_balAcc"].mean(),
            "balAcc_std": kept["test_balAcc"].std(ddof=1) if len(kept) > 1 else 0.0,
            "n_test_sum": int(kept["n_test"].sum()) if "n_test" in kept.columns else 0,
        }
    )
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Collect CNN window ablation result CSVs.")
    parser.add_argument("--root", type=Path, default=Path(r"results/cnn_window_ablation"))
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = args.out or (args.root / "summary_all.csv")

    rows = []
    for window in WINDOWS:
        for model in MODELS:
            for task in ("binary", "3class"):
                row = summarize_result(args.root, window, model, task)
                if row is not None:
                    rows.append(row)

    summary = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    print("Saved:", out_path)
    if len(summary) == 0:
        print("No finished ablation results found.")
    else:
        print(summary[["window", "model", "task", "kept_folds", "macroF1_mean", "balAcc_mean"]])


if __name__ == "__main__":
    main()
