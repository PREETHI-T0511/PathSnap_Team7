"""Evaluate classifier-only versus Viterbi route predictions on held-out trips."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathsnap_common import exclude_poor_alignment_trips, group_cv, summarize_folds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--best-model-json", required=True)
    ap.add_argument("--tuning-json", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--cv-max-rows", type=int, default=60_000)
    args = ap.parse_args()
    out = Path(args.output_dir)
    train = pd.read_csv(args.train_csv)
    train = train[train["target"].notna()].copy().reset_index(drop=True)
    train["target"] = train["target"].astype(int)
    train, excluded = exclude_poor_alignment_trips(
        train, Path(args.train_csv).parent / "train_preprocessing_audit.csv")
    selected = json.loads(Path(args.best_model_json).read_text(encoding="utf-8"))
    tuned = json.loads(Path(args.tuning_json).read_text(encoding="utf-8"))
    model = selected["selected_model"]
    params = tuned["tuned_configuration"]
    folds = group_cv(train, model, params, n_splits=args.folds,
                     max_train_rows=args.cv_max_rows, seed=2200, include_decoder=True)
    pd.DataFrame(folds).to_csv(out / "decoder_cv_folds.csv", index=False)
    summary = summarize_folds(folds)
    result = {
        "model": model,
        "configuration": params,
        "excluded_poor_alignment_trips": excluded,
        "cv_raw_observation_top1_accuracy": summary["cv_observation_top1_accuracy_mean"],
        "cv_viterbi_observation_top1_accuracy": summary["cv_hmm_observation_top1_accuracy_mean"],
        "cv_raw_route_continuity": summary["cv_raw_route_continuity_mean"],
        "cv_viterbi_route_continuity": summary["cv_hmm_route_continuity_mean"],
        "viterbi_top1_delta": summary["cv_hmm_observation_top1_accuracy_mean"] - summary["cv_observation_top1_accuracy_mean"],
        "grouped_folds": args.folds,
        "cv_max_train_rows_per_fold": args.cv_max_rows,
        "fold_metrics_csv": str(out / "decoder_cv_folds.csv"),
    }
    (out / "decoder_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
