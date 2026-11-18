"""Rank candidate features; fold-local selection is repeated inside model CV."""
from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathsnap_common import FEATURES, ensure_dir, exclude_poor_alignment_trips


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()
    out = ensure_dir(args.output_dir)
    df = pd.read_csv(args.train_csv)
    df = df[df["target"].notna()].copy()
    df, excluded = exclude_poor_alignment_trips(df, Path(args.train_csv).parent / "train_preprocessing_audit.csv")
    if df["target"].nunique() < 2:
        raise SystemExit("Feature selection needs both positive and negative labels. Check the preprocessing audit.")
    x = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, pvalues = f_classif(x.to_numpy(float), df["target"].astype(int).to_numpy())
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    pvalues = np.nan_to_num(pvalues, nan=1.0, posinf=1.0, neginf=1.0)
    ranking = pd.DataFrame({"feature": FEATURES, "f_score": scores, "p_value": pvalues})
    ranking = ranking.sort_values("f_score", ascending=False).reset_index(drop=True)
    ranking["selected_for_final_fit"] = ranking.index < min(args.top_k, len(FEATURES))
    ranking.to_csv(out / "feature_ranking.csv", index=False)
    selected = ranking.loc[ranking["selected_for_final_fit"], "feature"].tolist()
    (out / "selected_features.json").write_text(json.dumps({
        "selected_features": selected,
        "top_k": args.top_k,
        "method": "ANOVA F-score; ranking is descriptive over labeled training rows",
        "cv_note": "Model comparison and tuning refit feature selection inside each group-aware fold.",
        "labeled_candidate_rows": len(df),
        "excluded_poor_alignment_trips": excluded,
    }, indent=2), encoding="utf-8")
    print("Feature ranking:")
    print(ranking.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
