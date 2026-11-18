"""Rank road candidates and decode a trip with the Viterbi continuity layer."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathsnap_common import decode_candidate_sequences, ensure_dir, predict_bundle


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates-csv", required=True, help="Output from preprocessing/01_preprocess.py")
    ap.add_argument("--model", required=True, help="A baseline model or tuned_best_model.joblib")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-name", default="decoded_routes.csv")
    args = ap.parse_args()
    out = ensure_dir(args.output_dir)
    df = pd.read_csv(args.candidates_csv)
    bundle = joblib.load(args.model)
    probabilities, elapsed = predict_bundle(bundle, df)
    decoded, stats = decode_candidate_sequences(df, probabilities)
    scored = df.copy()
    scored["model_probability"] = probabilities
    scored["viterbi_selected"] = decoded
    chosen = scored[scored["viterbi_selected"]].copy()
    chosen["predicted_arc_id"] = chosen["candidate_arc_id"].astype(int)
    raw = scored.sort_values("model_probability", ascending=False, kind="stable").drop_duplicates(["trip_id", "observation_id"])
    raw_map = {(str(r.trip_id), int(r.observation_id)): int(r.candidate_arc_id) for r in raw.itertuples()}
    chosen["classifier_top1_arc_id"] = [raw_map[(str(r.trip_id), int(r.observation_id))] for r in chosen.itertuples()]
    path = out / args.output_name
    chosen.sort_values(["trip_id", "observation_id"]).to_csv(path, index=False)
    print(f"Decoded observations: {len(chosen):,}")
    print(f"Inference seconds: {elapsed:.3f}")
    print(f"Route continuity: {stats['hmm_route_continuity']:.4f}")
    if pd.notna(stats["hmm_observation_top1_accuracy"]):
        print(f"Labeled-observation decoded top-1: {stats['hmm_observation_top1_accuracy']:.4f}")
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
