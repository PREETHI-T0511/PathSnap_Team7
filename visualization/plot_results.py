"""Plot actual saved CV, decoder, and preprocessing results; no synthetic curves."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-dir", default=str(Path(__file__).resolve().parents[1] / "artifacts"))
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    artifacts = Path(args.artifacts_dir)
    out = Path(args.output_dir) if args.output_dir else artifacts / "plots"
    out.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(artifacts / "model_comparison.csv")
    order = comparison.sort_values("cv_observation_top1_accuracy_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    y = np.arange(len(order))
    ax.barh(y - 0.18, order["cv_observation_top1_accuracy_mean"], height=0.34, color="#1473e6", label="CV observation top-1")
    ax.barh(y + 0.18, order["cv_macro_f1_mean"], height=0.34, color="#39a57b", label="CV macro-F1")
    ax.set_yticks(y, order["model"].str.replace("_", " ").str.title())
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title("Five-model comparison · trip-grouped cross-validation")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=180)
    plt.close(fig)

    decoder = json.loads((artifacts / "decoder_comparison.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ["Classifier top-1", "Viterbi top-1", "Classifier continuity", "Viterbi continuity"]
    values = [decoder["cv_raw_observation_top1_accuracy"], decoder["cv_viterbi_observation_top1_accuracy"],
              decoder["cv_raw_route_continuity"], decoder["cv_viterbi_route_continuity"]]
    colors = ["#1473e6", "#1a9a70", "#8795a3", "#df6b20"]
    bars = ax.bar(names, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Classifier-only versus Viterbi · grouped CV")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "decoder_comparison.png", dpi=180)
    plt.close(fig)

    audit = pd.read_csv(artifacts / "train_preprocessing_audit.csv")
    audit["alignment_median_distance_m"] = pd.to_numeric(audit["alignment_median_distance_m"], errors="coerce")
    audit = audit.sort_values("alignment_median_distance_m", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    colors = ["#d96b4c" if val > 100 else "#4b8ec9" for val in audit["alignment_median_distance_m"].fillna(0)]
    ax.bar(np.arange(len(audit)), audit["alignment_median_distance_m"], color=colors)
    ax.axhline(100, color="#d96b4c", linestyle="--", linewidth=1, label="Training exclusion threshold (100 m)")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xticks(np.arange(len(audit)), audit["trip_id"].astype(str), rotation=90, fontsize=6)
    ax.set_ylabel("Median GPS-to-route alignment distance (m, symlog)")
    ax.set_title("Route alignment quality by training trip")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "alignment_quality.png", dpi=180)
    plt.close(fig)
    print(f"Saved actual-result plots to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
