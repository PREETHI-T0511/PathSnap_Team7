"""Run the complete preprocessing, modeling, tuning, and decoding pipeline."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "project_paths.json"


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def run(script: Path, args: list[str]) -> None:
    command = [sys.executable, str(script), *args]
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="JSON file containing project data/output paths")
    parser.add_argument("--train-dir", default=None, help="Override the training directory in the config")
    parser.add_argument("--student-test-dir", default=None, help="Optional separate development partition")
    parser.add_argument("--output-dir", default=None, help="Override the artifacts output directory")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--candidate-radius-m", type=float, default=80.0)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--max-label-distance-m", type=float, default=100.0)
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Path configuration not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_base = config_path.parent

    train_dir = (
        resolve_path(args.train_dir, PROJECT_ROOT)
        if args.train_dir else resolve_path(config["train_data_dir"], config_base)
    )
    student_test_value = args.student_test_dir if args.student_test_dir else config.get("student_test_dir")
    student_test_base = PROJECT_ROOT if args.student_test_dir else config_base
    student_test_dir = resolve_path(student_test_value, student_test_base) if student_test_value else None
    output_dir = (
        resolve_path(args.output_dir, PROJECT_ROOT)
        if args.output_dir else resolve_path(config.get("artifacts_dir", "../artifacts"), config_base)
    )
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training data directory not found: {train_dir}. Update {config_path} or pass --train-dir.")
    if student_test_dir and not student_test_dir.is_dir():
        raise FileNotFoundError(f"Student-test directory not found: {student_test_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = output_dir / "train_candidates.csv"
    preprocessing = PROJECT_ROOT / "preprocessing" / "01_preprocess.py"
    run(preprocessing, [
        "--data-dir", str(train_dir), "--output-dir", str(output_dir), "--prefix", "train",
        "--candidate-radius-m", str(args.candidate_radius_m), "--max-label-distance-m", str(args.max_label_distance_m),
        "--max-candidates", str(args.max_candidates),
    ])

    student_test_csv = None
    if student_test_dir:
        student_test_csv = output_dir / "student_test_candidates.csv"
        run(preprocessing, [
            "--data-dir", str(student_test_dir), "--output-dir", str(output_dir), "--prefix", "student_test",
            "--candidate-radius-m", str(args.candidate_radius_m), "--max-label-distance-m", str(args.max_label_distance_m),
            "--max-candidates", str(args.max_candidates),
        ])

    run(PROJECT_ROOT / "feature_selection" / "02_feature_selection.py", [
        "--train-csv", str(train_csv), "--output-dir", str(output_dir),
    ])
    model_args = ["--train-csv", str(train_csv), "--output-dir", str(output_dir), "--folds", str(args.folds)]
    if student_test_csv:
        model_args += ["--student-test-csv", str(student_test_csv)]
    run(PROJECT_ROOT / "modeling" / "03_train_models.py", model_args)
    run(PROJECT_ROOT / "modeling" / "04_select_best_model.py", [
        "--comparison-csv", str(output_dir / "model_comparison.csv"), "--output-dir", str(output_dir),
    ])

    tune_args = [
        "--train-csv", str(train_csv), "--best-model-json", str(output_dir / "best_model.json"),
        "--output-dir", str(output_dir), "--trials", str(args.trials), "--folds", str(args.folds),
    ]
    if student_test_csv:
        tune_args += ["--student-test-csv", str(student_test_csv)]
    run(PROJECT_ROOT / "hyperparameter_tuning" / "05_bayesian_tune.py", tune_args)
    run(PROJECT_ROOT / "decoding" / "07_decoder_comparison.py", [
        "--train-csv", str(train_csv), "--best-model-json", str(output_dir / "best_model.json"),
        "--tuning-json", str(output_dir / "tuning_comparison.json"), "--output-dir", str(output_dir),
        "--folds", str(args.folds),
    ])
    print(f"\nPipeline complete. Results and model artifacts are in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
