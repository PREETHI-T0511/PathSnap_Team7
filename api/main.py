"""FastAPI backend. Evaluator references are read only after prediction for scoring."""
from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathsnap_common import (  # noqa: E402
    align_track_to_route,
    decode_candidate_sequences,
    local_xy,
    predict_bundle,
    read_numeric,
    trip_directories,
)

PREPROCESSING_SCRIPT = ROOT / "preprocessing" / "01_preprocess.py"
_spec = importlib.util.spec_from_file_location("pathsnap_preprocess_module", PREPROCESSING_SCRIPT)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load preprocessing module: {PREPROCESSING_SCRIPT}")
_preprocess = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_preprocess)

logger = logging.getLogger("pathsnap.api")
DEFAULT_MODEL = ROOT / "artifacts" / "tuned_best_model.joblib"
DEFAULT_EVALUATOR_DIR = ROOT / "data" / "evaluator_test"
TRIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class TripRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=80)


def _normalized_trip_files(root: Path, trip_id: str) -> Path | None:
    try:
        matches = sorted(root.rglob(f"{trip_id}.track"))
    except OSError:
        return None
    matches = [p for p in matches if p.is_file()]
    if len(matches) > 1:
        raise HTTPException(status_code=409, detail=f"Trip ID {trip_id} is not unique in the evaluator folder.")
    return matches[0].parent if matches else None


def _valid_track(folder: Path, trip_id: str) -> tuple[np.ndarray, np.ndarray]:
    track = read_numeric(folder / f"{trip_id}.track", 3, float)
    valid = np.isfinite(track).all(axis=1)
    valid &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    obs_ids = np.flatnonzero(valid)
    track = track[valid]
    if len(track) and np.any(np.diff(track[:, 2]) < 0):
        order = np.argsort(track[:, 2], kind="stable")
        track, obs_ids = track[order], obs_ids[order]
    return track, obs_ids


def _reference_alignment(folder: Path, trip_id: str, track: np.ndarray, obs_ids: np.ndarray) -> dict[int, int]:
    """Score-only reference mapping. Called after predictions; never passed to model features."""
    nodes = read_numeric(folder / f"{trip_id}.nodes", 2, float)
    arcs = read_numeric(folder / f"{trip_id}.arcs", 2, int)
    route = read_numeric(folder / f"{trip_id}.route", 1, int).reshape(-1)
    node_valid = np.isfinite(nodes).all(axis=1)
    node_valid &= (np.abs(nodes[:, 0]) <= 180) & (np.abs(nodes[:, 1]) <= 90)
    arc_from, arc_to = arcs[:, 0], arcs[:, 1]
    valid = (arc_from >= 0) & (arc_to >= 0) & (arc_from < len(nodes)) & (arc_to < len(nodes))
    valid &= node_valid[np.clip(arc_from, 0, max(0, len(nodes) - 1))]
    valid &= node_valid[np.clip(arc_to, 0, max(0, len(nodes) - 1))]
    arc_ids = np.flatnonzero(valid)
    if not len(arc_ids) or not len(route) or not len(track):
        return {}
    arc_id_to_ix = {int(arc_id): i for i, arc_id in enumerate(arc_ids)}
    route_ids = np.asarray([int(x) for x in route if int(x) in arc_id_to_ix], dtype=int)
    route_ix = np.asarray([arc_id_to_ix[int(x)] for x in route_ids], dtype=int)
    if not len(route_ix):
        return {}
    lon0, lat0 = float(np.median(track[:, 0])), float(np.median(track[:, 1]))
    nodes_xy = local_xy(nodes, lon0, lat0)
    edge_a, edge_b = nodes_xy[arc_from[valid]], nodes_xy[arc_to[valid]]
    gps_xy = local_xy(track[:, :2], lon0, lat0)
    aligned_ix, distances = align_track_to_route(gps_xy, route_ix, edge_a, edge_b)
    mapped_arc_ids = arc_ids[np.maximum(aligned_ix, 0)]
    result = {}
    for obs_id, arc_id, dist in zip(obs_ids, mapped_arc_ids, distances):
        if arc_id >= 0 and np.isfinite(dist) and dist <= 100.0:
            result[int(obs_id)] = int(arc_id)
    return result


def _route_continuity(predicted_rows) -> float | None:
    if len(predicted_rows) < 2:
        return None
    ordered = predicted_rows.sort_values("observation_id", kind="stable")
    prev, nxt = ordered.iloc[:-1], ordered.iloc[1:]
    connected = (prev["candidate_arc_id"].to_numpy() == nxt["candidate_arc_id"].to_numpy()) | (
        prev["to_node_id"].to_numpy() == nxt["from_node_id"].to_numpy())
    return float(np.mean(connected))


def _score_predictions(candidates, predicted_rows, references: dict[int, int]) -> dict[str, Any]:
    if not references:
        return {"status": "reference route unavailable", "matched_observations": 0}
    candidate_map = {
        int(obs): set(group["candidate_arc_id"].astype(int).tolist())
        for obs, group in candidates.groupby("observation_id", sort=False)
    }
    predicted_map = {
        int(row.observation_id): int(row.candidate_arc_id)
        for row in predicted_rows.itertuples()
    }
    scored = [(obs, arc) for obs, arc in references.items() if obs in predicted_map]
    if not scored:
        return {"status": "no aligned observations to score", "matched_observations": 0}
    misses = sum(arc not in candidate_map.get(obs, set()) for obs, arc in scored)
    correct = sum(predicted_map[obs] == arc for obs, arc in scored)
    return {
        "status": "scored against server-side reference",
        "top1_accuracy": correct / len(scored),
        "matched_observations": len(scored),
        "candidate_miss_rate": misses / len(scored),
        "route_continuity": _route_continuity(predicted_rows),
    }


def create_app(
    evaluator_dir: str | Path | None = None,
    model_path: str | Path | None = None,
) -> FastAPI:
    data_root = Path(evaluator_dir or os.getenv("PATHSNAP_EVALUATOR_DATA_DIR", DEFAULT_EVALUATOR_DIR)).resolve()
    model_file = Path(model_path or os.getenv("PATHSNAP_MODEL_PATH", DEFAULT_MODEL)).resolve()
    try:
        model_bundle = joblib.load(model_file) if model_file.is_file() else None
        model_load_error = None if model_bundle is not None else "model artifact not found"
    except Exception as exc:  # keep health endpoint available with a useful deployment diagnosis
        model_bundle = None
        model_load_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Could not load model artifact %s", model_file)

    app = FastAPI(title="PathSnap Evaluator API", version="1.0.0")
    origins = [x.strip() for x in os.getenv("PATHSNAP_CORS_ORIGINS", "*").split(",") if x.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health():
        return {
            "status": "ok" if model_bundle is not None else "degraded",
            "model_loaded": model_bundle is not None,
            "evaluator_data_available": data_root.is_dir(),
            "evaluator_trip_count": len(_list_evaluator_trip_ids(data_root)),
            "model_error": model_load_error,
        }

    @app.get("/api/trips")
    def list_trips():
        return {"trip_ids": _list_evaluator_trip_ids(data_root)}

    @app.post("/api/predict")
    def predict(request: TripRequest):
        trip_id = request.trip_id.strip()
        if not TRIP_ID_RE.fullmatch(trip_id):
            raise HTTPException(status_code=400, detail="Invalid trip ID.")
        if model_bundle is None:
            raise HTTPException(status_code=503, detail=f"Model is unavailable: {model_load_error}")
        folder = _normalized_trip_files(data_root, trip_id)
        if folder is None:
            raise HTTPException(status_code=404, detail="Trip not found in the evaluator partition.")
        required = [folder / f"{trip_id}{ext}" for ext in (".track", ".nodes", ".arcs", ".route")]
        missing = [p.name for p in required if not p.is_file()]
        if missing:
            raise HTTPException(status_code=422, detail=f"Evaluator trip is missing required files: {missing}")
        try:
            # Explicitly suppress reference-route reading for prediction.
            candidates, _ = _preprocess.process_trip(
                folder, radius_m=80.0, max_candidates=12,
                max_label_distance_m=100.0, use_route=False,
            )
            if candidates.empty:
                raise HTTPException(status_code=422, detail="No GPS observations or road candidates were generated.")
            probabilities, inference_seconds = predict_bundle(model_bundle, candidates)
            decoded, _ = decode_candidate_sequences(candidates, probabilities)
            predicted = candidates.loc[decoded].copy().sort_values("observation_id", kind="stable")
            track, obs_ids = _valid_track(folder, trip_id)
            # Reference labels are loaded only after the prediction is complete.
            references = _reference_alignment(folder, trip_id, track, obs_ids)
            metrics = _score_predictions(candidates, predicted, references)
            gps_points = [[float(lat), float(lon)] for lon, lat in track[:, :2]]
            predicted_segments = [
                [[float(row.from_lat), float(row.from_lon)], [float(row.to_lat), float(row.to_lon)]]
                for row in predicted.itertuples()
            ]
            return {
                "trip_id": trip_id,
                "gps_points": gps_points,
                "predicted_segments": predicted_segments,
                "predicted_arc_ids": predicted["candidate_arc_id"].astype(int).tolist(),
                "prediction_count": int(len(predicted)),
                "inference_seconds": float(inference_seconds),
                "metrics": metrics,
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Prediction failed for trip %s", trip_id)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}") from exc

    return app


def _list_evaluator_trip_ids(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    ids = []
    try:
        for folder in trip_directories(root):
            track = next(folder.glob("*.track"), None)
            if track is None:
                continue
            trip_id = track.stem
            if all((folder / f"{trip_id}{ext}").is_file() for ext in (".nodes", ".arcs", ".route")):
                ids.append(trip_id)
    except (OSError, FileNotFoundError):
        return []
    return sorted(ids)


app = create_app()
