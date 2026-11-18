from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from api.main import create_app, _preprocess


def write_trip(root: Path, trip_id: str = "tiny01") -> Path:
    trip = root / trip_id
    trip.mkdir(parents=True)
    lon0, lat0 = 8.5900, 49.8440
    nodes = np.array([
        [lon0, lat0], [lon0 + 0.0003, lat0], [lon0 + 0.0006, lat0],
        [lon0 + 0.0009, lat0], [lon0 + 0.0003, lat0 + 0.0002],
    ])
    arcs = np.array([[0, 1], [1, 2], [2, 3], [1, 0], [2, 1], [3, 2], [1, 4], [4, 2]])
    track = np.array([[lon0 + i * 0.0001, lat0 + 0.00001, float(i + 1)] for i in range(10)])
    np.savetxt(trip / f"{trip_id}.nodes", nodes, fmt="%.8f", delimiter="\t")
    np.savetxt(trip / f"{trip_id}.arcs", arcs, fmt="%d", delimiter="\t")
    np.savetxt(trip / f"{trip_id}.track", track, fmt="%.8f", delimiter="\t")
    np.savetxt(trip / f"{trip_id}.route", np.array([[0], [1], [2]]), fmt="%d")
    return trip


def test_prediction_keeps_reference_labels_out_of_model_and_response(tmp_path: Path):
    evaluator_root = tmp_path / "evaluator_test"
    trip = write_trip(evaluator_root)
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "artifacts" / "tuned_best_model.joblib"
    assert model_path.is_file(), "Run the training pipeline before API integration tests."

    prediction_candidates, _ = _preprocess.process_trip(
        trip, radius_m=80.0, max_candidates=12, max_label_distance_m=100.0, use_route=False)
    assert prediction_candidates["target"].isna().all()
    assert (prediction_candidates["reference_arc_id"] == -1).all()

    client = TestClient(create_app(evaluator_root, model_path))
    health = client.get("/health")
    assert health.status_code == 200 and health.json()["model_loaded"]
    assert client.get("/api/trips").json()["trip_ids"] == ["tiny01"]
    response = client.post("/api/predict", json={"trip_id": "tiny01"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["prediction_count"] == 10
    assert len(data["gps_points"]) == 10
    assert len(data["predicted_segments"]) == 10
    assert "top1_accuracy" in data["metrics"]
    assert not any("reference_arc_id" in key or "ground_truth" in key for key in data)


def test_rejects_path_like_trip_ids(tmp_path: Path):
    evaluator_root = tmp_path / "evaluator_test"
    evaluator_root.mkdir()
    project_root = Path(__file__).resolve().parents[1]
    client = TestClient(create_app(evaluator_root, project_root / "artifacts" / "tuned_best_model.joblib"))
    response = client.post("/api/predict", json={"trip_id": "..\\outside"})
    assert response.status_code == 400
