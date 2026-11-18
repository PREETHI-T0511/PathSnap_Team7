"""Generate a Folium map with raw GPS, relevant candidate roads, and decoded/reference routes."""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROJECT_PATHS = json.loads((ROOT / "config" / "project_paths.json").read_text(encoding="utf-8"))
import folium
from pathsnap_common import decode_candidate_sequences, predict_bundle, read_numeric

preprocessing_script = ROOT / "preprocessing" / "01_preprocess.py"
spec = importlib.util.spec_from_file_location("pathsnap_preprocess_module", preprocessing_script)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load preprocessing module: {preprocessing_script}")
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)


def find_trip(data_root: Path, trip_id: str) -> Path:
    matches = sorted(data_root.rglob(f"{trip_id}.track"))
    if not matches:
        raise FileNotFoundError(f"No {trip_id}.track found under {data_root}")
    if len(matches) > 1:
        raise ValueError(f"Trip ID {trip_id} is ambiguous under {data_root}")
    return matches[0].parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=PROJECT_PATHS["train_data_dir"],
                    help="Trip directory (defaults to config/project_paths.json)")
    ap.add_argument("--trip-id", default="00000001")
    ap.add_argument("--model", default=str(ROOT / "artifacts" / "tuned_best_model.joblib"))
    ap.add_argument("--output", default=str(ROOT / "artifacts" / "maps" / "pathsnap_trip_map.html"))
    ap.add_argument("--max-road-segments", type=int, default=3000)
    ap.add_argument("--no-reference", action="store_true", help="Do not draw the supplied route layer")
    args = ap.parse_args()
    folder = find_trip(Path(args.data_dir), args.trip_id)
    # Prediction candidates are built without reading the reference route.
    candidates, _ = preprocess.process_trip(folder, 80.0, 12, 100.0, use_route=False)
    bundle = joblib.load(args.model)
    probabilities, _ = predict_bundle(bundle, candidates)
    decoded, _ = decode_candidate_sequences(candidates, probabilities)
    predicted = candidates.loc[decoded].sort_values("observation_id", kind="stable")

    track = read_numeric(folder / f"{args.trip_id}.track", 3, float)
    valid = np.isfinite(track).all(axis=1) & (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    gps = track[valid]
    if len(gps) and np.any(np.diff(gps[:, 2]) < 0):
        gps = gps[np.argsort(gps[:, 2], kind="stable")]
    gps_latlon = [[float(row[1]), float(row[0])] for row in gps]
    if not gps_latlon:
        raise ValueError(f"Trip {args.trip_id} has no valid GPS coordinates.")
    center = [float(np.mean(gps[:, 1])), float(np.mean(gps[:, 0]))]
    m = folium.Map(location=center, zoom_start=14, control_scale=True, tiles="OpenStreetMap")

    gps_layer = folium.FeatureGroup(name="GPS observations", show=True)
    folium.PolyLine(gps_latlon, color="#1473e6", weight=3, opacity=0.55, tooltip="GPS trajectory").add_to(gps_layer)
    # Markers are thinned for long trips but the full observed line remains visible.
    stride = max(1, len(gps_latlon) // 300)
    for point in gps_latlon[::stride]:
        folium.CircleMarker(point, radius=2, color="#1473e6", fill=True, fill_opacity=0.65, weight=1).add_to(gps_layer)
    gps_layer.add_to(m)

    roads = folium.FeatureGroup(name="Relevant candidate road segments", show=False)
    unique = candidates.drop_duplicates("candidate_arc_id")
    if len(unique) > args.max_road_segments:
        unique = unique.sample(args.max_road_segments, random_state=42)
    for row in unique.itertuples():
        folium.PolyLine([[row.from_lat, row.from_lon], [row.to_lat, row.to_lon]],
                        color="#8795a3", weight=1, opacity=0.35).add_to(roads)
    roads.add_to(m)

    predicted_layer = folium.FeatureGroup(name="Predicted route (Viterbi)", show=True)
    for row in predicted.itertuples():
        folium.PolyLine([[row.from_lat, row.from_lon], [row.to_lat, row.to_lon]],
                        color="#1a9a70", weight=4, opacity=0.85,
                        tooltip=f"Predicted arc {int(row.candidate_arc_id)}").add_to(predicted_layer)
    predicted_layer.add_to(m)

    if not args.no_reference and (folder / f"{args.trip_id}.route").exists():
        nodes = read_numeric(folder / f"{args.trip_id}.nodes", 2, float)
        arcs = read_numeric(folder / f"{args.trip_id}.arcs", 2, int)
        route = read_numeric(folder / f"{args.trip_id}.route", 1, int).reshape(-1)
        reference_layer = folium.FeatureGroup(name="Supplied reference route", show=False)
        for arc_id in route:
            if 0 <= int(arc_id) < len(arcs):
                source, dest = arcs[int(arc_id)]
                if 0 <= source < len(nodes) and 0 <= dest < len(nodes):
                    folium.PolyLine([[nodes[source, 1], nodes[source, 0]],
                                     [nodes[dest, 1], nodes[dest, 0]]],
                                    color="#df6b20", weight=3, opacity=0.8,
                                    tooltip="Supplied reference route").add_to(reference_layer)
        reference_layer.add_to(m)

    folium.Marker(gps_latlon[0], tooltip="Trip start", icon=folium.Icon(color="blue", icon="play")).add_to(m)
    folium.Marker(gps_latlon[-1], tooltip="Trip end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    bounds = [[min(p[0] for p in gps_latlon), min(p[1] for p in gps_latlon)],
              [max(p[0] for p in gps_latlon), max(p[1] for p in gps_latlon)]]
    m.fit_bounds(bounds, padding=(20, 20))
    legend = """<div style='position:fixed;bottom:28px;left:28px;z-index:9999;background:white;padding:10px 12px;border:1px solid #999;border-radius:5px;font:13px Arial'>
    <b>PathSnap layers</b><br><span style='color:#1473e6'>●</span> GPS observations<br>
    <span style='color:#1a9a70'>━</span> Viterbi predicted path<br>
    <span style='color:#df6b20'>━</span> Supplied reference route<br>
    <span style='color:#8795a3'>━</span> Candidate road segments</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))
    print(f"Saved interactive Folium map: {output}")
    print("Prediction was generated without reading the reference route; that layer is added afterward for visualization only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
