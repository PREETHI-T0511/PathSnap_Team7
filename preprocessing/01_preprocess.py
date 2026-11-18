"""Clean raw trips, generate road candidates, and infer auditable route labels."""
from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathsnap_common import (
    ROW_COLUMNS, align_track_to_route, candidate_edges_for_points, local_xy,
    make_observation_features, point_segment_distances, read_numeric,
    trip_directories, ensure_dir,
)


def process_trip(folder: Path, radius_m: float, max_candidates: int, max_label_distance_m: float,
                 use_route: bool = True) -> tuple[pd.DataFrame, dict]:
    track_path = next(folder.glob("*.track"))
    trip_id = track_path.stem
    base = folder / trip_id
    track_raw = read_numeric(base.with_suffix(".track"), 3, float)
    nodes = read_numeric(base.with_suffix(".nodes"), 2, float)
    arcs_raw = read_numeric(base.with_suffix(".arcs"), 2, int)
    route_path = base.with_suffix(".route")
    route_raw = read_numeric(route_path, 1, int).reshape(-1) if use_route and route_path.exists() else np.empty(0, dtype=int)

    valid_track = np.isfinite(track_raw).all(axis=1)
    valid_track &= (np.abs(track_raw[:, 0]) <= 180) & (np.abs(track_raw[:, 1]) <= 90)
    track = track_raw[valid_track]
    original_obs = np.flatnonzero(valid_track)
    timestamp_ordered = bool(len(track) < 2 or np.all(np.diff(track[:, 2]) >= 0))
    if len(track) and not timestamp_ordered:
        order = np.argsort(track[:, 2], kind="stable")
        track, original_obs = track[order], original_obs[order]

    node_valid = np.isfinite(nodes).all(axis=1)
    node_valid &= (np.abs(nodes[:, 0]) <= 180) & (np.abs(nodes[:, 1]) <= 90)
    lon0 = float(np.nanmedian(track[:, 0])) if len(track) else float(np.nanmedian(nodes[node_valid, 0]))
    lat0 = float(np.nanmedian(track[:, 1])) if len(track) else float(np.nanmedian(nodes[node_valid, 1]))
    nodes_xy = local_xy(nodes, lon0, lat0)
    arc_from, arc_to = arcs_raw[:, 0], arcs_raw[:, 1]
    edge_valid = (arc_from >= 0) & (arc_to >= 0) & (arc_from < len(nodes)) & (arc_to < len(nodes))
    edge_valid &= node_valid[np.clip(arc_from, 0, max(0, len(nodes) - 1))]
    edge_valid &= node_valid[np.clip(arc_to, 0, max(0, len(nodes) - 1))]
    arc_ids = np.flatnonzero(edge_valid)
    edge_a, edge_b = nodes_xy[arc_from[edge_valid]], nodes_xy[arc_to[edge_valid]]
    if len(track) == 0:
        raise ValueError(f"{trip_id}: no valid GPS observations remain.")
    gps_xy = local_xy(track[:, :2], lon0, lat0)

    valid_arc_ids = set(arc_ids.tolist())
    route_ids = np.asarray([int(r) for r in route_raw if int(r) in valid_arc_ids], dtype=int)
    arc_id_to_ix = {int(arc_id): i for i, arc_id in enumerate(arc_ids)}
    route_ix = np.asarray([arc_id_to_ix[int(r)] for r in route_ids], dtype=int)
    route_connected_breaks = 0
    if len(route_ids) > 1:
        route_edge_rows = arcs_raw[route_ids]
        route_connected_breaks = int(np.sum(route_edge_rows[:-1, 1] != route_edge_rows[1:, 0]))
    aligned_edge_ix, alignment_dist = align_track_to_route(gps_xy, route_ix, edge_a, edge_b) if len(route_ix) else (
        np.full(len(track), -1, dtype=int), np.full(len(track), np.nan))
    alignment_arc = np.where(aligned_edge_ix >= 0, arc_ids[np.maximum(aligned_edge_ix, 0)], -1)

    candidate_lists, distance_lists, fallback = candidate_edges_for_points(
        gps_xy, edge_a, edge_b, radius_m=radius_m, max_candidates=max_candidates)
    obs_features = make_observation_features(gps_xy, track[:, 2])
    out_degree = np.bincount(arc_from[edge_valid], minlength=len(nodes))
    in_degree = np.bincount(arc_to[edge_valid], minlength=len(nodes))

    rows = []
    labeled_obs = set()
    candidate_miss_obs = set()
    for i, (candidate_ix, candidate_dist) in enumerate(zip(candidate_lists, distance_lists)):
        candidate_arc_ids = arc_ids[candidate_ix]
        route_arc = int(alignment_arc[i]) if alignment_arc[i] >= 0 else -1
        alignment_ok = bool(route_arc >= 0 and np.isfinite(alignment_dist[i]) and alignment_dist[i] <= max_label_distance_m)
        has_reference_candidate = alignment_ok and route_arc in set(candidate_arc_ids.tolist())
        if alignment_ok and not has_reference_candidate:
            candidate_miss_obs.add(i)
        if has_reference_candidate:
            labeled_obs.add(i)
        gps_heading = float(obs_features["gps_heading_deg"][i])
        for rank, (edge_ix, distance) in enumerate(zip(candidate_ix, candidate_dist), start=1):
            arc_id = int(arc_ids[edge_ix])
            from_node = int(arc_from[arc_id])
            to_node = int(arc_to[arc_id])
            vec = edge_b[edge_ix] - edge_a[edge_ix]
            edge_bearing = float(np.degrees(np.arctan2(vec[0], vec[1])) % 360.0)
            heading_diff = abs((gps_heading - edge_bearing + 180.0) % 360.0 - 180.0) if obs_features["heading_available"][i] else 0.0
            target = int(arc_id == route_arc) if has_reference_candidate else np.nan
            rows.append({
                "trip_id": trip_id,
                "observation_id": int(original_obs[i]),
                "candidate_arc_id": arc_id,
                "reference_arc_id": route_arc if alignment_ok else -1,
                "from_node_id": from_node,
                "to_node_id": to_node,
                "from_lon": float(nodes[from_node, 0]),
                "from_lat": float(nodes[from_node, 1]),
                "to_lon": float(nodes[to_node, 0]),
                "to_lat": float(nodes[to_node, 1]),
                "target": target,
                "is_fallback": int(fallback[i]),
                "alignment_distance_m": float(alignment_dist[i]) if np.isfinite(alignment_dist[i]) else np.nan,
                "distance_m": float(distance),
                "log_distance_m": float(np.log1p(max(0.0, distance))),
                "heading_diff_deg": heading_diff,
                "heading_cosine": float(np.cos(np.radians(heading_diff))),
                "heading_available": float(obs_features["heading_available"][i]),
                "edge_length_m": float(np.linalg.norm(vec)),
                "edge_out_degree": float(out_degree[arc_from[arc_ids[edge_ix]]]),
                "edge_in_degree": float(in_degree[arc_to[arc_ids[edge_ix]]]),
                "step_distance_m": float(obs_features["step_distance_m"][i]),
                "speed_mps": float(obs_features["speed_mps"][i]),
                "time_gap_s": float(obs_features["time_gap_s"][i]),
                "candidate_rank": rank,
                "candidate_count": len(candidate_ix) if not fallback[i] else 0,
            })

    aligned = np.isfinite(alignment_dist)
    audit = {
        "trip_id": trip_id,
        "folder": str(folder),
        "track_rows_raw": int(len(track_raw)),
        "track_rows_valid": int(len(track)),
        "invalid_track_rows": int(len(track_raw) - len(track)),
        "timestamp_ordered_in_source": timestamp_ordered,
        "track_rows_reordered_by_timestamp": bool(not timestamp_ordered),
        "nodes": int(len(nodes)),
        "arcs_raw": int(len(arcs_raw)),
        "arcs_valid": int(len(arc_ids)),
        "route_arcs_raw": int(len(route_raw)),
        "route_arcs_valid": int(len(route_ids)),
        "route_discontinuities": route_connected_breaks,
        "gps_observations_aligned": int(np.sum(aligned)),
        "alignment_median_distance_m": float(np.nanmedian(alignment_dist)) if np.any(aligned) else None,
        "alignment_p90_distance_m": float(np.nanpercentile(alignment_dist, 90)) if np.any(aligned) else None,
        "alignment_max_label_distance_m": float(max_label_distance_m),
        "labeled_observations": int(len(labeled_obs)),
        "candidate_miss_observations": int(len(candidate_miss_obs)),
        "fallback_candidate_rows": int(np.sum(fallback)),
        "unresolved_candidate_rows": int(sum(pd.isna(r["target"]) for r in rows)),
        "candidate_rows": int(len(rows)),
        "label_source": "inferred_monotone_route_alignment" if len(route_ids) else "unavailable",
    }
    return pd.DataFrame.from_records(rows, columns=ROW_COLUMNS), audit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="Root containing trip folders and .track/.nodes/.arcs/.route files")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--prefix", default="train", help="Output prefix, such as train or student_test")
    ap.add_argument("--candidate-radius-m", type=float, default=80.0)
    ap.add_argument("--max-candidates", type=int, default=12)
    ap.add_argument("--max-label-distance-m", type=float, default=100.0)
    ap.add_argument("--allow-unlabeled", action="store_true", help="Return success for inference data with no .route files")
    ap.add_argument("--ignore-route", action="store_true", help="Do not read .route files; use for inference to prevent label leakage")
    args = ap.parse_args()
    output = ensure_dir(args.output_dir)
    candidate_path = output / f"{args.prefix}_candidates.csv"
    audit_path = output / f"{args.prefix}_preprocessing_audit.csv"
    if candidate_path.exists():
        candidate_path.unlink()
    audits = []
    total_rows = total_fallback = total_unresolved = 0
    dirs = trip_directories(args.data_dir)
    print(f"Trips discovered: {len(dirs)}")
    for idx, folder in enumerate(dirs, start=1):
        try:
            trip_df, audit = process_trip(folder, args.candidate_radius_m, args.max_candidates,
                                          args.max_label_distance_m, use_route=not args.ignore_route)
        except Exception as exc:
            audit = {"trip_id": folder.name, "folder": str(folder), "error": f"{type(exc).__name__}: {exc}"}
            audits.append(audit)
            print(f"[{idx}/{len(dirs)}] {folder.name}: ERROR {exc}")
            continue
        trip_df.to_csv(candidate_path, mode="a", index=False, header=not candidate_path.exists())
        audits.append(audit)
        total_rows += audit["candidate_rows"]
        total_fallback += audit["fallback_candidate_rows"]
        total_unresolved += audit["unresolved_candidate_rows"]
        med = audit["alignment_median_distance_m"]
        print(f"[{idx}/{len(dirs)}] {audit['trip_id']}: {audit['track_rows_valid']} GPS, {audit['labeled_observations']} labeled obs, median alignment={med}")
    pd.DataFrame(audits).to_csv(audit_path, index=False)
    labeled = pd.read_csv(candidate_path, usecols=["target"]) if candidate_path.exists() else pd.DataFrame()
    labeled_rows = int(labeled["target"].notna().sum()) if len(labeled) else 0
    summary = {
        "data_dir": str(Path(args.data_dir).resolve()), "trips_discovered": len(dirs),
        "trips_processed": int(sum("error" not in x for x in audits)), "candidate_rows": total_rows,
        "labeled_candidate_rows": labeled_rows, "unresolved_candidate_rows": total_unresolved,
        "fallback_candidate_rows": total_fallback,
        "candidate_radius_m": args.candidate_radius_m,
        "max_label_distance_m": args.max_label_distance_m,
        "label_source": "inferred_monotone_route_alignment; validate before describing as ground truth",
        "candidate_csv": str(candidate_path), "audit_csv": str(audit_path),
    }
    (output / f"{args.prefix}_preprocessing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Fallback candidate rows: {total_fallback}")
    print(f"Labels generated: {labeled_rows} candidate rows; unresolved rows: {total_unresolved}")
    if labeled_rows == 0 and not args.allow_unlabeled:
        print("No supervised labels were generated. Inspect the audit CSV, route IDs, and alignment distances; training was not attempted.", file=sys.stderr)
        return 2
    print(f"Saved: {candidate_path}")
    print(f"Audit: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
