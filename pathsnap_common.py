"""Shared data, candidate-generation, model, and Viterbi utilities for PathSnap."""
from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42
FEATURES = [
    "distance_m", "log_distance_m", "heading_diff_deg", "heading_cosine",
    "heading_available", "edge_length_m", "edge_out_degree", "edge_in_degree",
    "step_distance_m", "speed_mps", "time_gap_s", "candidate_rank",
    "candidate_count", "is_fallback",
]
ROW_COLUMNS = [
    "trip_id", "observation_id", "candidate_arc_id", "reference_arc_id",
    "from_node_id", "to_node_id", "from_lon", "from_lat", "to_lon", "to_lat",
    "target", "alignment_distance_m", *FEATURES,
]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_numeric(path: Path, columns: int, dtype=float) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if arr.size == 0:
        return np.empty((0, columns), dtype=dtype)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != columns:
        raise ValueError(f"{path.name}: expected {columns} whitespace-separated columns, got {arr.shape[1]}")
    return arr


def trip_directories(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    found = sorted({p.parent for p in root.rglob("*.track")})
    if not found:
        raise FileNotFoundError(f"No .track files found beneath {root}")
    return found


def local_xy(lonlat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    """Local equirectangular coordinates in metres, suitable for each trip's extent."""
    lonlat = np.asarray(lonlat, dtype=float)
    x = (lonlat[..., 0] - lon0) * (111_320.0 * math.cos(math.radians(lat0)))
    y = (lonlat[..., 1] - lat0) * 110_574.0
    return np.stack((x, y), axis=-1)


def point_segment_distances(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized point-to-segment distances. Inputs may be (N,2)/(N,K,2)."""
    v = b - a
    w = points - a
    denom = np.sum(v * v, axis=-1)
    t = np.divide(np.sum(w * v, axis=-1), denom, out=np.zeros_like(denom, dtype=float), where=denom > 1e-12)
    t = np.clip(t, 0.0, 1.0)
    projection = a + t[..., None] * v
    return np.linalg.norm(points - projection, axis=-1)


def align_track_to_route(
    gps_xy: np.ndarray,
    route_edge_ids: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    sigma_m: float = 20.0,
    skip_cost: float = 0.08,
    nearest_route_edges: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer a monotone GPS-to-reference-route alignment using Viterbi-style DP.

    This is a reproducible pseudo-labeling method, not proof that route and GPS
    rows were independently synchronized. Distances and unresolved rows are
    retained in the preprocessing audit.
    """
    n = len(gps_xy)
    m = len(route_edge_ids)
    if n == 0 or m == 0:
        return np.full(n, -1, dtype=int), np.full(n, np.nan, dtype=float)

    ra, rb = edge_a[route_edge_ids], edge_b[route_edge_ids]
    tree = cKDTree((ra + rb) / 2.0)
    k = min(max(1, nearest_route_edges), m)
    _, ix = tree.query(gps_xy, k=k)
    if k == 1:
        ix = ix[:, None]
    ix = np.asarray(ix, dtype=int)
    d = point_segment_distances(gps_xy[:, None, :], ra[ix], rb[ix])

    emissions = np.full((n, m), np.inf, dtype=np.float32)
    rows = np.repeat(np.arange(n), k)
    np.minimum.at(emissions, (rows, ix.reshape(-1)), (d / sigma_m).astype(np.float32).reshape(-1))

    back_dtype = np.uint16 if m < np.iinfo(np.uint16).max else np.uint32
    back = np.zeros((n, m), dtype=back_dtype)
    dp = emissions[0].astype(float)
    positions = np.arange(m, dtype=float)
    for t in range(1, n):
        adjusted = dp - skip_cost * positions
        prefix_cost = np.minimum.accumulate(adjusted)
        improves = adjusted <= prefix_cost
        prefix_idx = np.maximum.accumulate(np.where(improves, np.arange(m), 0))
        back[t] = prefix_idx.astype(back_dtype, copy=False)
        dp = emissions[t] + prefix_cost + skip_cost * positions
    state = int(np.argmin(dp))
    route_pos = np.empty(n, dtype=int)
    route_pos[-1] = state
    for t in range(n - 1, 0, -1):
        route_pos[t - 1] = int(back[t, route_pos[t]])
    mapped_arc_ids = route_edge_ids[route_pos]
    exact_dist = point_segment_distances(gps_xy, edge_a[mapped_arc_ids], edge_b[mapped_arc_ids])
    return mapped_arc_ids.astype(int), exact_dist.astype(float)


def candidate_edges_for_points(
    gps_xy: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    radius_m: float = 60.0,
    max_candidates: int = 8,
    midpoint_k: int = 64,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Return nearby candidate arc indices, distances, and fallback flags."""
    n, e = len(gps_xy), len(edge_a)
    if e == 0:
        raise ValueError("No valid road-network arcs remain after endpoint validation.")
    tree = cKDTree((edge_a + edge_b) / 2.0)
    k = min(max(1, midpoint_k), e)
    edge_lists: list[np.ndarray] = [np.empty(0, dtype=int) for _ in range(n)]
    dist_lists: list[np.ndarray] = [np.empty(0, dtype=float) for _ in range(n)]
    fallbacks = np.zeros(n, dtype=bool)
    chunk = 6000
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        _, ix = tree.query(gps_xy[start:stop], k=k)
        if k == 1:
            ix = ix[:, None]
        ix = np.asarray(ix, dtype=int)
        distances = point_segment_distances(gps_xy[start:stop, None, :], edge_a[ix], edge_b[ix])
        for local, row_dist in enumerate(distances):
            obs = start + local
            order = np.argsort(row_dist, kind="stable")
            within = order[row_dist[order] <= radius_m][:max_candidates]
            if len(within):
                edge_lists[obs] = ix[local, within]
                dist_lists[obs] = row_dist[within]
            else:
                best = int(order[0])
                edge_lists[obs] = ix[local, [best]]
                dist_lists[obs] = row_dist[[best]]
                fallbacks[obs] = True
    return edge_lists, dist_lists, fallbacks


def make_observation_features(gps_xy: np.ndarray, times: np.ndarray) -> dict[str, np.ndarray]:
    n = len(gps_xy)
    delta = np.zeros((n, 2), dtype=float)
    dt = np.zeros(n, dtype=float)
    if n > 1:
        delta[1:] = gps_xy[1:] - gps_xy[:-1]
        dt[1:] = np.maximum(0.0, np.diff(times))
    step = np.linalg.norm(delta, axis=1)
    speed = np.divide(step, np.maximum(dt, 1.0), out=np.zeros(n, dtype=float), where=np.isfinite(dt))
    speed = np.clip(speed, 0, 80)
    # Smooth the noisy 1 Hz bearing over a small window while retaining each
    # observation's one-step speed and distance features.
    indices = np.arange(n)
    left = np.maximum(0, indices - 2)
    right = np.minimum(max(0, n - 1), indices + 2)
    heading_delta = gps_xy[right] - gps_xy[left]
    heading_length = np.linalg.norm(heading_delta, axis=1)
    heading_available = (heading_length >= 2.0).astype(float)
    heading = np.degrees(np.arctan2(heading_delta[:, 0], heading_delta[:, 1])) % 360.0
    return {"time_gap_s": dt, "step_distance_m": step, "speed_mps": speed,
            "gps_heading_deg": heading, "heading_available": heading_available}


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"family": "linear", "params": {"C": 1.0}},
    "decision_tree": {"family": "tree", "params": {"max_depth": 12, "min_samples_leaf": 4}},
    "random_forest": {"family": "bagging ensemble", "params": {"n_estimators": 60, "max_depth": 16, "min_samples_leaf": 3, "max_features": "sqrt"}},
    "extra_trees": {"family": "randomized ensemble", "params": {"n_estimators": 80, "max_depth": 16, "min_samples_leaf": 2, "max_features": "sqrt"}},
    "hist_gradient_boosting": {"family": "boosting", "params": {"learning_rate": 0.08, "max_iter": 100, "max_leaf_nodes": 31, "l2_regularization": 1.0}},
}


def make_estimator(name: str, params: dict[str, Any] | None = None):
    cfg = dict(MODEL_SPECS[name]["params"])
    if params:
        cfg.update(params)
    if name == "logistic_regression":
        return LogisticRegression(C=float(cfg["C"]), solver="liblinear", max_iter=300, random_state=SEED)
    if name == "decision_tree":
        return DecisionTreeClassifier(max_depth=cfg["max_depth"], min_samples_leaf=int(cfg["min_samples_leaf"]), random_state=SEED)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=int(cfg["n_estimators"]), max_depth=cfg["max_depth"],
            min_samples_leaf=int(cfg["min_samples_leaf"]), max_features=cfg["max_features"], n_jobs=-1, random_state=SEED)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=int(cfg["n_estimators"]), max_depth=cfg["max_depth"],
            min_samples_leaf=int(cfg["min_samples_leaf"]), max_features=cfg["max_features"], n_jobs=-1, random_state=SEED)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(learning_rate=float(cfg["learning_rate"]), max_iter=int(cfg["max_iter"]),
            max_leaf_nodes=int(cfg["max_leaf_nodes"]), l2_regularization=float(cfg["l2_regularization"]),
            early_stopping=False, random_state=SEED)
    raise KeyError(name)


def sample_rows(df: pd.DataFrame, max_rows: int | None, seed: int = SEED) -> np.ndarray:
    """Trip-balanced, class-aware row cap used for practical CV runtimes."""
    n = len(df)
    if max_rows is None or n <= max_rows:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    groups = df["trip_id"].astype(str).to_numpy()
    y = df["target"].astype(int).to_numpy()
    unique = np.unique(groups)
    quota = max(2, int(math.ceil(max_rows / max(1, len(unique)))))
    chosen: list[np.ndarray] = []
    for group in unique:
        idx = np.flatnonzero(groups == group)
        if len(idx) <= quota:
            chosen.append(idx)
            continue
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        pos_quota = min(len(pos), max(1, quota // 3))
        neg_quota = min(len(neg), quota - pos_quota)
        remain = quota - pos_quota - neg_quota
        if remain > 0:
            pos_quota += min(remain, len(pos) - pos_quota)
            remain = quota - pos_quota - neg_quota
        if remain > 0:
            neg_quota += min(remain, len(neg) - neg_quota)
        selected = np.concatenate([
            rng.choice(pos, pos_quota, replace=False) if pos_quota else np.empty(0, dtype=int),
            rng.choice(neg, neg_quota, replace=False) if neg_quota else np.empty(0, dtype=int),
        ])
        chosen.append(selected)
    out = np.concatenate(chosen) if chosen else np.empty(0, dtype=int)
    if len(out) > max_rows:
        out = rng.choice(out, max_rows, replace=False)
    return np.sort(out)


def fit_bundle(
    train_df: pd.DataFrame,
    model_name: str,
    params: dict[str, Any] | None = None,
    max_rows: int | None = 120_000,
    top_k: int = 10,
    seed: int = SEED,
) -> dict[str, Any]:
    idx = sample_rows(train_df, max_rows, seed)
    sample = train_df.iloc[idx]
    y = sample["target"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Training sample has only one target class; check label generation and candidate misses.")
    x = sample[FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    selector = SelectKBest(score_func=f_classif, k=min(top_k, len(FEATURES)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = imputer.fit_transform(x)
        x = scaler.fit_transform(x)
        x = selector.fit_transform(x, y)
    estimator = make_estimator(model_name, params)
    weights = compute_sample_weight(class_weight="balanced", y=y)
    estimator.fit(x, y, sample_weight=weights)
    return {
        "model_name": model_name,
        "params": {**MODEL_SPECS[model_name]["params"], **(params or {})},
        "features": FEATURES,
        "imputer": imputer,
        "scaler": scaler,
        "selector": selector,
        "selected_features": [f for f, keep in zip(FEATURES, selector.get_support()) if keep],
        "estimator": estimator,
        "fit_rows": int(len(sample)),
        "seed": seed,
    }


def predict_bundle(bundle: dict[str, Any], df: pd.DataFrame) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    x = df[FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    x = bundle["imputer"].transform(x)
    x = bundle["scaler"].transform(x)
    x = bundle["selector"].transform(x)
    probabilities = bundle["estimator"].predict_proba(x)[:, 1]
    return probabilities, time.perf_counter() - start


def evaluate_scores(df: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    y = df["target"].astype(int).to_numpy()
    pred = (probabilities >= 0.5).astype(int)
    result: dict[str, float] = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "log_loss": float(log_loss(y, np.column_stack([1 - probabilities, probabilities]), labels=[0, 1])),
    }
    scored = df[["trip_id", "observation_id", "target"]].copy()
    scored["score"] = probabilities
    top = scored.sort_values("score", ascending=False, kind="stable").drop_duplicates(["trip_id", "observation_id"])
    result["observation_top1_accuracy"] = float((top["target"].astype(int) == 1).mean()) if len(top) else float("nan")
    result["scored_observations"] = float(len(top))
    return result


def _continuity_rate(df: pd.DataFrame, chosen_rows: np.ndarray) -> float:
    if len(chosen_rows) < 2 or not {"candidate_arc_id", "from_node_id", "to_node_id"}.issubset(df.columns):
        return float("nan")
    chosen = df.iloc[chosen_rows].copy()
    chosen = chosen.sort_values(["trip_id", "observation_id"], kind="stable")
    rates = []
    for _, trip in chosen.groupby("trip_id", sort=False):
        if len(trip) < 2:
            continue
        prev = trip.iloc[:-1]
        nxt = trip.iloc[1:]
        connected = (prev["candidate_arc_id"].to_numpy() == nxt["candidate_arc_id"].to_numpy()) | (
            prev["to_node_id"].to_numpy() == nxt["from_node_id"].to_numpy())
        rates.extend(connected.tolist())
    return float(np.mean(rates)) if rates else float("nan")


def decode_candidate_sequences(df: pd.DataFrame, probabilities: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Viterbi decode candidate sequences using classifier emissions and road continuity."""
    n = len(df)
    decoded = np.zeros(n, dtype=bool)
    picked_rows: list[int] = []
    raw_rows: list[int] = []
    truth_correct: list[bool] = []
    raw_truth_correct: list[bool] = []
    work = df.copy()
    work["_row_pos"] = np.arange(n)
    for _, trip in work.groupby("trip_id", sort=False, observed=True):
        trip = trip.sort_values(["observation_id", "candidate_rank"], kind="stable")
        obs = trip["observation_id"].to_numpy()
        starts = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
        ends = np.r_[starts[1:], len(trip)]
        row_ids = trip["_row_pos"].to_numpy(dtype=int)
        probs = np.clip(probabilities[row_ids], 1e-8, 1.0)
        arc_ids = trip["candidate_arc_id"].to_numpy()
        from_nodes = trip["from_node_id"].to_numpy()
        to_nodes = trip["to_node_id"].to_numpy()
        from_lon = np.radians(trip["from_lon"].to_numpy(float))
        from_lat = np.radians(trip["from_lat"].to_numpy(float))
        to_lon = np.radians(trip["to_lon"].to_numpy(float))
        to_lat = np.radians(trip["to_lat"].to_numpy(float))
        steps = trip["step_distance_m"].to_numpy(float)
        gaps = trip["time_gap_s"].to_numpy(float)
        targets = trip["target"].to_numpy(float)
        back: list[np.ndarray] = []
        dp = None
        raw_local_choices = []
        for t, (start, end) in enumerate(zip(starts, ends)):
            p = probs[start:end]
            emission = np.log(p / max(float(p.sum()), 1e-12))
            raw_local_choices.append(int(np.argmax(p)))
            if t == 0:
                dp = emission
                back.append(np.full(end - start, -1, dtype=int))
                continue
            ps, pe = starts[t - 1], ends[t - 1]
            prev_to = to_nodes[ps:pe]
            prev_arc = arc_ids[ps:pe]
            curr_from = from_nodes[start:end]
            curr_arc = arc_ids[start:end]
            same = prev_arc[:, None] == curr_arc[None, :]
            connected = prev_to[:, None] == curr_from[None, :]
            lon1, lat1 = to_lon[ps:pe, None], to_lat[ps:pe, None]
            lon2, lat2 = from_lon[None, start:end], from_lat[None, start:end]
            dlat, dlon = lat2 - lat1, lon2 - lon1
            h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            gap_m = 6_371_000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
            gps_step = float(steps[start])
            gap_s = float(gaps[start])
            allowed = max(30.0, 2.5 * gps_step + 15.0) * min(max(gap_s, 1.0), 30.0)
            transition = -1.25 - np.maximum(0.0, gap_m - allowed) / 20.0
            transition = np.where(connected | same, 0.25, transition)
            scores = dp[:, None] + transition
            back_idx = np.argmax(scores, axis=0)
            dp = emission + scores[back_idx, np.arange(end - start)]
            back.append(back_idx)
        state = int(np.argmax(dp))
        decoded_local = np.empty(len(starts), dtype=int)
        for t in range(len(starts) - 1, -1, -1):
            decoded_local[t] = state
            if t:
                state = int(back[t][state])
        for t, (start, end) in enumerate(zip(starts, ends)):
            selected = int(row_ids[start + decoded_local[t]])
            raw = int(row_ids[start + raw_local_choices[t]])
            decoded[selected] = True
            picked_rows.append(selected)
            raw_rows.append(raw)
            positive = np.flatnonzero(targets[start:end] == 1)
            if len(positive):
                truth_correct.append(bool(arc_ids[start + decoded_local[t]] == arc_ids[start + positive[0]]))
                raw_truth_correct.append(bool(arc_ids[start + raw_local_choices[t]] == arc_ids[start + positive[0]]))
    decoded_accuracy = float(np.mean(truth_correct)) if truth_correct else float("nan")
    raw_accuracy = float(np.mean(raw_truth_correct)) if raw_truth_correct else float("nan")
    raw_chosen = np.asarray(raw_rows, dtype=int)
    decoded_chosen = np.asarray(picked_rows, dtype=int)
    stats = {
        "hmm_observation_top1_accuracy": decoded_accuracy,
        "raw_sequence_observation_top1_accuracy": raw_accuracy,
        "raw_route_continuity": _continuity_rate(df, raw_chosen),
        "hmm_route_continuity": _continuity_rate(df, decoded_chosen),
    }
    return decoded, stats


def group_cv(
    df: pd.DataFrame,
    model_name: str,
    params: dict[str, Any] | None = None,
    n_splits: int = 3,
    max_train_rows: int | None = 120_000,
    seed: int = SEED,
    include_decoder: bool = False,
) -> list[dict[str, Any]]:
    groups = df["trip_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    results = []
    for fold, (tr, va) in enumerate(splitter.split(df, df["target"], groups), start=1):
        train_df = df.iloc[tr].reset_index(drop=True)
        valid_df = df.iloc[va].reset_index(drop=True)
        start = time.perf_counter()
        bundle = fit_bundle(train_df, model_name, params, max_train_rows, seed=seed + fold)
        train_seconds = time.perf_counter() - start
        probabilities, inference_seconds = predict_bundle(bundle, valid_df)
        metrics = evaluate_scores(valid_df, probabilities)
        sequence_metrics = {}
        if include_decoder:
            _, sequence_metrics = decode_candidate_sequences(valid_df, probabilities)
        results.append({
            "fold": fold,
            "model": model_name,
            "params": json.dumps({**MODEL_SPECS[model_name]["params"], **(params or {})}, sort_keys=True),
            "train_rows_total": len(train_df),
            "train_rows_fit": bundle["fit_rows"],
            "validation_rows": len(valid_df),
            "train_seconds": train_seconds,
            "inference_seconds": inference_seconds,
            **metrics,
            **sequence_metrics,
        })
    return results


def summarize_folds(folds: list[dict[str, Any]]) -> dict[str, float]:
    metrics = ["accuracy", "macro_f1", "positive_f1", "log_loss", "observation_top1_accuracy",
               "train_seconds", "inference_seconds"]
    metrics += [m for m in ["hmm_observation_top1_accuracy", "raw_sequence_observation_top1_accuracy", "raw_route_continuity", "hmm_route_continuity"]
                if any(m in row for row in folds)]
    summary: dict[str, float] = {}
    for metric in metrics:
        vals = np.asarray([row[metric] for row in folds], dtype=float)
        summary[f"cv_{metric}_mean"] = float(np.nanmean(vals))
        summary[f"cv_{metric}_std"] = float(np.nanstd(vals))
    return summary


def evaluate_external(bundle: dict[str, Any], df: pd.DataFrame, label: str = "student_test") -> dict[str, Any]:
    if df.empty:
        return {"partition": label, "status": "no labeled rows"}
    probs, elapsed = predict_bundle(bundle, df)
    return {"partition": label, "status": "measured", "inference_seconds": elapsed, **evaluate_scores(df, probs)}


def save_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def exclude_poor_alignment_trips(
    df: pd.DataFrame,
    audit_csv: str | Path | None,
    median_distance_threshold_m: float = 100.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop training trips whose GPS-to-reference alignment is globally unreliable."""
    if not audit_csv or not Path(audit_csv).exists():
        return df, []
    audit = pd.read_csv(audit_csv)
    required = {"trip_id", "alignment_median_distance_m"}
    if not required.issubset(audit.columns):
        return df, []
    def normalize(values: pd.Series) -> pd.Series:
        return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    medians = pd.to_numeric(audit["alignment_median_distance_m"], errors="coerce")
    bad_ids = normalize(audit.loc[medians > median_distance_threshold_m, "trip_id"]).tolist()
    if not bad_ids:
        return df, []
    result = df.copy()
    result["trip_id"] = normalize(result["trip_id"])
    kept = result[~result["trip_id"].isin(set(bad_ids))].copy()
    return kept.reset_index(drop=True), bad_ids


def bayesian_candidate_pool(model_name: str) -> list[dict[str, Any]]:
    """Finite search domains for GP expected-improvement optimization."""
    import itertools
    if model_name == "logistic_regression":
        domains = {"C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]}
    elif model_name == "decision_tree":
        domains = {"max_depth": [6, 10, 14, 20, None], "min_samples_leaf": [2, 4, 8, 16]}
    elif model_name in ("random_forest", "extra_trees"):
        domains = {"n_estimators": [40, 60, 90, 120], "max_depth": [10, 14, 18, None],
                   "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.7]}
    elif model_name == "hist_gradient_boosting":
        domains = {"learning_rate": [0.03, 0.05, 0.08, 0.12], "max_iter": [60, 100, 150, 220],
                   "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0.0, 0.3, 1.0, 3.0]}
    else:
        raise KeyError(model_name)
    keys = list(domains)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(domains[k] for k in keys))]


def config_vector(config: dict[str, Any], domains: list[dict[str, Any]]) -> np.ndarray:
    keys = list(domains[0])
    dims = []
    for key in keys:
        values = sorted({str(d[key]) for d in domains})
        lookup = {v: i / max(1, len(values) - 1) for i, v in enumerate(values)}
        dims.append(lookup[str(config[key])])
    return np.asarray(dims, dtype=float)


def gp_expected_improvement(
    candidates: list[dict[str, Any]],
    observed: list[dict[str, Any]],
    scores: list[float],
) -> dict[str, Any]:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    tried = {json.dumps(c, sort_keys=True) for c in observed}
    pool = [c for c in candidates if json.dumps(c, sort_keys=True) not in tried]
    if not pool:
        raise ValueError("Bayesian search exhausted its candidate pool.")
    if len(observed) < 3:
        return pool[0]
    x = np.vstack([config_vector(c, candidates) for c in observed])
    xp = np.vstack([config_vector(c, candidates) for c in pool])
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=0.3, nu=2.5) + WhiteKernel(noise_level=1e-4)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=1, random_state=SEED)
    gp.fit(x, np.asarray(scores, dtype=float))
    mean, std = gp.predict(xp, return_std=True)
    improvement = mean - max(scores) - 0.005
    z = np.divide(improvement, std, out=np.zeros_like(improvement), where=std > 1e-12)
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return pool[int(np.argmax(ei))]
