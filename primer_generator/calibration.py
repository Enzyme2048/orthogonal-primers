from __future__ import annotations

import argparse
import csv
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ConfigError, effective_config
from .filters import FilterSettings, filter_candidate
from .generator import PROGRESS_UPDATE_INTERVAL_SECONDS, _build_filter_settings, _sample_gc_balanced_candidate
from .reporting import write_json, write_yaml
from .thermo import heterodimer_score_details


DEFAULT_CALIBRATION = {
    "filtered_sample_size": 400,
    "max_filter_attempts": 50000,
    "pair_sample_size": 4000,
    "graph_trials": 8,
    "graph_size_cap": 300,
    "random_seed_offset": 1000003,
}

LARGE_ATTEMPT_CAP = 10**18


@dataclass(frozen=True)
class CalibrationSequence:
    sequence: str


class _WallclockProgressMeter:
    def __init__(self, interval_seconds: float = PROGRESS_UPDATE_INTERVAL_SECONDS) -> None:
        self.interval_seconds = interval_seconds
        self.last_emit = time.monotonic()

    def should_emit(self) -> bool:
        now = time.monotonic()
        if now - self.last_emit >= self.interval_seconds:
            self.last_emit = now
            return True
        return False


def _calibration_settings(config: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = dict(DEFAULT_CALIBRATION)
    settings.update(config.get("calibration", {}))
    if overrides:
        settings.update({key: value for key, value in overrides.items() if value is not None})
    settings["filtered_sample_size"] = int(settings["filtered_sample_size"])
    settings["max_filter_attempts"] = int(settings["max_filter_attempts"])
    settings["pair_sample_size"] = int(settings["pair_sample_size"])
    settings["graph_trials"] = int(settings["graph_trials"])
    settings["graph_size_cap"] = int(settings["graph_size_cap"])
    settings["random_seed_offset"] = int(settings["random_seed_offset"])
    if settings["filtered_sample_size"] < 2:
        raise ConfigError("calibration.filtered_sample_size must be at least 2.")
    if settings["pair_sample_size"] < 1:
        raise ConfigError("calibration.pair_sample_size must be at least 1.")
    if settings["graph_trials"] < 1:
        raise ConfigError("calibration.graph_trials must be at least 1.")
    if settings["graph_size_cap"] < 8:
        raise ConfigError("calibration.graph_size_cap must be at least 8.")
    return settings


def _resolve_run_directory(config: dict[str, Any]) -> Path:
    root = Path(config["_meta"]["config_path"]).parent
    base_dir = (root / str(config["output"]["root_dir"])).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{timestamp}_calibration"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = base_dir / f"{timestamp}_calibration_{suffix:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _count_gc_constrained_sequences(length: int, gc_min_fraction: float, gc_max_fraction: float) -> int:
    min_gc = round(length * gc_min_fraction)
    max_gc = round(length * gc_max_fraction)
    total = 0
    for gc_count in range(min_gc, max_gc + 1):
        total += math.comb(length, gc_count) * (2**length)
    return total


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sample_filtered_sequences(
    config: dict[str, Any],
    settings: FilterSettings,
    sample_size: int,
    max_attempts: int,
    rng: random.Random,
    progress_enabled: bool,
) -> tuple[list[CalibrationSequence], int, Counter, float]:
    attempts = 0
    counters: Counter = Counter()
    seen: set[str] = set()
    records: list[CalibrationSequence] = []
    start = time.perf_counter()
    progress_meter = _WallclockProgressMeter()
    while len(records) < sample_size and attempts < max_attempts:
        attempts += 1
        candidate = _sample_gc_balanced_candidate(
            rng,
            int(config["generator"]["primer_length"]),
            float(config["composition"]["gc_min_fraction"]),
            float(config["composition"]["gc_max_fraction"]),
        )
        if candidate in seen:
            counters["duplicate"] += 1
            continue
        seen.add(candidate)
        failure_reason = filter_candidate(candidate, settings)
        if failure_reason is not None:
            counters[failure_reason] += 1
            continue
        records.append(
            CalibrationSequence(
                sequence=candidate,
            )
        )
        if progress_enabled and progress_meter.should_emit():
            print(
                "Calibration progress:"
                f" stage=filter_sampling,"
                f" attempts={attempts}/{max_attempts},"
                f" accepted={len(records)}/{sample_size},"
                f" duplicates={counters.get('duplicate', 0)}"
            )
    elapsed = time.perf_counter() - start
    return records, attempts, counters, elapsed


def _sample_pair_metrics(
    sequences: list[CalibrationSequence],
    pair_sample_size: int,
    thermo_config: dict[str, float],
    pair_tm_max_c: float,
    rng: random.Random,
    progress_enabled: bool,
) -> tuple[list[dict[str, Any]], float]:
    total_possible = len(sequences) * (len(sequences) - 1) // 2
    target_pairs = min(pair_sample_size, total_possible)
    seen_pairs: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    progress_meter = _WallclockProgressMeter()
    while len(rows) < target_pairs:
        left, right = sorted(rng.sample(range(len(sequences)), 2))
        pair = (left, right)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        left_record = sequences[left]
        right_record = sequences[right]
        details = heterodimer_score_details(left_record.sequence, right_record.sequence, thermo_config)
        rows.append(
            {
                "left_index": left,
                "right_index": right,
                "anchored_run_length": details["anchored_run_length"],
                "heterodimer_tm_c": details["tm_c"],
                "is_conflict": float(details["tm_c"]) > pair_tm_max_c,
            }
        )
        if progress_enabled and progress_meter.should_emit():
            print(
                "Calibration progress:"
                f" stage=pair_sampling,"
                f" pairs={len(rows)}/{target_pairs},"
                f" conflicts={sum(1 for row in rows if row['is_conflict'])}"
            )
    elapsed = time.perf_counter() - start
    return rows, elapsed


def _retained_fraction_for_degree(mean_degree: float, graph_size: int, trials: int, rng: random.Random) -> float:
    if mean_degree <= 0:
        return 1.0
    probability = min(1.0, mean_degree / max(graph_size - 1, 1))
    fractions: list[float] = []
    for _ in range(trials):
        adjacency = [set() for _ in range(graph_size)]
        for left in range(graph_size):
            for right in range(left + 1, graph_size):
                if rng.random() < probability:
                    adjacency[left].add(right)
                    adjacency[right].add(left)
        remaining = set(range(graph_size))
        while True:
            conflicted = [index for index in remaining if adjacency[index] & remaining]
            if not conflicted:
                break
            victim = max(conflicted, key=lambda idx: (len(adjacency[idx] & remaining), idx))
            remaining.remove(victim)
        fractions.append(len(remaining) / graph_size)
    return sum(fractions) / len(fractions)


def _estimate_search_size(
    target_size: int,
    conflict_probability: float,
    graph_size_cap: int,
    graph_trials: int,
    rng_seed: int,
) -> tuple[int, float, float]:
    if conflict_probability <= 0:
        return target_size, 1.0, 0.0
    accepted_before_pruning = max(1, target_size)
    retained_fraction = 0.0
    mean_degree = 0.0
    for _ in range(25):
        mean_degree = conflict_probability * max(accepted_before_pruning - 1, 0)
        sim_size = min(graph_size_cap, max(8, accepted_before_pruning))
        sim_rng = random.Random(rng_seed + accepted_before_pruning)
        retained_fraction = _retained_fraction_for_degree(mean_degree, sim_size, graph_trials, sim_rng)
        retained_fraction = max(retained_fraction, 1.0 / max(accepted_before_pruning, 1))
        revised = max(target_size, math.ceil(target_size / retained_fraction))
        if revised == accepted_before_pruning:
            break
        accepted_before_pruning = revised
    return accepted_before_pruning, retained_fraction, mean_degree


def _estimate_raw_attempts(accepted_before_pruning: int, local_pass_rate: float) -> float:
    if local_pass_rate <= 0:
        return float("inf")
    expected_attempts = accepted_before_pruning / local_pass_rate
    return expected_attempts if expected_attempts < LARGE_ATTEMPT_CAP else float("inf")


def _print_startup(config: dict[str, Any], run_dir: Path, settings: dict[str, Any]) -> None:
    print("Starting thermo calibration")
    print(f"  config: {config['_meta']['config_path']}")
    print(f"  output_dir: {run_dir}")
    print(
        "  sampling:"
        f" filtered_sequences={settings['filtered_sample_size']},"
        f" pair_samples={settings['pair_sample_size']},"
        f" graph_trials={settings['graph_trials']}"
    )


def _print_completion(summary: dict[str, Any], run_dir: Path) -> None:
    recommendation = summary["recommended"]
    print("Completed thermo calibration")
    print(
        "  recommendation:"
        f" estimated_runtime_seconds={recommendation['estimated_runtime_seconds']:.2f},"
        f" predicted_search_size={recommendation['predicted_search_size']},"
        f" predicted_raw_attempts={recommendation['predicted_raw_attempts']:.0f},"
        f" predicted_retained_fraction={recommendation['predicted_retained_fraction']:.3f}"
    )
    print(
        "  outputs:"
        f" summary_json={run_dir / 'calibration_summary.json'},"
        f" estimate_csv={run_dir / 'thermo_search_estimate.csv'}"
    )


def run_calibration(config: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = _calibration_settings(config, overrides)
    run_dir = _resolve_run_directory(config)
    _print_startup(config, run_dir, settings)

    seed = config["generator"].get("random_seed")
    base_seed = 0 if seed is None else int(seed)
    rng = random.Random(base_seed + settings["random_seed_offset"])
    filter_settings = _build_filter_settings(config)
    pair_tm_max_c = float(config["thermodynamics"]["pair_tm_max_c"])
    filtered_sequences, raw_attempts, rejection_counts, filter_elapsed = _sample_filtered_sequences(
        config,
        filter_settings,
        settings["filtered_sample_size"],
        settings["max_filter_attempts"],
        rng,
        progress_enabled=True,
    )
    if len(filtered_sequences) < 2:
        raise ConfigError(
            "Calibration could not collect enough locally valid sequences."
            f" Collected {len(filtered_sequences)} after {raw_attempts} attempts."
        )
    pair_rows, pair_elapsed = _sample_pair_metrics(
        filtered_sequences,
        settings["pair_sample_size"],
        dict(config["thermodynamics"]),
        pair_tm_max_c,
        rng,
        progress_enabled=True,
    )
    if not pair_rows:
        raise ConfigError("Calibration could not collect any distinct sequence pairs.")

    local_pass_rate = len(filtered_sequences) / raw_attempts
    seconds_per_attempt = filter_elapsed / max(raw_attempts, 1)
    seconds_per_tm_eval = pair_elapsed / len(pair_rows)
    gc_space_size = _count_gc_constrained_sequences(
        int(config["generator"]["primer_length"]),
        float(config["composition"]["gc_min_fraction"]),
        float(config["composition"]["gc_max_fraction"]),
    )
    filtered_space_estimate = gc_space_size * local_pass_rate

    pair_count = len(pair_rows)
    estimate_rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    max_run_length = max((row["anchored_run_length"] for row in pair_rows), default=0)
    for run_length in range(max_run_length + 1):
        exact_matches = [row for row in pair_rows if row["anchored_run_length"] == run_length]
        exact_conflicts = sum(1 for row in exact_matches if row["is_conflict"])
        histogram_rows.append(
            {
                "anchored_run_length": run_length,
                "pair_count": len(exact_matches),
                "conflict_count": exact_conflicts,
                "conflict_rate": (exact_conflicts / len(exact_matches)) if exact_matches else None,
            }
        )

    conflict_probability = sum(1 for row in pair_rows if row["is_conflict"]) / pair_count
    predicted_search_size, retained_fraction, mean_conflict_degree = _estimate_search_size(
        int(config["generator"]["target_size"]),
        conflict_probability,
        settings["graph_size_cap"],
        settings["graph_trials"],
        base_seed,
    )
    predicted_raw_attempts = _estimate_raw_attempts(predicted_search_size, local_pass_rate)
    predicted_validation_pairs = predicted_search_size * (predicted_search_size - 1) / 2
    estimated_runtime_seconds = (
        predicted_raw_attempts * seconds_per_attempt
        + predicted_validation_pairs * seconds_per_tm_eval
        if math.isfinite(predicted_raw_attempts)
        else float("inf")
    )
    print(
        "Calibration progress:"
        " stage=thermo_model,"
        f" sampled_pairs={pair_count},"
        f" conflict_probability={conflict_probability:.4f}"
    )
    estimate_rows.append(
        {
            "model": "thermo_only_3prime_rc",
            "local_pass_rate": local_pass_rate,
            "conflict_probability": conflict_probability,
            "predicted_search_size": predicted_search_size,
            "predicted_retained_fraction": retained_fraction,
            "predicted_mean_conflict_degree": mean_conflict_degree,
            "predicted_raw_attempts": predicted_raw_attempts,
            "predicted_validation_pairs": predicted_validation_pairs,
            "estimated_runtime_seconds": estimated_runtime_seconds,
            "filtered_space_estimate": filtered_space_estimate,
            "space_feasible": predicted_search_size <= filtered_space_estimate,
            "pair_support": pair_count,
        }
    )
    recommended = estimate_rows[0]

    write_json(
        run_dir / "calibration_summary.json",
        {
            "recommended": recommended,
            "calibration_inputs": {
                "filtered_sample_size": settings["filtered_sample_size"],
                "pair_sample_size": settings["pair_sample_size"],
                "graph_trials": settings["graph_trials"],
                "graph_size_cap": settings["graph_size_cap"],
                "seconds_per_attempt": seconds_per_attempt,
                "seconds_per_tm_eval": seconds_per_tm_eval,
                "gc_constrained_sequence_space": gc_space_size,
                "filtered_space_estimate": filtered_space_estimate,
                "raw_sampling_attempts": raw_attempts,
                "filtered_sequences_collected": len(filtered_sequences),
                "pair_samples_collected": len(pair_rows),
                "rejection_counts": dict(rejection_counts),
            },
            "search_estimate": estimate_rows,
        },
    )
    _write_csv(
        run_dir / "thermo_search_estimate.csv",
        [
            "model",
            "local_pass_rate",
            "conflict_probability",
            "predicted_search_size",
            "predicted_retained_fraction",
            "predicted_mean_conflict_degree",
            "predicted_raw_attempts",
            "predicted_validation_pairs",
            "estimated_runtime_seconds",
            "filtered_space_estimate",
            "space_feasible",
            "pair_support",
        ],
        estimate_rows,
    )
    _write_csv(
        run_dir / "pair_3prime_run_histogram.csv",
        ["anchored_run_length", "pair_count", "conflict_count", "conflict_rate"],
        histogram_rows,
    )
    write_yaml(run_dir / "calibration_resolved_config.yaml", effective_config(config))

    summary = {
        "recommended": recommended,
        "run_dir": str(run_dir),
        "search_estimate": estimate_rows,
        "pair_3prime_run_histogram": histogram_rows,
    }
    _print_completion(summary, run_dir)
    return summary
