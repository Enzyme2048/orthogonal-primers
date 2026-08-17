from __future__ import annotations

import itertools
import math
import random
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import ConfigError, effective_config, normalize_effective_config
from .encoding import reverse_complement
from .filters import (
    FilterSettings,
    exceeds_homopolymer_limit,
    exceeds_repeat_limit,
    filter_candidate,
)
from .graph import build_conflict_graph, greedy_maximal_independent_set
from .reporting import load_checkpoint, save_checkpoint, write_csv, write_json, write_yaml
from .thermo import heterodimer_score_details, heterodimer_score_details_from_partner, intrinsic_tm_with_config_c, sense_score_details


@dataclass
class PrimerRecord:
    sequence: str
    rc_sequence: str
    gc_fraction: float
    intrinsic_tm_c: float


@dataclass
class PairingRecord:
    left_sequence: str
    right_sequence: str
    left_vs_right_sense_tm_c: float
    right_vs_left_sense_tm_c: float
    pair_tm_c: float


def _make_primer_record(sequence: str, thermo_runtime_config: dict[str, float]) -> PrimerRecord:
    return PrimerRecord(
        sequence=sequence,
        rc_sequence=reverse_complement(sequence),
        gc_fraction=(sequence.count("G") + sequence.count("C")) / len(sequence),
        intrinsic_tm_c=intrinsic_tm_with_config_c(sequence, thermo_runtime_config),
    )


def _format_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


FILTER_FUNNEL_ORDER = [
    "duplicate",
    "homopolymer",
    "trinucleotide_repeat",
    "intrinsic_tm",
    "dinucleotide_repeat",
    "forbidden_motif",
    "self_complementarity",
    "near_palindrome",
]

PROGRESS_UPDATE_INTERVAL_SECONDS = 30.0
OUTPUT_FILE_KEYS = (
    "final_csv",
    "pairings_csv",
    "unpaired_csv",
    "summary_json",
    "checkpoint_file",
    "resolved_config_yaml",
)


def _latest_run_pointer(base_dir: Path) -> Path:
    return base_dir / "latest_run.txt"


def _resolve_run_directory(config: dict, resume: bool) -> Path:
    root = Path(config["_meta"]["config_path"]).parent
    output = config["output"]
    base_dir = (root / str(output["root_dir"])).resolve()
    if resume:
        pointer = _latest_run_pointer(base_dir)
        if not pointer.exists():
            raise ConfigError(f"No previous run directory recorded at: {pointer}")
        run_dir = Path(pointer.read_text(encoding="utf-8").strip())
        if not run_dir.exists():
            raise ConfigError(f"Recorded run directory does not exist: {run_dir}")
        return run_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = base_dir / f"{timestamp}_{suffix:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    pointer = _latest_run_pointer(base_dir)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(run_dir), encoding="utf-8")
    return run_dir


def _normalize_output_paths(config: dict, resume: bool) -> dict[str, Path]:
    output = config["output"]
    run_dir = _resolve_run_directory(config, resume)
    paths = {key: (run_dir / str(output[key])).resolve() for key in OUTPUT_FILE_KEYS if output.get(key)}
    paths["run_dir"] = run_dir
    return paths


def _build_filter_settings(config: dict) -> FilterSettings:
    motifs = list(config["forbidden_motifs"]["motifs"])
    if config["forbidden_motifs"].get("include_reverse_complements", True):
        motifs.extend(reverse_complement(motif) for motif in list(motifs))
    return FilterSettings(
        primer_length=int(config["generator"]["primer_length"]),
        gc_min_fraction=float(config["composition"]["gc_min_fraction"]),
        gc_max_fraction=float(config["composition"]["gc_max_fraction"]),
        intrinsic_tm_min_c=float(config["filters"]["intrinsic_tm_min_c"]),
        intrinsic_tm_max_c=float(config["filters"]["intrinsic_tm_max_c"]),
        thermo_config=dict(config["thermodynamics"]),
        max_homopolymer=int(config["filters"]["max_homopolymer"]),
        max_dinucleotide_repeats=int(config["filters"]["max_dinucleotide_repeats"]),
        max_trinucleotide_repeats=int(config["filters"]["max_trinucleotide_repeats"]),
        near_palindrome_min_match=int(config["filters"]["near_palindrome_min_match"]),
        self_complementarity_run_threshold=int(config["filters"]["self_complementarity_run_threshold"]),
        forbidden_motifs=tuple(sorted(set(motifs))),
    )


def _gc_count_range(length: int, gc_min_fraction: float, gc_max_fraction: float) -> tuple[int, int]:
    return round(length * gc_min_fraction), round(length * gc_max_fraction)


def _can_embed_gc_count_in_full_length(
    sequence_gc_count: int,
    sequence_length: int,
    full_length: int,
    gc_min_fraction: float,
    gc_max_fraction: float,
) -> bool:
    if sequence_length > full_length:
        return False
    min_full_gc, max_full_gc = _gc_count_range(full_length, gc_min_fraction, gc_max_fraction)
    remaining_length = full_length - sequence_length
    min_possible_full_gc = sequence_gc_count
    max_possible_full_gc = sequence_gc_count + remaining_length
    return max(min_full_gc, min_possible_full_gc) <= min(max_full_gc, max_possible_full_gc)


def _sample_gc_balanced_candidate(rng: random.Random, length: int, gc_min_fraction: float, gc_max_fraction: float) -> str:
    min_gc, max_gc = _gc_count_range(length, gc_min_fraction, gc_max_fraction)
    gc_count = rng.randint(min_gc, max_gc)
    at_count = length - gc_count
    bases = [rng.choice("GC") for _ in range(gc_count)] + [rng.choice("AT") for _ in range(at_count)]
    rng.shuffle(bases)
    return "".join(bases)


def _relevant_three_prime_kmers(sequence: str, kmer_length: int, three_prime_window_nt: int) -> set[str]:
    if kmer_length <= 0 or kmer_length > len(sequence):
        return set()
    three_prime_start = max(0, len(sequence) - max(1, three_prime_window_nt))
    start_min = max(0, three_prime_start - kmer_length + 1)
    start_max = len(sequence) - kmer_length
    return {sequence[start : start + kmer_length] for start in range(start_min, start_max + 1)}


@dataclass
class _PruningKmerIndex:
    min_alignment_run: int
    tm_trigger_run: int
    three_prime_window_nt: int
    min_run_index: dict[str, set[str]]
    tm_run_index: dict[str, set[str]] | None

    @classmethod
    def from_thermo_config(cls, thermo_config: dict[str, float]) -> "_PruningKmerIndex":
        min_alignment_run = max(1, int(thermo_config.get("min_alignment_run", 4)))
        tm_trigger_run = max(
            min_alignment_run,
            int(
                thermo_config.get(
                    "min_tm_calculation_run_length_reverse_complement",
                    thermo_config.get("min_tm_calculation_run_length", 0),
                )
            ),
        )
        tm_run_index = {} if tm_trigger_run > min_alignment_run else None
        return cls(
            min_alignment_run=min_alignment_run,
            tm_trigger_run=tm_trigger_run,
            three_prime_window_nt=max(1, int(thermo_config.get("three_prime_window_nt", 3))),
            min_run_index={},
            tm_run_index=tm_run_index,
        )

    def add_record(self, record: PrimerRecord) -> None:
        self._add_sequence_to_index(record.rc_sequence, record.sequence, self.min_alignment_run, self.min_run_index)
        if self.tm_run_index is not None:
            self._add_sequence_to_index(record.rc_sequence, record.sequence, self.tm_trigger_run, self.tm_run_index)

    def remove_record(self, record: PrimerRecord) -> None:
        self._remove_sequence_from_index(record.rc_sequence, record.sequence, self.min_alignment_run, self.min_run_index)
        if self.tm_run_index is not None:
            self._remove_sequence_from_index(record.rc_sequence, record.sequence, self.tm_trigger_run, self.tm_run_index)

    def build_candidate_sets(self, accepted: list[PrimerRecord]) -> tuple[dict[int, set[int]], dict[int, set[int]] | None]:
        sequence_to_index = {record.sequence: index for index, record in enumerate(accepted)}
        min_run_candidates: dict[int, set[int]] = {}
        tm_run_candidates: dict[int, set[int]] | None = {} if self.tm_run_index is not None else None
        for left_index, record in enumerate(accepted):
            min_candidate_indices: set[int] = set()
            for candidate_sequence in self._candidate_sequences(record.sequence, self.min_alignment_run, self.min_run_index):
                candidate_index = sequence_to_index.get(candidate_sequence)
                if candidate_index is not None and candidate_index != left_index:
                    min_candidate_indices.add(candidate_index)
            min_run_candidates[left_index] = min_candidate_indices
            if self.tm_run_index is not None and tm_run_candidates is not None:
                tm_candidate_indices: set[int] = set()
                for candidate_sequence in self._candidate_sequences(record.sequence, self.tm_trigger_run, self.tm_run_index):
                    candidate_index = sequence_to_index.get(candidate_sequence)
                    if candidate_index is not None and candidate_index != left_index:
                        tm_candidate_indices.add(candidate_index)
                tm_run_candidates[left_index] = tm_candidate_indices
        return min_run_candidates, tm_run_candidates

    def bulk_add_records(self, records: list[PrimerRecord]) -> None:
        if not records:
            return
        progress_meter = _WallclockProgressMeter()
        print(f"Building pruning k-mer index: records=0/{len(records)}")
        for index, record in enumerate(records, start=1):
            self.add_record(record)
            if index == len(records) or progress_meter.should_emit():
                print(f"K-mer index progress: records={index}/{len(records)}")
        print(f"Completed pruning k-mer index build: records={len(records)}/{len(records)}")

    def bulk_remove_records(self, records: list[PrimerRecord]) -> None:
        if not records:
            return
        progress_meter = _WallclockProgressMeter()
        print(f"Updating pruning k-mer index for removals: records=0/{len(records)}")
        for index, record in enumerate(records, start=1):
            self.remove_record(record)
            if index == len(records) or progress_meter.should_emit():
                print(f"K-mer index removal progress: records={index}/{len(records)}")
        print(f"Completed pruning k-mer index removal update: records={len(records)}/{len(records)}")

    def build_candidate_sets_with_progress(
        self,
        accepted: list[PrimerRecord],
    ) -> tuple[dict[int, set[int]], dict[int, set[int]] | None]:
        if not accepted:
            return {}, {} if self.tm_run_index is not None else None
        sequence_to_index = {record.sequence: index for index, record in enumerate(accepted)}
        min_run_candidates: dict[int, set[int]] = {}
        tm_run_candidates: dict[int, set[int]] | None = {} if self.tm_run_index is not None else None
        progress_meter = _WallclockProgressMeter()
        print(f"Building pruning candidate sets: primers=0/{len(accepted)}")
        for left_index, record in enumerate(accepted):
            min_candidate_indices: set[int] = set()
            for candidate_sequence in self._candidate_sequences(record.sequence, self.min_alignment_run, self.min_run_index):
                candidate_index = sequence_to_index.get(candidate_sequence)
                if candidate_index is not None and candidate_index != left_index:
                    min_candidate_indices.add(candidate_index)
            min_run_candidates[left_index] = min_candidate_indices
            if self.tm_run_index is not None and tm_run_candidates is not None:
                tm_candidate_indices: set[int] = set()
                for candidate_sequence in self._candidate_sequences(record.sequence, self.tm_trigger_run, self.tm_run_index):
                    candidate_index = sequence_to_index.get(candidate_sequence)
                    if candidate_index is not None and candidate_index != left_index:
                        tm_candidate_indices.add(candidate_index)
                tm_run_candidates[left_index] = tm_candidate_indices
            completed = left_index + 1
            if completed == len(accepted) or progress_meter.should_emit():
                print(f"Candidate-set progress: primers={completed}/{len(accepted)}")
        print(f"Completed pruning candidate-set build: primers={len(accepted)}/{len(accepted)}")
        return min_run_candidates, tm_run_candidates

    def _candidate_sequences(
        self,
        sequence: str,
        kmer_length: int,
        index: dict[str, set[str]],
    ) -> set[str]:
        candidate_sequences: set[str] = set()
        for kmer in _relevant_three_prime_kmers(sequence, kmer_length, self.three_prime_window_nt):
            candidate_sequences.update(index.get(kmer, set()))
        return candidate_sequences

    @staticmethod
    def _add_sequence_to_index(
        sequence: str,
        sequence_id: str,
        kmer_length: int,
        index: dict[str, set[str]],
    ) -> None:
        if kmer_length <= 0 or kmer_length > len(sequence):
            return
        for start in range(len(sequence) - kmer_length + 1):
            kmer = sequence[start : start + kmer_length]
            index.setdefault(kmer, set()).add(sequence_id)

    @staticmethod
    def _remove_sequence_from_index(
        sequence: str,
        sequence_id: str,
        kmer_length: int,
        index: dict[str, set[str]],
    ) -> None:
        if kmer_length <= 0 or kmer_length > len(sequence):
            return
        for start in range(len(sequence) - kmer_length + 1):
            kmer = sequence[start : start + kmer_length]
            members = index.get(kmer)
            if not members:
                continue
            members.discard(sequence_id)
            if not members:
                del index[kmer]


def _count_candidate_pairs(
    candidate_indices_by_left: dict[int, set[int]],
    sequence_count: int,
    seed_count: int,
) -> int:
    total = 0
    for left_index in range(sequence_count):
        for right_index in candidate_indices_by_left.get(left_index, set()):
            if right_index <= left_index:
                continue
            if left_index < seed_count and right_index < seed_count:
                continue
            total += 1
    return total


def _checkpoint_payload(config: dict, accepted: list[PrimerRecord], counters: Counter, attempts: int, rng: random.Random) -> dict:
    return {
        "config_digest": config["_meta"]["config_digest"],
        "config_snapshot": effective_config(config),
        "accepted_sequences": [record.sequence for record in accepted],
        "counters": dict(counters),
        "attempts": attempts,
        "rng_state": rng.getstate(),
    }


def _restore_from_checkpoint(
    config: dict,
    paths: dict[str, Path],
    accepted: list[PrimerRecord],
    counters: Counter,
    rng: random.Random,
) -> int:
    checkpoint_path = paths["checkpoint_file"]
    if not checkpoint_path.exists():
        raise ConfigError(f"Checkpoint file does not exist: {checkpoint_path}")
    payload = load_checkpoint(checkpoint_path)
    if payload["config_digest"] != config["_meta"]["config_digest"]:
        payload_snapshot = payload.get("config_snapshot")
        if payload_snapshot is not None:
            current_effective = normalize_effective_config(effective_config(config))
            checkpoint_effective = normalize_effective_config(payload_snapshot)
            if checkpoint_effective != current_effective:
                raise ConfigError("Checkpoint configuration does not match the current config.yaml.")
        else:
            print("Warning: resuming from a legacy checkpoint without a config snapshot; skipping strict config compatibility validation.")
    for sequence in payload["accepted_sequences"]:
        accepted.append(
            PrimerRecord(
                sequence=sequence,
                rc_sequence=reverse_complement(sequence),
                gc_fraction=(sequence.count("G") + sequence.count("C")) / len(sequence),
                intrinsic_tm_c=intrinsic_tm_with_config_c(sequence, config["thermodynamics"]),
            )
        )
    counters.update(payload["counters"])
    rng.setstate(payload["rng_state"])
    return int(payload["attempts"])


def _rows_from_records(records: list[PrimerRecord]) -> list[dict[str, object]]:
    return [
        {
            "sequence": record.sequence,
            "reverse_complement": record.rc_sequence,
            "gc_fraction": round(record.gc_fraction, 4),
            "intrinsic_tm_c": round(record.intrinsic_tm_c, 4),
        }
        for record in records
    ]


def _rows_from_pairings(pairings: list[PairingRecord]) -> list[dict[str, object]]:
    return [
        {
            "pair_index": index + 1,
            "left_sequence": pairing.left_sequence,
            "right_sequence": pairing.right_sequence,
            "left_vs_right_sense_tm_c": pairing.left_vs_right_sense_tm_c,
            "right_vs_left_sense_tm_c": pairing.right_vs_left_sense_tm_c,
            "pair_tm_c": pairing.pair_tm_c,
        }
        for index, pairing in enumerate(pairings)
    ]


def _print_startup_summary(config: dict, paths: dict[str, Path], resume: bool) -> None:
    generator_config = config["generator"]
    thermo_config = config["thermodynamics"]
    pairing_config = config["pairing"]
    print("Starting orthogonal primer generation")
    print(f"  mode: {'resume' if resume else 'new run'}")
    print(f"  config: {config['_meta']['config_path']}")
    print(f"  output_dir: {paths['run_dir']}")
    print(
        "  target:"
        f" size={generator_config['target_size']},"
        f" length={generator_config['primer_length']},"
        f" max_attempts={generator_config['max_attempts']}"
    )
    print(
        "  constraints:"
        f" intrinsic_tm_min_c={config['filters']['intrinsic_tm_min_c']},"
        f" intrinsic_tm_max_c={config['filters']['intrinsic_tm_max_c']},"
        f" pair_tm_max_c={thermo_config['pair_tm_max_c']},"
        f" three_prime_window_nt={thermo_config.get('three_prime_window_nt', 3)},"
        f" precompute_shortest_threshold_seq={bool(thermo_config.get('precompute_shortest_threshold_seq', False))},"
        f" pairing_enabled={bool(pairing_config.get('enabled', False))}"
    )


def _print_progress(attempts: int, max_attempts: int, accepted_count: int, target_size: int, counters: Counter) -> None:
    print(
        "Progress:"
        f" attempts={attempts} ({_format_rate(attempts, max_attempts):.1%}),"
        f" accepted={accepted_count}/{target_size},"
        f" acceptance_rate={_format_rate(accepted_count, attempts):.4f},"
        f" duplicate_proposals={counters.get('duplicate', 0)}"
    )


def _print_completion_summary(
    attempts: int,
    cycle_count: int,
    accepted_count: int,
    final_count: int,
    pairing_count: int,
    unpaired_count: int,
    trimmed_excess_count: int,
    num_pairs_below_min_alignment_run: int,
    num_tms_shortcircuited: int,
    num_tmnn_evaluations_performed: int,
    elapsed_wallclock_seconds: float,
    conflicts: list[dict[str, float | int | str]],
    paths: dict[str, Path],
) -> None:
    print("Completed orthogonal primer generation")
    print(
        "  results:"
        f" attempts={attempts},"
        f" cycles={cycle_count},"
        f" accepted_before_pruning={accepted_count},"
        f" final_validated={final_count},"
        f" paired_primers={pairing_count * 2},"
        f" unpaired_primers={unpaired_count},"
        f" trimmed_excess={trimmed_excess_count},"
        f" num_pairs_below_min_alignment_run={num_pairs_below_min_alignment_run},"
        f" num_Tms_shortcircuited={num_tms_shortcircuited},"
        f" num_TmNN_evaluations_performed={num_tmnn_evaluations_performed},"
        f" wallclock_seconds={elapsed_wallclock_seconds:.2f},"
        f" conflicts={len(conflicts)}"
    )
    print(
        "  outputs:"
        f" final_csv={paths['final_csv']},"
        f" summary_json={paths['summary_json']}"
    )


def _cycle_generation_limits(
    current_validated_count: int,
    target_size: int,
    force_target_size: bool,
    generator_config: dict[str, object],
) -> tuple[int, int, float]:
    needed = max(0, target_size - current_validated_count)
    if not force_target_size or needed == 0:
        return target_size, 0, 0.0
    oversample_multiplier = float(generator_config.get("force_target_cycle_oversample_multiplier", 10.0))
    post_target_seconds = float(generator_config.get("force_target_post_target_seconds", 30.0))
    cycle_accept_limit = current_validated_count + math.ceil(needed * oversample_multiplier)
    return cycle_accept_limit, needed, post_target_seconds


def _build_generation_diagnostics(
    counters: Counter,
    attempts: int,
    accepted_count: int,
    target_size: int,
) -> dict[str, object]:
    rejection_counts = {name: int(counters.get(name, 0)) for name in FILTER_FUNNEL_ORDER}
    remaining = attempts
    funnel_failures: list[dict[str, object]] = []
    for name in FILTER_FUNNEL_ORDER:
        count = rejection_counts[name]
        funnel_failures.append(
            {
                "filter": name,
                "count": count,
                "fraction_of_attempts": _format_rate(count, attempts),
                "attempts_reaching_filter": remaining,
                "fraction_of_reaching_filter": _format_rate(count, remaining),
            }
        )
        remaining -= count
    return {
        "target_reached": accepted_count >= target_size,
        "attempted_candidates": attempts,
        "accepted_count": accepted_count,
        "target_size": target_size,
        "acceptance_rate": _format_rate(accepted_count, attempts),
        "rejection_counts": rejection_counts,
        "funnel_failures": funnel_failures,
    }


def _print_generation_diagnostics(diagnostics: dict[str, object]) -> None:
    print("Generation diagnostics:")
    print(
        "  totals:"
        f" accepted={diagnostics['accepted_count']}/{diagnostics['target_size']},"
        f" attempts={diagnostics['attempted_candidates']},"
        f" acceptance_rate={diagnostics['acceptance_rate']:.4f}"
    )
    funnel_failures = diagnostics["funnel_failures"]
    if not funnel_failures:
        print("  no rejected candidates were recorded")
        return
    for item in funnel_failures:
        print(
            "  filter:"
            f" {item['filter']} ->"
            f" failed={item['count']},"
            f" of_attempts={item['fraction_of_attempts']:.4%},"
            f" of_reaching_filter={item['fraction_of_reaching_filter']:.4%}"
        )


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


def _make_validation_progress_logger(total_pairs: int):
    if total_pairs <= 0:
        return None
    meter = _WallclockProgressMeter()

    def callback(pairs_checked: int, _: int, conflict_count: int) -> None:
        if pairs_checked == total_pairs or meter.should_emit():
            print(
                "Validation progress:"
                f" pairs_checked={pairs_checked}/{total_pairs},"
                f" conflicts_found={conflict_count}"
            )

    return callback


def _make_pruning_progress_logger(total_nodes: int):
    if total_nodes <= 0:
        return None
    meter = _WallclockProgressMeter()

    def callback(removed: int, _: int, remaining: int) -> None:
        if removed == total_nodes or meter.should_emit():
            print(
                "Pruning progress:"
                f" removed={removed},"
                f" remaining={remaining}"
            )

    return callback


def _print_pairing_progress(
    attempts: int,
    max_attempts: int,
    paired_count: int,
    unpaired_count: int,
    recent_rejection_rate: float,
    recycled_pair_count: int,
) -> None:
    print(
        "Pairing progress:"
        f" attempts={attempts}/{max_attempts},"
        f" paired={paired_count},"
        f" unpaired={unpaired_count},"
        f" recent_rejection_rate={recent_rejection_rate:.2%},"
        f" recycled_pairs={recycled_pair_count}"
    )


def _sense_pairing_details(
    left: PrimerRecord,
    right: PrimerRecord,
    thermo_runtime_config: dict[str, float],
) -> tuple[dict[str, float | int | str], dict[str, float | int | str], float]:
    left_details = sense_score_details(left.sequence, right.sequence, thermo_runtime_config)
    right_details = sense_score_details(right.sequence, left.sequence, thermo_runtime_config)
    pair_tm_c = max(float(left_details["tm_c"]), float(right_details["tm_c"]))
    return left_details, right_details, pair_tm_c


def _pair_primers(
    records: list[PrimerRecord],
    config: dict,
    thermo_runtime_config: dict[str, float],
    rng: random.Random,
) -> dict[str, object]:
    pairing_config = config["pairing"]
    if not pairing_config.get("enabled", False):
        return {
            "enabled": False,
            "completed": False,
            "pairings": [],
            "unpaired_records": list(records),
            "delivered_records": list(records),
            "attempts": 0,
            "accepted_pairs": 0,
            "rejected_pairs": 0,
            "recycled_pairs": 0,
            "recent_rejection_rate": 0.0,
            "sense_tm_max_c": None,
            "num_pairing_tms_shortcircuited": 0,
            "discarded_surplus_primers": 0,
            "target_pair_count": 0,
            "target_pair_count_reached": False,
        }

    sense_tm_max_c = pairing_config.get("sense_tm_max_c")
    if sense_tm_max_c is None:
        sense_tm_max_c = thermo_runtime_config["pair_tm_max_c"]
    sense_tm_max_c = float(sense_tm_max_c)
    target_pair_count = int(config["generator"]["target_size"]) // 2
    rejection_window = int(pairing_config["rejection_window"])
    requeue_probability_scale = float(pairing_config["requeue_probability_scale"])
    max_attempts = max(1, len(records) * int(pairing_config["max_attempts_multiplier"]))
    stagnation_attempt_window = int(pairing_config["stagnation_attempt_window"])
    pool = list(records)
    pairings: list[PairingRecord] = []
    recent_rejections: deque[int] = deque(maxlen=rejection_window)
    attempts = 0
    rejected_pairs = 0
    recycled_pairs = 0
    attempts_since_best = 0
    best_unpaired_count = len(pool)
    progress_meter = _WallclockProgressMeter()
    num_pairing_tms_shortcircuited = 0

    print(
        "Starting pairing phase:"
        f" primers={len(pool)},"
        f" target_pairs={target_pair_count},"
        f" sense_tm_max_c={sense_tm_max_c},"
        f" max_attempts={max_attempts}"
    )

    while (
        len(pool) >= 2
        and attempts < max_attempts
        and attempts_since_best < stagnation_attempt_window
        and len(pairings) < target_pair_count
    ):
        rejection_rate = sum(recent_rejections) / len(recent_rejections) if recent_rejections else 0.0
        if pairings:
            recycle_probability = max(0.0, min(1.0, rejection_rate * requeue_probability_scale))
            if rng.random() < recycle_probability:
                recycled_pair_index = rng.randrange(len(pairings))
                recycled_pair = pairings.pop(recycled_pair_index)
                pool.append(_make_primer_record(recycled_pair.left_sequence, thermo_runtime_config))
                pool.append(_make_primer_record(recycled_pair.right_sequence, thermo_runtime_config))
                recycled_pairs += 1

        left_index = rng.randrange(len(pool))
        left = pool.pop(left_index)
        right_index = rng.randrange(len(pool))
        right = pool.pop(right_index)
        attempts += 1

        left_details, right_details, pair_tm_c = _sense_pairing_details(left, right, thermo_runtime_config)
        if left_details.get("tm_shortcircuited", False):
            num_pairing_tms_shortcircuited += 1
        if right_details.get("tm_shortcircuited", False):
            num_pairing_tms_shortcircuited += 1

        if pair_tm_c <= sense_tm_max_c:
            pairings.append(
                PairingRecord(
                    left_sequence=left.sequence,
                    right_sequence=right.sequence,
                    left_vs_right_sense_tm_c=float(left_details["tm_c"]),
                    right_vs_left_sense_tm_c=float(right_details["tm_c"]),
                    pair_tm_c=pair_tm_c,
                )
            )
            recent_rejections.append(0)
            if len(pool) < best_unpaired_count:
                best_unpaired_count = len(pool)
                attempts_since_best = 0
            else:
                attempts_since_best += 1
        else:
            pool.extend([left, right])
            rejected_pairs += 1
            recent_rejections.append(1)
            attempts_since_best += 1

        if progress_meter.should_emit():
            _print_pairing_progress(
                attempts,
                max_attempts,
                len(pairings),
                len(pool),
                sum(recent_rejections) / len(recent_rejections) if recent_rejections else 0.0,
                recycled_pairs,
            )

    target_pair_count_reached = len(pairings) >= target_pair_count
    discarded_surplus_primers = len(pool) if target_pair_count_reached else 0
    delivered_records: list[PrimerRecord] = []
    for pairing in pairings:
        delivered_records.append(_make_primer_record(pairing.left_sequence, thermo_runtime_config))
        delivered_records.append(_make_primer_record(pairing.right_sequence, thermo_runtime_config))
    completed = target_pair_count_reached or len(pool) <= 1
    final_rejection_rate = sum(recent_rejections) / len(recent_rejections) if recent_rejections else 0.0
    print(
        "Completed pairing phase:"
        f" completed={completed},"
        f" pairs={len(pairings)},"
        f" unpaired={0 if target_pair_count_reached else len(pool)},"
        f" discarded_surplus={discarded_surplus_primers},"
        f" attempts={attempts},"
        f" rejection_rate={final_rejection_rate:.2%},"
        f" recycled_pairs={recycled_pairs}"
    )
    return {
        "enabled": True,
        "completed": completed,
        "pairings": pairings,
        "unpaired_records": [] if target_pair_count_reached else pool,
        "delivered_records": delivered_records,
        "attempts": attempts,
        "accepted_pairs": len(pairings),
        "rejected_pairs": rejected_pairs,
        "recycled_pairs": recycled_pairs,
        "recent_rejection_rate": final_rejection_rate,
        "sense_tm_max_c": sense_tm_max_c,
        "num_pairing_tms_shortcircuited": num_pairing_tms_shortcircuited,
        "discarded_surplus_primers": discarded_surplus_primers,
        "target_pair_count": target_pair_count,
        "target_pair_count_reached": target_pair_count_reached,
    }


def _passes_precompute_local_filters(sequence: str, config: dict) -> bool:
    composition = config["composition"]
    filters = config["filters"]
    full_length = int(config["generator"]["primer_length"])
    gc_min_fraction = float(composition["gc_min_fraction"])
    gc_max_fraction = float(composition["gc_max_fraction"])
    gc_count = sequence.count("G") + sequence.count("C")
    if not _can_embed_gc_count_in_full_length(
        gc_count,
        len(sequence),
        full_length,
        gc_min_fraction,
        gc_max_fraction,
    ):
        return False
    if exceeds_homopolymer_limit(sequence, int(filters["max_homopolymer"])):
        return False
    if exceeds_repeat_limit(sequence, 2, int(filters["max_dinucleotide_repeats"])):
        return False
    if exceeds_repeat_limit(sequence, 3, int(filters["max_trinucleotide_repeats"])):
        return False
    return True


def _print_precompute_progress(
    current_length: int,
    tested_total: int,
    tested_this_length: int,
    valid_this_length: int,
    best_tm_this_length: float,
    resolved_lengths: dict[str, int] | None = None,
) -> None:
    best_tm_text = f"{best_tm_this_length:.2f}" if best_tm_this_length != float("-inf") else "-inf"
    resolved_text = ""
    if resolved_lengths:
        resolved_text = ", resolved=" + ",".join(f"{label}:{length}" for label, length in sorted(resolved_lengths.items()))
    print(
        "Precompute progress:"
        f" length={current_length},"
        f" tested_total={tested_total},"
        f" tested_this_length={tested_this_length},"
        f" valid_this_length={valid_this_length},"
        f" best_tm_this_length={best_tm_text}"
        f"{resolved_text}"
    )


def _precompute_shortest_threshold_seq_lengths(
    config: dict,
    thresholds_by_label: dict[str, float],
) -> tuple[dict[str, int], bool]:
    thermo_config = config["thermodynamics"]
    meter = _WallclockProgressMeter()
    tested_total = 0
    current_length = 1
    interrupted = False
    if not thresholds_by_label:
        return {}, False

    unresolved_labels = set(thresholds_by_label)
    resolved_lengths: dict[str, int] = {}
    triggering_sequences: dict[str, tuple[str, float]] = {}

    print(
        "Starting shortest-threshold-sequence precompute:"
        f" thresholds={{{', '.join(f'{label}:{thresholds_by_label[label]}' for label in sorted(thresholds_by_label))}}}"
    )
    while True:
        tested_this_length = 0
        valid_this_length = 0
        best_tm_this_length = float("-inf")
        try:
            for bases in itertools.product("ACGT", repeat=current_length):
                sequence = "".join(bases)
                tested_total += 1
                tested_this_length += 1
                if not _passes_precompute_local_filters(sequence, config):
                    if meter.should_emit():
                        _print_precompute_progress(
                            current_length,
                            tested_total,
                            tested_this_length,
                            valid_this_length,
                            best_tm_this_length,
                            resolved_lengths,
                        )
                    continue
                valid_this_length += 1
                tm_c = intrinsic_tm_with_config_c(sequence, thermo_config)
                if tm_c > best_tm_this_length:
                    best_tm_this_length = tm_c
                for label in tuple(sorted(unresolved_labels)):
                    if tm_c > thresholds_by_label[label]:
                        resolved_lengths[label] = current_length
                        triggering_sequences[label] = (sequence, tm_c)
                        unresolved_labels.remove(label)
                if not unresolved_labels:
                    for label in sorted(resolved_lengths):
                        trigger_sequence, trigger_tm_c = triggering_sequences[label]
                        print(
                            "Completed shortest-threshold-sequence precompute:"
                            f" label={label},"
                            f" min_tm_calculation_run_length={resolved_lengths[label]},"
                            f" triggering_sequence={trigger_sequence},"
                            f" triggering_tm_c={trigger_tm_c:.2f}"
                        )
                    return resolved_lengths, False
                if meter.should_emit():
                    _print_precompute_progress(
                        current_length,
                        tested_total,
                        tested_this_length,
                        valid_this_length,
                        best_tm_this_length,
                        resolved_lengths,
                    )
        except KeyboardInterrupt:
            interrupted = True
            fallback_lengths = {
                label: resolved_lengths.get(label, current_length) for label in thresholds_by_label
            }
            print(
                "Precompute interrupt received:"
                f" using_min_tm_calculation_run_lengths={{{', '.join(f'{label}:{fallback_lengths[label]}' for label in sorted(fallback_lengths))}}}"
            )
            break
        _print_precompute_progress(
            current_length,
            tested_total,
            tested_this_length,
            valid_this_length,
            best_tm_this_length,
            resolved_lengths,
        )
        current_length += 1

    return {label: resolved_lengths.get(label, current_length) for label in thresholds_by_label}, interrupted


def run_generation(config: dict, resume: bool = False) -> None:
    run_start_time = time.monotonic()
    generator_config = config["generator"]
    thermo_runtime_config = dict(config["thermodynamics"])
    paths = _normalize_output_paths(config, resume)
    filter_settings = _build_filter_settings(config)
    rng = random.Random(generator_config.get("random_seed"))
    accepted: list[PrimerRecord] = []
    seen_sequences: set[str] = set()
    counters: Counter = Counter()
    attempts = 0
    generation_progress_meter = _WallclockProgressMeter()

    _print_startup_summary(config, paths, resume)

    precomputed_min_tm_run_length = 0
    precomputed_min_tm_run_length_reverse_complement = 0
    precomputed_min_tm_run_length_sense = 0
    precompute_interrupted = False
    if bool(config["thermodynamics"].get("precompute_shortest_threshold_seq", False)):
        precompute_thresholds = {
            "reverse_complement": float(config["thermodynamics"]["pair_tm_max_c"]),
        }
        if bool(config["pairing"].get("enabled", False)):
            sense_tm_threshold = config["pairing"].get("sense_tm_max_c")
            if sense_tm_threshold is None:
                sense_tm_threshold = config["thermodynamics"]["pair_tm_max_c"]
            precompute_thresholds["sense"] = float(sense_tm_threshold)
        precomputed_lengths, precompute_interrupted = _precompute_shortest_threshold_seq_lengths(
            config,
            precompute_thresholds,
        )
        precomputed_min_tm_run_length_reverse_complement = precomputed_lengths["reverse_complement"]
        thermo_runtime_config["min_tm_calculation_run_length_reverse_complement"] = (
            precomputed_min_tm_run_length_reverse_complement
        )
        thermo_runtime_config["min_tm_calculation_run_length"] = precomputed_min_tm_run_length_reverse_complement
        precomputed_min_tm_run_length = precomputed_min_tm_run_length_reverse_complement
        print(
            "Using thermo run-length gate:"
            f" reverse_complement_min_tm_calculation_run_length={precomputed_min_tm_run_length_reverse_complement}"
        )
        if "sense" in precomputed_lengths:
            precomputed_min_tm_run_length_sense = precomputed_lengths["sense"]
            thermo_runtime_config["min_tm_calculation_run_length_sense"] = precomputed_min_tm_run_length_sense
            print(
                "Using thermo run-length gate:"
                f" sense_min_tm_calculation_run_length={precomputed_min_tm_run_length_sense}"
            )

    pruning_kmer_index = _PruningKmerIndex.from_thermo_config(thermo_runtime_config)

    if resume:
        attempts = _restore_from_checkpoint(config, paths, accepted, counters, rng)
        seen_sequences.update(record.sequence for record in accepted)
        pruning_kmer_index.bulk_add_records(accepted)
        print(f"  restored checkpoint: accepted={len(accepted)}, attempts={attempts}")

    primer_length = int(generator_config["primer_length"])
    target_size = int(generator_config["target_size"])
    max_attempts = int(generator_config["max_attempts"])
    checkpoint_interval = int(generator_config["checkpoint_interval"])
    force_target_size = bool(generator_config.get("force_target_size", False))
    cycle_count = 0
    diagnostics: dict[str, object] = {}
    conflicts: list[dict[str, float | int | str]] = []
    graph: dict[int, set[int]] = {}
    final_records: list[PrimerRecord] = []
    trimmed_excess_count = 0
    interrupted = False
    cycle_accept_limit = target_size
    num_pairs_below_min_alignment_run = 0
    num_tms_shortcircuited = 0
    num_tmnn_evaluations_performed = 0
    pairing_summary: dict[str, object] = {
        "enabled": False,
        "completed": False,
        "pairings": [],
        "unpaired_records": [],
        "attempts": 0,
        "accepted_pairs": 0,
        "rejected_pairs": 0,
        "recycled_pairs": 0,
        "recent_rejection_rate": 0.0,
        "sense_tm_max_c": None,
        "num_pairing_tms_shortcircuited": 0,
    }

    while True:
        cycle_count += 1
        cycle_seed_count = len(accepted)
        cycle_accept_limit, cycle_needed, post_target_seconds = _cycle_generation_limits(
            cycle_seed_count,
            target_size,
            force_target_size,
            generator_config,
        )
        print(
            "Starting generation cycle:"
            f" cycle={cycle_count},"
            f" seed_pool={len(accepted)},"
            f" attempts_used={attempts}/{max_attempts},"
            f" cycle_accept_limit={cycle_accept_limit}"
        )
        if force_target_size and not interrupted:
            print(
                "  force-target cycle strategy:"
                f" needed={cycle_needed},"
                f" oversample_multiplier={float(generator_config.get('force_target_cycle_oversample_multiplier', 10.0)):.2f},"
                f" post_target_seconds={post_target_seconds:.1f}"
            )

        try:
            target_reached_at: float | None = None
            while attempts < max_attempts:
                if not force_target_size and len(accepted) >= target_size:
                    break
                if force_target_size:
                    if len(accepted) >= cycle_accept_limit:
                        break
                    if len(accepted) >= target_size:
                        if target_reached_at is None:
                            target_reached_at = time.monotonic()
                        elif time.monotonic() - target_reached_at >= post_target_seconds:
                            break
                attempts += 1
                candidate = _sample_gc_balanced_candidate(
                    rng,
                    primer_length,
                    float(config["composition"]["gc_min_fraction"]),
                    float(config["composition"]["gc_max_fraction"]),
                )
                if candidate in seen_sequences:
                    counters["duplicate"] += 1
                    if generation_progress_meter.should_emit():
                        _print_progress(attempts, max_attempts, len(accepted), target_size, counters)
                    continue
                failure_reason = filter_candidate(candidate, filter_settings)
                if failure_reason:
                    counters[failure_reason] += 1
                    seen_sequences.add(candidate)
                    if generation_progress_meter.should_emit():
                        _print_progress(attempts, max_attempts, len(accepted), target_size, counters)
                    continue
                record = PrimerRecord(
                    sequence=candidate,
                    rc_sequence=reverse_complement(candidate),
                    gc_fraction=(candidate.count("G") + candidate.count("C")) / primer_length,
                    intrinsic_tm_c=intrinsic_tm_with_config_c(candidate, thermo_runtime_config),
                )
                accepted.append(record)
                pruning_kmer_index.add_record(record)
                seen_sequences.add(candidate)
                counters["accepted"] += 1
                if force_target_size and len(accepted) >= target_size and target_reached_at is None:
                    target_reached_at = time.monotonic()
                if checkpoint_interval and len(accepted) % checkpoint_interval == 0:
                    save_checkpoint(paths["checkpoint_file"], _checkpoint_payload(config, accepted, counters, attempts, rng))
                if generation_progress_meter.should_emit():
                    _print_progress(attempts, max_attempts, len(accepted), target_size, counters)
        except KeyboardInterrupt:
            interrupted = True
            print(
                "Manual interrupt received:"
                f" proceeding to validation/pruning with accepted={len(accepted)}"
            )

        save_checkpoint(paths["checkpoint_file"], _checkpoint_payload(config, accepted, counters, attempts, rng))
        diagnostics = _build_generation_diagnostics(
            counters,
            attempts,
            len(accepted),
            target_size,
        )
        if len(accepted) < target_size:
            print(
                "Generation stopped before target size:"
                f" accepted={len(accepted)}/{target_size},"
                f" attempts={attempts}/{max_attempts}"
            )
        else:
            print(f"Generation reached target size: accepted={len(accepted)}/{target_size}")
        _print_generation_diagnostics(diagnostics)

        min_run_candidate_indices, tm_run_candidate_indices = pruning_kmer_index.build_candidate_sets_with_progress(accepted)
        rc_sequence_by_sequence = {record.sequence: record.rc_sequence for record in accepted}
        using_original_heterodimer_score = (
            getattr(heterodimer_score_details, "__module__", "") == "primer_generator.thermo"
            and getattr(heterodimer_score_details, "__name__", "") == "heterodimer_score_details"
        )
        candidate_indices_for_graph: dict[int, set[int]] | None = None
        if using_original_heterodimer_score and tm_run_candidate_indices is not None:
            candidate_indices_for_graph = tm_run_candidate_indices
        elif using_original_heterodimer_score:
            candidate_indices_for_graph = min_run_candidate_indices

        def tm_function(left: str, right: str) -> float:
            nonlocal num_pairs_below_min_alignment_run, num_tms_shortcircuited, num_tmnn_evaluations_performed
            if candidate_indices_for_graph is not None and using_original_heterodimer_score:
                details = heterodimer_score_details_from_partner(left, rc_sequence_by_sequence[right], thermo_runtime_config)
            else:
                details = heterodimer_score_details(left, right, thermo_runtime_config)
            if details.get("below_min_alignment_run", False):
                num_pairs_below_min_alignment_run += 1
            if details.get("tm_shortcircuited", False):
                num_tms_shortcircuited += 1
            if details.get("tm_evaluated", False):
                num_tmnn_evaluations_performed += 1
            return float(details["tm_c"])

        accepted_sequences = [record.sequence for record in accepted]
        seed_count = cycle_seed_count if cycle_count > 1 else 0
        new_count = max(0, len(accepted_sequences) - seed_count)
        total_pair_slots = seed_count * new_count + (new_count * (new_count - 1)) // 2
        if candidate_indices_for_graph is not None:
            candidate_pair_checks = _count_candidate_pairs(candidate_indices_for_graph, len(accepted_sequences), seed_count)
            min_candidate_pair_checks = _count_candidate_pairs(min_run_candidate_indices, len(accepted_sequences), seed_count)
            if tm_run_candidate_indices is not None:
                num_pairs_below_min_alignment_run += total_pair_slots - min_candidate_pair_checks
                num_tms_shortcircuited += min_candidate_pair_checks - candidate_pair_checks
            else:
                num_pairs_below_min_alignment_run += total_pair_slots - candidate_pair_checks
            print(
                "Validating accepted pool thermodynamically:"
                f" n={len(accepted_sequences)},"
                f" pair_slots={total_pair_slots},"
                f" candidate_pair_checks={candidate_pair_checks}"
            )
        else:
            candidate_pair_checks = total_pair_slots
            print(f"Validating accepted pool thermodynamically: n={len(accepted_sequences)}, pairwise_checks={total_pair_slots}")
        graph, conflicts = build_conflict_graph(
            accepted_sequences,
            float(thermo_runtime_config["pair_tm_max_c"]),
            tm_function,
            existing_validated_count=seed_count,
            candidate_indices_by_left=candidate_indices_for_graph,
            progress_callback=_make_validation_progress_logger(candidate_pair_checks),
        )
        print(f"Pruning conflict graph: conflicted_primers={len(graph)}, conflict_edges={len(conflicts)}")
        survivor_indices = greedy_maximal_independent_set(
            accepted_sequences,
            graph,
            progress_callback=_make_pruning_progress_logger(len(accepted_sequences)),
        )
        survivor_indices_in_addition_order = sorted(survivor_indices)
        final_records = [accepted[index] for index in survivor_indices_in_addition_order]
        if len(final_records) > target_size and not config["pairing"].get("enabled", False):
            trimmed_excess_count = len(final_records) - target_size
            final_records = final_records[:target_size]
            print(
                "Trimming excess validated primers to requested target size:"
                f" removed_newest={trimmed_excess_count},"
                f" final_validated={len(final_records)}/{target_size}"
            )
        else:
            trimmed_excess_count = 0
        if interrupted:
            print("Forced-target mode disabled after manual interrupt.")
            break
        if not force_target_size:
            break
        if len(final_records) >= target_size:
            print(f"Forced-target mode satisfied after pruning: final_validated={len(final_records)}/{target_size}")
            break
        if attempts >= max_attempts:
            print(
                "Forced-target mode stopped at max_attempts after pruning:"
                f" final_validated={len(final_records)}/{target_size},"
                f" attempts={attempts}/{max_attempts}"
            )
            break

        print(
            "Forced-target mode continuing with another cycle:"
            f" final_validated={len(final_records)}/{target_size},"
            f" attempts_remaining={max_attempts - attempts}"
        )
        final_sequences = {record.sequence for record in final_records}
        removed_records = [record for record in accepted if record.sequence not in final_sequences]
        pruning_kmer_index.bulk_remove_records(removed_records)
        accepted = list(final_records)
        save_checkpoint(paths["checkpoint_file"], _checkpoint_payload(config, accepted, counters, attempts, rng))

    pairing_summary = _pair_primers(final_records, config, thermo_runtime_config, rng)
    delivered_records = (
        pairing_summary["delivered_records"] if config["pairing"].get("enabled", False) else final_records
    )

    write_csv(paths["final_csv"], _rows_from_records(delivered_records))
    if config["pairing"].get("enabled", False):
        write_csv(paths["pairings_csv"], _rows_from_pairings(pairing_summary["pairings"]))
        if pairing_summary["unpaired_records"]:
            write_csv(paths["unpaired_csv"], _rows_from_records(pairing_summary["unpaired_records"]))
    write_yaml(paths["resolved_config_yaml"], {key: value for key, value in config.items() if key != "_meta"})
    elapsed_wallclock_seconds = time.monotonic() - run_start_time
    summary = {
        "requested_target_size": target_size,
        "attempted_candidates": attempts,
        "cycle_count": cycle_count,
        "force_target_size": force_target_size,
        "accepted_before_pruning": len(accepted),
        "final_validated_size": len(final_records),
        "final_delivered_size": len(delivered_records),
        "pruned_primer_count": len(accepted) - len(final_records),
        "pruning_loss": len(accepted) - len(final_records),
        "trimmed_excess_count": trimmed_excess_count,
        "acceptance_rate": (len(accepted) / attempts) if attempts else 0.0,
        "rejection_counts": dict(counters),
        "generation_diagnostics": diagnostics,
        "conflict_count": len(conflicts),
        "conflict_edge_count": len(conflicts),
        "conflicted_primer_count": len(graph),
        "precomputed_min_tm_calculation_run_length": precomputed_min_tm_run_length,
        "precomputed_min_tm_calculation_run_length_reverse_complement": precomputed_min_tm_run_length_reverse_complement,
        "precomputed_min_tm_calculation_run_length_sense": precomputed_min_tm_run_length_sense,
        "precompute_interrupted": precompute_interrupted,
        "num_pairs_below_min_alignment_run": num_pairs_below_min_alignment_run,
        "num_Tms_shortcircuited": num_tms_shortcircuited,
        "num_TmNN_evaluations_performed": num_tmnn_evaluations_performed,
        "elapsed_wallclock_seconds": elapsed_wallclock_seconds,
        "pairing": {
            "enabled": pairing_summary["enabled"],
            "completed": pairing_summary["completed"],
            "attempts": pairing_summary["attempts"],
            "accepted_pairs": pairing_summary["accepted_pairs"],
            "rejected_pairs": pairing_summary["rejected_pairs"],
            "recycled_pairs": pairing_summary["recycled_pairs"],
            "recent_rejection_rate": pairing_summary["recent_rejection_rate"],
            "sense_tm_max_c": pairing_summary["sense_tm_max_c"],
            "unpaired_count": len(pairing_summary["unpaired_records"]),
            "num_pairing_tms_shortcircuited": pairing_summary["num_pairing_tms_shortcircuited"],
            "discarded_surplus_primers": pairing_summary["discarded_surplus_primers"],
            "target_pair_count": pairing_summary["target_pair_count"],
            "target_pair_count_reached": pairing_summary["target_pair_count_reached"],
        },
        "conflicts": conflicts,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    write_json(paths["summary_json"], summary)
    _print_completion_summary(
        attempts,
        cycle_count,
        len(accepted),
        len(delivered_records),
        len(pairing_summary["pairings"]),
        len(pairing_summary["unpaired_records"]),
        trimmed_excess_count,
        num_pairs_below_min_alignment_run,
        num_tms_shortcircuited,
        num_tmnn_evaluations_performed,
        elapsed_wallclock_seconds,
        conflicts,
        paths,
    )
