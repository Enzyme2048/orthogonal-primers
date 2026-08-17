from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the configuration file is invalid."""


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def parse_yaml(text: str) -> Any:
    if not text.strip():
        raise ConfigError("Configuration file is empty.")
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML configuration: {exc}") from exc


DEFAULTS: dict[str, Any] = {
    "generator": {
        "random_seed": None,
        "max_attempts": 100000,
        "checkpoint_interval": 1000,
        "force_target_size": False,
        "force_target_cycle_oversample_multiplier": 10.0,
        "force_target_post_target_seconds": 30.0,
    },
    "composition": {
        "gc_min_fraction": 0.45,
        "gc_max_fraction": 0.55,
    },
    "filters": {
        "intrinsic_tm_min_c": 60.0,
        "intrinsic_tm_max_c": 70.0,
        "max_homopolymer": 3,
        "max_dinucleotide_repeats": 3,
        "max_trinucleotide_repeats": 2,
        "near_palindrome_min_match": 16,
        "self_complementarity_run_threshold": 6,
    },
    "forbidden_motifs": {
        "motifs": [],
        "motif_file": None,
        "include_reverse_complements": True,
    },
    "thermodynamics": {
        "pair_tm_max_c": 40.0,
        "strand_concentration_nM": 250.0,
        "monovalent_mM": 50.0,
        "magnesium_mM": 0.0,
        "temperature_unit": "C",
        "min_alignment_run": 4,
        "three_prime_window_nt": 3,
        "precompute_shortest_threshold_seq": False,
    },
    "pairing": {
        "enabled": False,
        "sense_tm_max_c": None,
        "rejection_window": 50,
        "requeue_probability_scale": 1.0,
        "max_attempts_multiplier": 50,
        "stagnation_attempt_window": 500,
    },
    "output": {
        "root_dir": "outputs",
        "final_csv": "final_primers.csv",
        "pairings_csv": "primer_pairs.csv",
        "unpaired_csv": "unpaired_primers.csv",
        "summary_json": "run_summary.json",
        "checkpoint_file": "checkpoint.pkl",
        "resolved_config_yaml": "resolved_config.yaml",
    },
}


REQUIRED_FIELDS = {
    "generator": ["primer_length", "target_size"],
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_motif_file(config_path: Path, config: dict[str, Any]) -> None:
    motif_section = config["forbidden_motifs"]
    motifs = [str(item).upper() for item in motif_section.get("motifs", [])]
    motif_file = motif_section.get("motif_file")
    if motif_file:
        motif_path = (config_path.parent / str(motif_file)).resolve()
        if not motif_path.exists():
            raise ConfigError(f"Motif file does not exist: {motif_path}")
        for line in motif_path.read_text(encoding="utf-8").splitlines():
            stripped = _strip_comment(line).strip().upper()
            if stripped:
                motifs.append(stripped)
    motif_section["motifs"] = sorted(set(motifs))


def _validate_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    for section, fields in REQUIRED_FIELDS.items():
        if section not in config or not isinstance(config[section], dict):
            raise ConfigError(f"Missing required section: {section}")
        for field in fields:
            if field not in config[section]:
                raise ConfigError(f"Missing required config value: {section}.{field}")
    if config["generator"]["primer_length"] <= 0:
        raise ConfigError("generator.primer_length must be positive.")
    if config["generator"]["target_size"] <= 0:
        raise ConfigError("generator.target_size must be positive.")
    if config["generator"]["force_target_cycle_oversample_multiplier"] < 1.0:
        raise ConfigError("generator.force_target_cycle_oversample_multiplier must be at least 1.0.")
    if config["generator"]["force_target_post_target_seconds"] < 0.0:
        raise ConfigError("generator.force_target_post_target_seconds must be non-negative.")
    gc_min = config["composition"]["gc_min_fraction"]
    gc_max = config["composition"]["gc_max_fraction"]
    if not (0 <= gc_min <= gc_max <= 1):
        raise ConfigError("composition.gc_min_fraction and gc_max_fraction must satisfy 0 <= min <= max <= 1.")
    if config["filters"]["max_homopolymer"] < 1:
        raise ConfigError("filters.max_homopolymer must be at least 1.")
    if config["filters"]["intrinsic_tm_min_c"] > config["filters"]["intrinsic_tm_max_c"]:
        raise ConfigError("filters.intrinsic_tm_min_c must be less than or equal to filters.intrinsic_tm_max_c.")
    if config["thermodynamics"]["three_prime_window_nt"] < 1:
        raise ConfigError("thermodynamics.three_prime_window_nt must be at least 1.")
    if config["pairing"]["rejection_window"] < 1:
        raise ConfigError("pairing.rejection_window must be at least 1.")
    if config["pairing"]["max_attempts_multiplier"] < 1:
        raise ConfigError("pairing.max_attempts_multiplier must be at least 1.")
    if config["pairing"]["stagnation_attempt_window"] < 1:
        raise ConfigError("pairing.stagnation_attempt_window must be at least 1.")
    _load_motif_file(config_path, config)
    return config


def dump_yaml(data: Any, indent: int = 0) -> str:
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False).rstrip()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file does not exist: {path}")
    parsed = parse_yaml(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ConfigError("Configuration root must be a mapping.")
    merged = _deep_merge(DEFAULTS, parsed)
    merged = _validate_config(merged, path.resolve())
    merged["_meta"] = {
        "config_path": str(path.resolve()),
        "config_digest": hashlib.sha256(
            json.dumps(merged, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
    }
    return merged


def effective_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in config.items() if key != "_meta"}


def normalize_effective_config(config: dict[str, Any]) -> dict[str, Any]:
    return _validate_config(_deep_merge(DEFAULTS, copy.deepcopy(config)), Path(config.get("_meta", {}).get("config_path", ".")))
