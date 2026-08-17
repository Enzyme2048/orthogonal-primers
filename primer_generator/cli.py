from __future__ import annotations

import argparse
from pathlib import Path

from .calibration import run_calibration
from .config import load_config
from .generator import run_generation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Streaming greedy orthogonal primer generator")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the root YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the configured checkpoint file.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run thermo calibration instead of generation.",
    )
    parser.add_argument(
        "--filtered-sample-size",
        type=int,
        help="Override calibration.filtered_sample_size for this run.",
    )
    parser.add_argument(
        "--max-filter-attempts",
        type=int,
        help="Override calibration.max_filter_attempts for this run.",
    )
    parser.add_argument(
        "--pair-sample-size",
        type=int,
        help="Override calibration.pair_sample_size for this run.",
    )
    parser.add_argument(
        "--graph-trials",
        type=int,
        help="Override calibration.graph_trials for this run.",
    )
    parser.add_argument(
        "--graph-size-cap",
        type=int,
        help="Override calibration.graph_size_cap for this run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.calibrate and args.resume:
        parser.error("--resume cannot be used together with --calibrate.")
    config = load_config(Path(args.config))
    if args.calibrate:
        run_calibration(
            config,
            {
                "filtered_sample_size": args.filtered_sample_size,
                "max_filter_attempts": args.max_filter_attempts,
                "pair_sample_size": args.pair_sample_size,
                "graph_trials": args.graph_trials,
                "graph_size_cap": args.graph_size_cap,
            },
        )
    else:
        run_generation(config, resume=args.resume)
    return 0
