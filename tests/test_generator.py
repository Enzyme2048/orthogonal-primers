from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from primer_generator.config import ConfigError, effective_config, load_config, parse_yaml
from primer_generator.calibration import run_calibration
from primer_generator.cli import build_parser as build_cli_parser
from primer_generator.encoding import decode_sequence, encode_reverse_complement, encode_sequence, reverse_complement
from primer_generator.filters import (
    FilterSettings,
    contains_forbidden_motif,
    exceeds_homopolymer_limit,
    exceeds_repeat_limit,
    filter_candidate,
    has_self_complementarity_risk,
    is_near_palindrome,
)
from primer_generator.generator import run_generation
from primer_generator.graph import build_conflict_graph, greedy_maximal_independent_set
from primer_generator.thermo import heterodimer_score_details, heterodimer_tm_c, sense_score_details


class ConfigTests(unittest.TestCase):
    def test_parse_yaml_mapping_and_list(self) -> None:
        parsed = parse_yaml(
            """
            root:
              name: test
              items:
                - A
                - B
            """
        )
        self.assertEqual(parsed["root"]["name"], "test")
        self.assertEqual(parsed["root"]["items"], ["A", "B"])

    def test_load_config_requires_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "config.yaml")
            path.write_text("generator:\n  primer_length: 20\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_load_config_rejects_intrinsic_tm_min_above_max(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "config.yaml")
            path.write_text(
                """
generator:
  primer_length: 20
  target_size: 10
filters:
  intrinsic_tm_min_c: 70.0
  intrinsic_tm_max_c: 60.0
                """,
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_load_config_rejects_invalid_force_target_oversample_multiplier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "config.yaml")
            path.write_text(
                """
generator:
  primer_length: 20
  target_size: 10
  force_target_cycle_oversample_multiplier: 0.5
                """,
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(path)


class EncodingTests(unittest.TestCase):
    def test_encode_decode_round_trip(self) -> None:
        sequence = "ACGTTGCA"
        encoded = encode_sequence(sequence)
        self.assertEqual(decode_sequence(encoded, len(sequence)), sequence)

    def test_reverse_complement_encoding(self) -> None:
        self.assertEqual(reverse_complement("AAGC"), "GCTT")
        self.assertEqual(encode_reverse_complement("AAGC"), encode_sequence("GCTT"))


class FilterTests(unittest.TestCase):
    def test_basic_filters(self) -> None:
        settings = FilterSettings(
            primer_length=20,
            gc_min_fraction=0.45,
            gc_max_fraction=0.55,
            intrinsic_tm_min_c=40.0,
            intrinsic_tm_max_c=120.0,
            thermo_config={
                "strand_concentration_nM": 250.0,
                "monovalent_mM": 50.0,
                "magnesium_mM": 0.0,
            },
            max_homopolymer=3,
            max_dinucleotide_repeats=2,
            max_trinucleotide_repeats=2,
            near_palindrome_min_match=16,
            self_complementarity_run_threshold=6,
            forbidden_motifs=("GAATTC",),
        )
        self.assertTrue(contains_forbidden_motif("ACGAATTCTG", settings.forbidden_motifs))
        self.assertTrue(exceeds_homopolymer_limit("AAAATCGC", 3))
        self.assertTrue(exceeds_repeat_limit("ATATATATGC", 2, 3))
        self.assertTrue(is_near_palindrome("ATGCATGCATGCATGCAT", 8))
        self.assertTrue(has_self_complementarity_risk("GCGTTAAACGCG", 4))
        self.assertEqual(filter_candidate("GCGCGATATATATATATATA", settings), "dinucleotide_repeat")

    def test_intrinsic_tm_upper_bound_filter(self) -> None:
        settings = FilterSettings(
            primer_length=20,
            gc_min_fraction=0.45,
            gc_max_fraction=0.55,
            intrinsic_tm_min_c=-100.0,
            intrinsic_tm_max_c=30.0,
            thermo_config={
                "strand_concentration_nM": 250.0,
                "monovalent_mM": 50.0,
                "magnesium_mM": 0.0,
            },
            max_homopolymer=10,
            max_dinucleotide_repeats=10,
            max_trinucleotide_repeats=10,
            near_palindrome_min_match=99,
            self_complementarity_run_threshold=99,
            forbidden_motifs=(),
        )
        self.assertEqual(filter_candidate("GCGCGCGCGCGCGCGCGCGC", settings), "intrinsic_tm")


class ThermoAndGraphTests(unittest.TestCase):
    def test_heterodimer_tm_detects_strong_complementarity(self) -> None:
        config = {
            "strand_concentration_nM": 250.0,
            "monovalent_mM": 50.0,
            "magnesium_mM": 0.0,
            "min_alignment_run": 4,
            "three_prime_window_nt": 3,
        }
        strong = heterodimer_tm_c("GCGCGCGCGCGCGCGCGCGC", "GCGCGCGCGCGCGCGCGCGC", config)
        weak = heterodimer_tm_c("AAAAAAAAAAAAAAAAAAAA", "AAAAAAAAAAAAAAAAAAAA", config)
        self.assertGreater(strong, weak)

    def test_heterodimer_tm_requires_3prime_proximal_complementarity(self) -> None:
        config = {
            "strand_concentration_nM": 250.0,
            "monovalent_mM": 50.0,
            "magnesium_mM": 0.0,
            "min_alignment_run": 4,
            "three_prime_window_nt": 3,
        }
        three_prime_anchored = heterodimer_tm_c("AAAACCCC", "GGGGTTTT", config)
        internal_only = heterodimer_tm_c("CCCCGGGG", "GGGGTTTT", config)
        self.assertGreater(three_prime_anchored, float("-inf"))
        self.assertEqual(internal_only, float("-inf"))

    def test_precomputed_min_run_skips_tm_calculation(self) -> None:
        config = {
            "strand_concentration_nM": 250.0,
            "monovalent_mM": 50.0,
            "magnesium_mM": 0.0,
            "min_alignment_run": 4,
            "three_prime_window_nt": 3,
            "min_tm_calculation_run_length_reverse_complement": 9,
        }
        with patch("primer_generator.thermo.mt.Tm_NN", side_effect=AssertionError("Tm_NN should not be called")):
            details = heterodimer_score_details("AAAACCCC", "GGGGTTTT", config)
        self.assertEqual(details["anchored_run_length"], 8)
        self.assertEqual(details["tm_c"], float("-inf"))
        self.assertTrue(details["tm_shortcircuited"])

    def test_sense_score_detects_internal_complementarity(self) -> None:
        config = {
            "strand_concentration_nM": 250.0,
            "monovalent_mM": 50.0,
            "magnesium_mM": 0.0,
            "min_alignment_run": 4,
            "three_prime_window_nt": 3,
        }
        details = sense_score_details("CCCCGGGG", "GGGGTTTT", config)
        self.assertGreater(float(details["tm_c"]), float("-inf"))
        self.assertEqual(details["anchored_run_length"], 4)

    def test_greedy_independent_set(self) -> None:
        sequences = ["AAA", "CCC", "GGG"]
        graph = {0: {1}, 1: {0, 2}, 2: {1}}
        survivors = greedy_maximal_independent_set(sequences, graph)
        self.assertEqual(survivors, [0, 2])

    def test_build_conflict_graph_skips_existing_validated_pairs(self) -> None:
        sequences = ["AAAA", "AAAT", "AATA", "TTTT"]
        checked_pairs: list[tuple[str, str]] = []

        def tm_fn(left: str, right: str) -> float:
            checked_pairs.append((left, right))
            return -100.0

        build_conflict_graph(sequences, pair_tm_max_c=0.0, tm_fn=tm_fn, existing_validated_count=2)
        self.assertEqual(
            checked_pairs,
            [("AAAA", "AATA"), ("AAAA", "TTTT"), ("AAAT", "AATA"), ("AAAT", "TTTT"), ("AATA", "TTTT")],
        )

    def test_build_conflict_graph_batches_progress_callbacks(self) -> None:
        sequence_count = 143
        sequences = [f"S{index:03d}" for index in range(sequence_count)]
        progress_calls: list[tuple[int, int, int]] = []

        build_conflict_graph(
            sequences,
            pair_tm_max_c=0.0,
            tm_fn=lambda left, right: -100.0,
            progress_callback=lambda checked, total, conflicts: progress_calls.append((checked, total, conflicts)),
        )
        self.assertEqual(progress_calls[0][0], 10_000)
        self.assertEqual(progress_calls[-1][0], sequence_count * (sequence_count - 1) // 2)
        self.assertEqual(len(progress_calls), 2)

    def test_cycle_generation_limits_returns_force_target_oversample_limit(self) -> None:
        import primer_generator.generator as gen

        cycle_accept_limit, cycle_needed, post_target_seconds = gen._cycle_generation_limits(
            current_validated_count=0,
            target_size=10,
            force_target_size=True,
            generator_config={
                "force_target_cycle_oversample_multiplier": 10.0,
                "force_target_post_target_seconds": 30.0,
            },
        )
        self.assertEqual(cycle_accept_limit, 100)
        self.assertEqual(cycle_needed, 10)
        self.assertEqual(post_target_seconds, 30.0)

    def test_cycle_generation_limits_returns_target_when_force_target_disabled(self) -> None:
        import primer_generator.generator as gen

        cycle_accept_limit, cycle_needed, post_target_seconds = gen._cycle_generation_limits(
            current_validated_count=9,
            target_size=10,
            force_target_size=False,
            generator_config={},
        )
        self.assertEqual(cycle_accept_limit, 10)
        self.assertEqual(cycle_needed, 0)
        self.assertEqual(post_target_seconds, 0.0)


class IntegrationTests(unittest.TestCase):
    def test_pruning_kmer_prefilter_skips_full_scan_when_no_min_run_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 4
  target_size: 2
  random_seed: 31
  max_attempts: 10
  checkpoint_interval: 10
composition:
  gc_min_fraction: 0.0
  gc_max_fraction: 1.0
filters:
  intrinsic_tm_min_c: -100.0
  max_homopolymer: 20
  max_dinucleotide_repeats: 20
  max_trinucleotide_repeats: 20
  near_palindrome_min_match: 99
  self_complementarity_run_threshold: 99
forbidden_motifs:
  motifs: []
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: 40.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
  three_prime_window_nt: 3
output:
  root_dir: outputs
  final_csv: final.csv
  summary_json: summary.json
  checkpoint_file: checkpoint.pkl
  resolved_config_yaml: resolved.yaml
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            import primer_generator.generator as gen

            original_sampler = gen._sample_gc_balanced_candidate
            original_cached_score = gen.heterodimer_score_details_from_partner
            sequence_iter = iter(["AAAA", "CCCC"])
            gen._sample_gc_balanced_candidate = lambda *args, **kwargs: next(sequence_iter)
            gen.heterodimer_score_details_from_partner = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("cached thermo scan should have been skipped")
            )
            try:
                run_generation(config, resume=False)
            finally:
                gen._sample_gc_balanced_candidate = original_sampler
                gen.heterodimer_score_details_from_partner = original_cached_score
            run_dir = next(path for path in (tmp_path / "outputs").iterdir() if path.is_dir())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["num_pairs_below_min_alignment_run"], 1)
            self.assertEqual(summary["num_Tms_shortcircuited"], 0)

    def test_pruning_kmer_prefilter_shortcircuits_tm_band_without_full_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 8
  target_size: 2
  random_seed: 31
  max_attempts: 10
  checkpoint_interval: 10
composition:
  gc_min_fraction: 0.0
  gc_max_fraction: 1.0
filters:
  intrinsic_tm_min_c: -100.0
  max_homopolymer: 20
  max_dinucleotide_repeats: 20
  max_trinucleotide_repeats: 20
  near_palindrome_min_match: 99
  self_complementarity_run_threshold: 99
forbidden_motifs:
  motifs: []
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: 40.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
  min_tm_calculation_run_length_reverse_complement: 6
  three_prime_window_nt: 3
output:
  root_dir: outputs
  final_csv: final.csv
  summary_json: summary.json
  checkpoint_file: checkpoint.pkl
  resolved_config_yaml: resolved.yaml
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            import primer_generator.generator as gen

            original_sampler = gen._sample_gc_balanced_candidate
            original_cached_score = gen.heterodimer_score_details_from_partner
            sequence_iter = iter(["AAAACCCC", "GGGGAAAA"])
            gen._sample_gc_balanced_candidate = lambda *args, **kwargs: next(sequence_iter)
            gen.heterodimer_score_details_from_partner = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("cached thermo scan should have been skipped")
            )
            try:
                run_generation(config, resume=False)
            finally:
                gen._sample_gc_balanced_candidate = original_sampler
                gen.heterodimer_score_details_from_partner = original_cached_score
            run_dir = next(path for path in (tmp_path / "outputs").iterdir() if path.is_dir())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["num_pairs_below_min_alignment_run"], 0)
            self.assertEqual(summary["num_Tms_shortcircuited"], 1)

    def test_precompute_gc_rule_matches_generator_count_rounding_for_odd_lengths(self) -> None:
        import primer_generator.generator as gen

        config = {
            "generator": {
                "primer_length": 9,
            },
            "composition": {
                "gc_min_fraction": 0.45,
                "gc_max_fraction": 0.55,
            },
            "filters": {
                "max_homopolymer": 20,
                "max_dinucleotide_repeats": 20,
                "max_trinucleotide_repeats": 20,
            },
        }
        self.assertTrue(gen._passes_precompute_local_filters("GCGCATATA", config))
        self.assertTrue(gen._passes_precompute_local_filters("GCGCGATAT", config))
        self.assertFalse(gen._passes_precompute_local_filters("GCGCGGATA", config))

    def test_precompute_gc_rule_allows_high_gc_subsequence_when_embeddable(self) -> None:
        import primer_generator.generator as gen

        config = {
            "generator": {
                "primer_length": 20,
            },
            "composition": {
                "gc_min_fraction": 0.45,
                "gc_max_fraction": 0.55,
            },
            "filters": {
                "max_homopolymer": 20,
                "max_dinucleotide_repeats": 20,
                "max_trinucleotide_repeats": 20,
            },
        }
        self.assertTrue(gen._passes_precompute_local_filters("ACGCACGCG", config))
        self.assertTrue(gen._passes_precompute_local_filters("CGCACGCGA", config))
        self.assertTrue(gen._passes_precompute_local_filters("GCCGCACGT", config))
        self.assertFalse(gen._passes_precompute_local_filters("GCGCGCGCGCGCG", config))

    def test_generation_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 12
  target_size: 4
  random_seed: 7
  max_attempts: 800
  checkpoint_interval: 2
composition:
  gc_min_fraction: 0.4
  gc_max_fraction: 0.6
filters:
  intrinsic_tm_min_c: 20.0
  max_homopolymer: 3
  max_dinucleotide_repeats: 3
  max_trinucleotide_repeats: 2
  near_palindrome_min_match: 10
  self_complementarity_run_threshold: 5
forbidden_motifs:
  motifs: [GAATTC]
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: 45.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
output:
  root_dir: outputs
  final_csv: final.csv
  summary_json: summary.json
  checkpoint_file: checkpoint.pkl
  resolved_config_yaml: resolved.yaml
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            first_stdout = io.StringIO()
            with redirect_stdout(first_stdout):
                run_generation(config, resume=False)
            outputs_root = tmp_path / "outputs"
            run_dirs = [path for path in outputs_root.iterdir() if path.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            self.assertTrue((run_dir / "final.csv").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertIn("Starting orthogonal primer generation", first_stdout.getvalue())
            self.assertIn("Completed orthogonal primer generation", first_stdout.getvalue())
            self.assertIn("Generation diagnostics:", first_stdout.getvalue())
            resumed_config = load_config(config_path)
            resume_stdout = io.StringIO()
            with redirect_stdout(resume_stdout):
                run_generation(resumed_config, resume=True)
            self.assertTrue((run_dir / "checkpoint.pkl").exists())
            self.assertIn("restored checkpoint", resume_stdout.getvalue())

    def test_resume_from_legacy_checkpoint_without_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 12
  target_size: 2
  random_seed: 7
  max_attempts: 20
  checkpoint_interval: 1
composition:
  gc_min_fraction: 0.4
  gc_max_fraction: 0.6
filters:
  intrinsic_tm_min_c: 20.0
  max_homopolymer: 3
  max_dinucleotide_repeats: 3
  max_trinucleotide_repeats: 2
  near_palindrome_min_match: 10
  self_complementarity_run_threshold: 5
forbidden_motifs:
  motifs: [GAATTC]
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: 45.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
output:
  root_dir: outputs
  final_csv: final.csv
  summary_json: summary.json
  checkpoint_file: checkpoint.pkl
  resolved_config_yaml: resolved.yaml
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            run_generation(config, resume=False)
            run_dir = next(path for path in (tmp_path / "outputs").iterdir() if path.is_dir())
            checkpoint_path = run_dir / "checkpoint.pkl"
            import pickle

            payload = pickle.loads(checkpoint_path.read_bytes())
            payload.pop("config_snapshot", None)
            payload["config_digest"] = "legacy-mismatch"
            checkpoint_path.write_bytes(pickle.dumps(payload))

            resumed_config = load_config(config_path)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                run_generation(resumed_config, resume=True)
            self.assertIn("legacy checkpoint", stdout.getvalue())

    def test_early_stop_writes_filter_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 20
  target_size: 20
  random_seed: 11
  max_attempts: 100
  checkpoint_interval: 10
composition:
  gc_min_fraction: 0.45
  gc_max_fraction: 0.55
filters:
  intrinsic_tm_min_c: 60.0
  max_homopolymer: 3
  max_dinucleotide_repeats: 3
  max_trinucleotide_repeats: 2
  near_palindrome_min_match: 16
  self_complementarity_run_threshold: 6
forbidden_motifs:
  motifs: [GAATTC]
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: 45.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
output:
  root_dir: outputs
  final_csv: final.csv
  summary_json: summary.json
  checkpoint_file: checkpoint.pkl
  resolved_config_yaml: resolved.yaml
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                run_generation(config, resume=False)
            run_dir = next(path for path in (tmp_path / "outputs").iterdir() if path.is_dir())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            diagnostics = summary["generation_diagnostics"]
            self.assertFalse(diagnostics["target_reached"])
            self.assertTrue(diagnostics["funnel_failures"])
            self.assertEqual(diagnostics["funnel_failures"][0]["filter"], "duplicate")
            self.assertNotIn("invalid_alphabet_or_length", diagnostics["rejection_counts"])
            self.assertNotIn("gc_fraction", diagnostics["rejection_counts"])
            self.assertIn("of_reaching_filter=", stdout.getvalue())
            self.assertIn("Generation diagnostics:", stdout.getvalue())

    def test_progress_updates_print_even_without_acceptances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 20
  target_size: 5
  random_seed: 21
  max_attempts: 5
  checkpoint_interval: 10
composition:
  gc_min_fraction: 0.45
  gc_max_fraction: 0.55
filters:
  intrinsic_tm_min_c: 200.0
  intrinsic_tm_max_c: 250.0
  max_homopolymer: 3
  max_dinucleotide_repeats: 3
  max_trinucleotide_repeats: 2
  near_palindrome_min_match: 16
  self_complementarity_run_threshold: 6
forbidden_motifs:
  motifs: [GAATTC]
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: 45.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
output:
  root_dir: outputs
  final_csv: final.csv
  summary_json: summary.json
  checkpoint_file: checkpoint.pkl
  resolved_config_yaml: resolved.yaml
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            stdout = io.StringIO()
            monotonic_values = iter([0.0, 31.0, 62.0, 93.0, 124.0, 155.0, 186.0, 217.0, 248.0, 279.0])
            with patch("primer_generator.generator.time.monotonic", side_effect=lambda: next(monotonic_values)):
                with redirect_stdout(stdout):
                    run_generation(config, resume=False)
            self.assertIn("Progress: attempts=1 (20.0%), accepted=0/5", stdout.getvalue())

    def test_force_target_size_runs_multiple_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 4
  target_size: 3
  random_seed: 3
  max_attempts: 20
  checkpoint_interval: 10
  force_target_size: true
  force_target_cycle_oversample_multiplier: 1.0
  force_target_post_target_seconds: 0.0
composition:
  gc_min_fraction: 0.0
  gc_max_fraction: 1.0
filters:
  intrinsic_tm_min_c: -100.0
  max_homopolymer: 10
  max_dinucleotide_repeats: 10
  max_trinucleotide_repeats: 10
  near_palindrome_min_match: 99
  self_complementarity_run_threshold: 99
forbidden_motifs:
  motifs: []
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: -1.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
output:
  root_dir: outputs
  final_csv: final.csv
  summary_json: summary.json
  checkpoint_file: checkpoint.pkl
  resolved_config_yaml: resolved.yaml
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            import primer_generator.generator as gen

            original_sampler = gen._sample_gc_balanced_candidate
            original_build_conflict_graph = gen.build_conflict_graph
            sequence_iter = iter(["AAAA", "AAAT", "AATA", "TTTT", "TTTA", "TTAA"])
            gen._sample_gc_balanced_candidate = lambda *args, **kwargs: next(sequence_iter)
            call_state = {"count": 0}

            def fake_build_conflict_graph(
                sequences,
                pair_tm_max_c,
                tm_fn,
                existing_validated_count=0,
                candidate_indices_by_left=None,
                progress_callback=None,
            ):
                call_state["count"] += 1
                if call_state["count"] == 1:
                    return {
                        0: {1},
                        1: {0},
                    }, [
                        {"left_index": 0, "right_index": 1, "tm_c": 1.0},
                    ]
                return {}, []

            gen.build_conflict_graph = fake_build_conflict_graph
            try:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    run_generation(config, resume=False)
            finally:
                gen._sample_gc_balanced_candidate = original_sampler
                gen.build_conflict_graph = original_build_conflict_graph
            run_dir = next(path for path in (tmp_path / "outputs").iterdir() if path.is_dir())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["force_target_size"])
            self.assertEqual(summary["final_validated_size"], 3)
            self.assertEqual(summary["trimmed_excess_count"], 0)
            self.assertEqual(summary["cycle_count"], 2)
            with (run_dir / "final.csv").open("r", encoding="utf-8", newline="") as handle:
                final_rows = list(csv.DictReader(handle))
            self.assertEqual([row["sequence"] for row in final_rows], ["AAAA", "AATA", "TTTT"])
            self.assertIn("cycle_accept_limit=3", stdout.getvalue())
            self.assertIn("oversample_multiplier=1.00", stdout.getvalue())
            self.assertIn("Forced-target mode satisfied after pruning: final_validated=3/3", stdout.getvalue())

    def test_manual_interrupt_proceeds_to_pruning_and_ignores_force_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 4
  target_size: 3
  random_seed: 5
  max_attempts: 20
  checkpoint_interval: 10
  force_target_size: true
composition:
  gc_min_fraction: 0.0
  gc_max_fraction: 1.0
filters:
  intrinsic_tm_min_c: -100.0
  max_homopolymer: 10
  max_dinucleotide_repeats: 10
  max_trinucleotide_repeats: 10
  near_palindrome_min_match: 99
  self_complementarity_run_threshold: 99
forbidden_motifs:
  motifs: []
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: 999.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
output:
  root_dir: outputs
  final_csv: final.csv
  summary_json: summary.json
  checkpoint_file: checkpoint.pkl
  resolved_config_yaml: resolved.yaml
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            import primer_generator.generator as gen

            original_sampler = gen._sample_gc_balanced_candidate
            call_state = {"count": 0}

            def interrupting_sampler(*args, **kwargs):
                call_state["count"] += 1
                if call_state["count"] == 1:
                    return "AAAA"
                raise KeyboardInterrupt()

            gen._sample_gc_balanced_candidate = interrupting_sampler
            try:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    run_generation(config, resume=False)
            finally:
                gen._sample_gc_balanced_candidate = original_sampler
            run_dir = next(path for path in (tmp_path / "outputs").iterdir() if path.is_dir())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["force_target_size"])
            self.assertEqual(summary["final_validated_size"], 1)
            self.assertEqual(summary["cycle_count"], 1)
            self.assertIn("Manual interrupt received:", stdout.getvalue())
            self.assertIn("Forced-target mode disabled after manual interrupt.", stdout.getvalue())

    def test_precompute_shortest_threshold_seq_sets_runtime_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 4
  target_size: 2
  random_seed: 31
  max_attempts: 10
  checkpoint_interval: 10
composition:
  gc_min_fraction: 0.0
  gc_max_fraction: 1.0
filters:
  intrinsic_tm_min_c: -100.0
  max_homopolymer: 10
  max_dinucleotide_repeats: 10
  max_trinucleotide_repeats: 10
  near_palindrome_min_match: 99
  self_complementarity_run_threshold: 99
forbidden_motifs:
  motifs: []
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: 5.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
  three_prime_window_nt: 3
  precompute_shortest_threshold_seq: true
output:
  root_dir: outputs
  final_csv: final.csv
  summary_json: summary.json
  checkpoint_file: checkpoint.pkl
  resolved_config_yaml: resolved.yaml
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            import primer_generator.generator as gen

            original_precompute = gen._precompute_shortest_threshold_seq_lengths
            original_score_details = gen.heterodimer_score_details
            original_sampler = gen._sample_gc_balanced_candidate
            captured_limits: list[int] = []
            sequence_iter = iter(["AAAA", "AAAT"])

            gen._precompute_shortest_threshold_seq_lengths = lambda cfg, thresholds: ({"reverse_complement": 9}, False)
            gen._sample_gc_balanced_candidate = lambda *args, **kwargs: next(sequence_iter)

            def fake_score_details(left: str, right: str, thermo: dict[str, float]) -> dict[str, float | int | str]:
                captured_limits.append(int(thermo.get("min_tm_calculation_run_length_reverse_complement", 0)))
                return {
                    "anchored_run_length": 0,
                    "anchored_run_sequence": "",
                    "tm_c": float("-inf"),
                }

            gen.heterodimer_score_details = fake_score_details
            try:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    run_generation(config, resume=False)
            finally:
                gen._precompute_shortest_threshold_seq_lengths = original_precompute
                gen.heterodimer_score_details = original_score_details
                gen._sample_gc_balanced_candidate = original_sampler
            run_dir = next(path for path in (tmp_path / "outputs").iterdir() if path.is_dir())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["precomputed_min_tm_calculation_run_length"], 9)
            self.assertEqual(summary["precomputed_min_tm_calculation_run_length_reverse_complement"], 9)
            self.assertFalse(summary["precompute_interrupted"])
            self.assertTrue(captured_limits)
            self.assertEqual(set(captured_limits), {9})
            self.assertIn("reverse_complement_min_tm_calculation_run_length=9", stdout.getvalue())

    def test_summary_reports_num_tms_shortcircuited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 8
  target_size: 2
  random_seed: 31
  max_attempts: 10
  checkpoint_interval: 10
composition:
  gc_min_fraction: 0.0
  gc_max_fraction: 1.0
filters:
  intrinsic_tm_min_c: -100.0
  max_homopolymer: 20
  max_dinucleotide_repeats: 20
  max_trinucleotide_repeats: 20
  near_palindrome_min_match: 99
  self_complementarity_run_threshold: 99
forbidden_motifs:
  motifs: []
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: 40.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
  three_prime_window_nt: 3
output:
  root_dir: outputs
  final_csv: final.csv
  summary_json: summary.json
  checkpoint_file: checkpoint.pkl
  resolved_config_yaml: resolved.yaml
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            import primer_generator.generator as gen

            original_sampler = gen._sample_gc_balanced_candidate
            original_score_details = gen.heterodimer_score_details
            sequence_iter = iter(["AAAACCCC", "GGGGTTTT"])
            gen._sample_gc_balanced_candidate = lambda *args, **kwargs: next(sequence_iter)

            def fake_score_details(left: str, right: str, thermo: dict[str, float]) -> dict[str, float | int | str]:
                return {
                    "anchored_run_length": 6,
                    "anchored_run_sequence": "AACCCC",
                    "tm_c": float("-inf"),
                    "tm_shortcircuited": True,
                }

            gen.heterodimer_score_details = fake_score_details
            try:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    run_generation(config, resume=False)
            finally:
                gen._sample_gc_balanced_candidate = original_sampler
                gen.heterodimer_score_details = original_score_details
            run_dir = next(path for path in (tmp_path / "outputs").iterdir() if path.is_dir())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["num_Tms_shortcircuited"], 1)
            self.assertIn("num_Tms_shortcircuited=1", stdout.getvalue())

    def test_summary_reports_tm_accounting_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 4
  target_size: 3
  random_seed: 31
  max_attempts: 10
  checkpoint_interval: 10
composition:
  gc_min_fraction: 0.0
  gc_max_fraction: 1.0
filters:
  intrinsic_tm_min_c: -100.0
  max_homopolymer: 20
  max_dinucleotide_repeats: 20
  max_trinucleotide_repeats: 20
  near_palindrome_min_match: 99
  self_complementarity_run_threshold: 99
forbidden_motifs:
  motifs: []
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: 40.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
  three_prime_window_nt: 3
output:
  root_dir: outputs
  final_csv: final.csv
  summary_json: summary.json
  checkpoint_file: checkpoint.pkl
  resolved_config_yaml: resolved.yaml
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            import primer_generator.generator as gen

            original_sampler = gen._sample_gc_balanced_candidate
            original_score_details = gen.heterodimer_score_details
            sequence_iter = iter(["AAAA", "CCCC", "GGGG"])
            detail_iter = iter(
                [
                    {
                        "anchored_run_length": 2,
                        "anchored_run_sequence": "AA",
                        "tm_c": float("-inf"),
                        "below_min_alignment_run": True,
                        "tm_shortcircuited": False,
                        "tm_evaluated": False,
                    },
                    {
                        "anchored_run_length": 6,
                        "anchored_run_sequence": "AACCCC",
                        "tm_c": float("-inf"),
                        "below_min_alignment_run": False,
                        "tm_shortcircuited": True,
                        "tm_evaluated": False,
                    },
                    {
                        "anchored_run_length": 9,
                        "anchored_run_sequence": "AACCCCGGG",
                        "tm_c": 10.0,
                        "below_min_alignment_run": False,
                        "tm_shortcircuited": False,
                        "tm_evaluated": True,
                    },
                ]
            )
            gen._sample_gc_balanced_candidate = lambda *args, **kwargs: next(sequence_iter)
            gen.heterodimer_score_details = lambda left, right, thermo: next(detail_iter)
            try:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    run_generation(config, resume=False)
            finally:
                gen._sample_gc_balanced_candidate = original_sampler
                gen.heterodimer_score_details = original_score_details
            run_dir = next(path for path in (tmp_path / "outputs").iterdir() if path.is_dir())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["num_pairs_below_min_alignment_run"], 1)
            self.assertEqual(summary["num_Tms_shortcircuited"], 1)
            self.assertEqual(summary["num_TmNN_evaluations_performed"], 1)
            self.assertIn("num_pairs_below_min_alignment_run=1", stdout.getvalue())
            self.assertIn("num_TmNN_evaluations_performed=1", stdout.getvalue())

    def test_optional_pairing_writes_pairings_and_unpaired_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 4
  target_size: 4
  random_seed: 7
  max_attempts: 20
  checkpoint_interval: 10
composition:
  gc_min_fraction: 0.0
  gc_max_fraction: 1.0
filters:
  intrinsic_tm_min_c: -100.0
  max_homopolymer: 20
  max_dinucleotide_repeats: 20
  max_trinucleotide_repeats: 20
  near_palindrome_min_match: 99
  self_complementarity_run_threshold: 99
forbidden_motifs:
  motifs: []
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: 999.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
  three_prime_window_nt: 3
pairing:
  enabled: true
  sense_tm_max_c: 5.0
  max_attempts_multiplier: 10
  stagnation_attempt_window: 20
output:
  root_dir: outputs
  final_csv: final.csv
  pairings_csv: pairs.csv
  unpaired_csv: leftovers.csv
  summary_json: summary.json
  checkpoint_file: checkpoint.pkl
  resolved_config_yaml: resolved.yaml
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            import primer_generator.generator as gen

            original_sampler = gen._sample_gc_balanced_candidate
            original_sense_score_details = gen.sense_score_details
            sequence_iter = iter(["AAAA", "CCCC", "GGGG", "TTTT"])
            gen._sample_gc_balanced_candidate = lambda *args, **kwargs: next(sequence_iter)
            gen.sense_score_details = lambda left, right, thermo: {
                "anchored_run_length": 0,
                "anchored_run_sequence": "",
                "tm_c": -100.0,
                "tm_shortcircuited": False,
            }
            try:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    run_generation(config, resume=False)
            finally:
                gen._sample_gc_balanced_candidate = original_sampler
                gen.sense_score_details = original_sense_score_details
            run_dir = next(path for path in (tmp_path / "outputs").iterdir() if path.is_dir())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            with (run_dir / "pairs.csv").open("r", encoding="utf-8", newline="") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertTrue(summary["pairing"]["enabled"])
            self.assertTrue(summary["pairing"]["completed"])
            self.assertEqual(summary["pairing"]["accepted_pairs"], 2)
            self.assertEqual(summary["pairing"]["unpaired_count"], 0)
            self.assertEqual(len(pair_rows), 2)
            self.assertFalse((run_dir / "leftovers.csv").exists())
            self.assertIn("Starting pairing phase:", stdout.getvalue())
            self.assertIn("Completed pairing phase:", stdout.getvalue())

    def test_pairing_uses_full_validated_pool_then_discards_surplus_after_target_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 4
  target_size: 2
  random_seed: 7
  max_attempts: 20
  checkpoint_interval: 10
  force_target_size: true
  force_target_cycle_oversample_multiplier: 2.0
  force_target_post_target_seconds: 30.0
composition:
  gc_min_fraction: 0.0
  gc_max_fraction: 1.0
filters:
  intrinsic_tm_min_c: -100.0
  max_homopolymer: 20
  max_dinucleotide_repeats: 20
  max_trinucleotide_repeats: 20
  near_palindrome_min_match: 99
  self_complementarity_run_threshold: 99
forbidden_motifs:
  motifs: []
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: 999.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
  three_prime_window_nt: 3
pairing:
  enabled: true
  sense_tm_max_c: 5.0
  max_attempts_multiplier: 10
  stagnation_attempt_window: 20
output:
  root_dir: outputs
  final_csv: final.csv
  pairings_csv: pairs.csv
  unpaired_csv: leftovers.csv
  summary_json: summary.json
  checkpoint_file: checkpoint.pkl
  resolved_config_yaml: resolved.yaml
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            import primer_generator.generator as gen

            original_sampler = gen._sample_gc_balanced_candidate
            original_sense_score_details = gen.sense_score_details
            sequence_iter = iter(["AAAA", "CCCC", "GGGG", "TTTT"])
            gen._sample_gc_balanced_candidate = lambda *args, **kwargs: next(sequence_iter)
            gen.sense_score_details = lambda left, right, thermo: {
                "anchored_run_length": 0,
                "anchored_run_sequence": "",
                "tm_c": -100.0,
                "tm_shortcircuited": False,
            }
            try:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    run_generation(config, resume=False)
            finally:
                gen._sample_gc_balanced_candidate = original_sampler
                gen.sense_score_details = original_sense_score_details
            run_dir = next(path for path in (tmp_path / "outputs").iterdir() if path.is_dir())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            with (run_dir / "final.csv").open("r", encoding="utf-8", newline="") as handle:
                final_rows = list(csv.DictReader(handle))
            with (run_dir / "pairs.csv").open("r", encoding="utf-8", newline="") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual(summary["final_validated_size"], 4)
            self.assertEqual(summary["final_delivered_size"], 2)
            self.assertEqual(summary["pairing"]["accepted_pairs"], 1)
            self.assertTrue(summary["pairing"]["target_pair_count_reached"])
            self.assertEqual(summary["pairing"]["discarded_surplus_primers"], 2)
            self.assertEqual(summary["pairing"]["unpaired_count"], 0)
            self.assertEqual(len(final_rows), 2)
            self.assertEqual(len(pair_rows), 1)
            self.assertFalse((run_dir / "leftovers.csv").exists())
            self.assertIn("discarded_surplus=2", stdout.getvalue())

    def test_pairing_uses_precomputed_minimum_run_length_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 4
  target_size: 2
  random_seed: 7
  max_attempts: 20
  checkpoint_interval: 10
composition:
  gc_min_fraction: 0.0
  gc_max_fraction: 1.0
filters:
  intrinsic_tm_min_c: -100.0
  max_homopolymer: 20
  max_dinucleotide_repeats: 20
  max_trinucleotide_repeats: 20
  near_palindrome_min_match: 99
  self_complementarity_run_threshold: 99
forbidden_motifs:
  motifs: []
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: 999.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
  three_prime_window_nt: 3
  precompute_shortest_threshold_seq: true
pairing:
  enabled: true
output:
  root_dir: outputs
  final_csv: final.csv
  pairings_csv: pairs.csv
  unpaired_csv: leftovers.csv
  summary_json: summary.json
  checkpoint_file: checkpoint.pkl
  resolved_config_yaml: resolved.yaml
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            import primer_generator.generator as gen

            original_precompute = gen._precompute_shortest_threshold_seq_lengths
            original_sampler = gen._sample_gc_balanced_candidate
            original_sense_score_details = gen.sense_score_details
            captured_limits: list[int] = []
            sequence_iter = iter(["AAAA", "CCCC"])
            gen._precompute_shortest_threshold_seq_lengths = (
                lambda cfg, thresholds: (
                    {"reverse_complement": 9, "sense": 11},
                    False,
                )
            )
            gen._sample_gc_balanced_candidate = lambda *args, **kwargs: next(sequence_iter)

            def fake_sense_score_details(left: str, right: str, thermo: dict[str, float]) -> dict[str, float | int | str]:
                captured_limits.append(int(thermo.get("min_tm_calculation_run_length_sense", 0)))
                return {
                    "anchored_run_length": 0,
                    "anchored_run_sequence": "",
                    "tm_c": -100.0,
                    "tm_shortcircuited": False,
                }

            gen.sense_score_details = fake_sense_score_details
            try:
                run_generation(config, resume=False)
            finally:
                gen._precompute_shortest_threshold_seq_lengths = original_precompute
                gen._sample_gc_balanced_candidate = original_sampler
                gen.sense_score_details = original_sense_score_details
            self.assertEqual(captured_limits, [11, 11])

    def test_precompute_stores_separate_rc_and_sense_minima(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 4
  target_size: 2
  random_seed: 7
  max_attempts: 20
  checkpoint_interval: 10
composition:
  gc_min_fraction: 0.0
  gc_max_fraction: 1.0
filters:
  intrinsic_tm_min_c: -100.0
  max_homopolymer: 20
  max_dinucleotide_repeats: 20
  max_trinucleotide_repeats: 20
  near_palindrome_min_match: 99
  self_complementarity_run_threshold: 99
forbidden_motifs:
  motifs: []
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: 40.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
  three_prime_window_nt: 3
  precompute_shortest_threshold_seq: true
pairing:
  enabled: true
  sense_tm_max_c: 35.0
output:
  root_dir: outputs
  final_csv: final.csv
  pairings_csv: pairs.csv
  unpaired_csv: leftovers.csv
  summary_json: summary.json
  checkpoint_file: checkpoint.pkl
  resolved_config_yaml: resolved.yaml
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            import primer_generator.generator as gen

            original_precompute = gen._precompute_shortest_threshold_seq_lengths
            original_sampler = gen._sample_gc_balanced_candidate
            original_sense_score_details = gen.sense_score_details
            original_score_details = gen.heterodimer_score_details
            sequence_iter = iter(["AAAA", "CCCC"])

            def fake_precompute(cfg: dict, thresholds: dict[str, float]) -> tuple[dict[str, int], bool]:
                return {"reverse_complement": 9, "sense": 11}, False

            gen._precompute_shortest_threshold_seq_lengths = fake_precompute
            gen._sample_gc_balanced_candidate = lambda *args, **kwargs: next(sequence_iter)
            gen.sense_score_details = lambda left, right, thermo: {
                "anchored_run_length": 0,
                "anchored_run_sequence": "",
                "tm_c": -100.0,
                "tm_shortcircuited": False,
            }
            gen.heterodimer_score_details = lambda left, right, thermo: {
                "anchored_run_length": 0,
                "anchored_run_sequence": "",
                "tm_c": -100.0,
                "below_min_alignment_run": True,
                "tm_shortcircuited": False,
                "tm_evaluated": False,
            }
            try:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    run_generation(config, resume=False)
            finally:
                gen._precompute_shortest_threshold_seq_lengths = original_precompute
                gen._sample_gc_balanced_candidate = original_sampler
                gen.sense_score_details = original_sense_score_details
                gen.heterodimer_score_details = original_score_details
            run_dir = next(path for path in (tmp_path / "outputs").iterdir() if path.is_dir())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["precomputed_min_tm_calculation_run_length_reverse_complement"], 9)
            self.assertEqual(summary["precomputed_min_tm_calculation_run_length_sense"], 11)
            self.assertIn("reverse_complement_min_tm_calculation_run_length=9", stdout.getvalue())
            self.assertIn("sense_min_tm_calculation_run_length=11", stdout.getvalue())

    def test_summary_reports_elapsed_wallclock_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 4
  target_size: 2
  random_seed: 7
  max_attempts: 20
  checkpoint_interval: 10
composition:
  gc_min_fraction: 0.0
  gc_max_fraction: 1.0
filters:
  intrinsic_tm_min_c: -100.0
  max_homopolymer: 20
  max_dinucleotide_repeats: 20
  max_trinucleotide_repeats: 20
  near_palindrome_min_match: 99
  self_complementarity_run_threshold: 99
forbidden_motifs:
  motifs: []
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: 999.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
  three_prime_window_nt: 3
output:
  root_dir: outputs
  final_csv: final.csv
  summary_json: summary.json
  checkpoint_file: checkpoint.pkl
  resolved_config_yaml: resolved.yaml
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            stdout = io.StringIO()
            monotonic_call_count = {"count": 0}

            def fake_monotonic() -> float:
                monotonic_call_count["count"] += 1
                return 100.0 if monotonic_call_count["count"] == 1 else 112.5

            with patch("primer_generator.generator.time.monotonic", side_effect=fake_monotonic):
                with redirect_stdout(stdout):
                    run_generation(config, resume=False)
            run_dir = next(path for path in (tmp_path / "outputs").iterdir() if path.is_dir())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["elapsed_wallclock_seconds"], 12.5)
            self.assertIn("wallclock_seconds=12.50", stdout.getvalue())

    def test_thermo_calibration_writes_summary_and_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 8
  target_size: 6
  random_seed: 19
  max_attempts: 200
  checkpoint_interval: 10
composition:
  gc_min_fraction: 0.25
  gc_max_fraction: 0.75
filters:
  intrinsic_tm_min_c: -100.0
  max_homopolymer: 4
  max_dinucleotide_repeats: 4
  max_trinucleotide_repeats: 4
  near_palindrome_min_match: 99
  self_complementarity_run_threshold: 99
forbidden_motifs:
  motifs: []
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: 10.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
output:
  root_dir: outputs
calibration:
  filtered_sample_size: 18
  max_filter_attempts: 200
  pair_sample_size: 40
  graph_trials: 2
  graph_size_cap: 30
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            import primer_generator.calibration as calibration_module

            original_score_details = calibration_module.heterodimer_score_details
            calibration_module.heterodimer_score_details = lambda left, right, thermo: {
                "anchored_run_length": sum(a == b for a, b in zip(left, right)),
                "anchored_run_sequence": "",
                "tm_c": float(sum(a == b for a, b in zip(left, right))),
            }
            try:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    summary = run_calibration(config)
            finally:
                calibration_module.heterodimer_score_details = original_score_details
            run_dir = Path(summary["run_dir"])
            summary_json = json.loads((run_dir / "calibration_summary.json").read_text(encoding="utf-8"))
            with (run_dir / "thermo_search_estimate.csv").open("r", encoding="utf-8", newline="") as handle:
                estimate_rows = list(csv.DictReader(handle))
            self.assertTrue((run_dir / "pair_3prime_run_histogram.csv").exists())
            self.assertTrue((run_dir / "calibration_resolved_config.yaml").exists())
            self.assertEqual(len(estimate_rows), 1)
            self.assertIn("predicted_search_size", summary_json["recommended"])
            self.assertIn("predicted_raw_attempts", summary_json["recommended"])
            self.assertIn("Starting thermo calibration", stdout.getvalue())
            self.assertIn("Completed thermo calibration", stdout.getvalue())
            self.assertIn("predicted_raw_attempts=", stdout.getvalue())

    def test_thermo_calibration_progress_and_cli_overrides(self) -> None:
        parser = build_cli_parser()
        args = parser.parse_args(
            [
                "--calibrate",
                "--config",
                "config.yaml",
                "--filtered-sample-size",
                "24",
                "--max-filter-attempts",
                "240",
                "--pair-sample-size",
                "60",
                "--graph-trials",
                "3",
                "--graph-size-cap",
                "40",
            ]
        )
        self.assertEqual(args.filtered_sample_size, 24)
        self.assertEqual(args.max_filter_attempts, 240)
        self.assertEqual(args.pair_sample_size, 60)
        self.assertEqual(args.graph_trials, 3)
        self.assertEqual(args.graph_size_cap, 40)
        self.assertTrue(args.calibrate)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
generator:
  primer_length: 8
  target_size: 6
  random_seed: 23
composition:
  gc_min_fraction: 0.25
  gc_max_fraction: 0.75
filters:
  intrinsic_tm_min_c: -100.0
  max_homopolymer: 4
  max_dinucleotide_repeats: 4
  max_trinucleotide_repeats: 4
  near_palindrome_min_match: 99
  self_complementarity_run_threshold: 99
forbidden_motifs:
  motifs: []
  include_reverse_complements: true
thermodynamics:
  pair_tm_max_c: 10.0
  strand_concentration_nM: 250.0
  monovalent_mM: 50.0
  magnesium_mM: 0.0
  min_alignment_run: 4
output:
  root_dir: outputs
calibration:
  filtered_sample_size: 10
  max_filter_attempts: 100
  pair_sample_size: 15
  graph_trials: 2
  graph_size_cap: 20
                """,
                encoding="utf-8",
            )
            config = load_config(config_path)
            import primer_generator.calibration as calibration_module

            original_score_details = calibration_module.heterodimer_score_details
            calibration_module.heterodimer_score_details = lambda left, right, thermo: {
                "anchored_run_length": sum(a == b for a, b in zip(left, right)),
                "anchored_run_sequence": "",
                "tm_c": float(sum(a == b for a, b in zip(left, right))),
            }
            monotonic_values = iter(range(0, 100000, 31))
            try:
                stdout = io.StringIO()
                with patch("primer_generator.calibration.time.monotonic", side_effect=lambda: next(monotonic_values)):
                    with redirect_stdout(stdout):
                        summary = run_calibration(
                            config,
                            {
                                "filtered_sample_size": 12,
                                "max_filter_attempts": 120,
                                "pair_sample_size": 20,
                                "graph_trials": 3,
                                "graph_size_cap": 24,
                            },
                        )
            finally:
                calibration_module.heterodimer_score_details = original_score_details
            run_dir = Path(summary["run_dir"])
            summary_json = json.loads((run_dir / "calibration_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_json["calibration_inputs"]["filtered_sample_size"], 12)
            self.assertEqual(summary_json["calibration_inputs"]["pair_sample_size"], 20)
            self.assertEqual(summary_json["calibration_inputs"]["graph_trials"], 3)
            self.assertEqual(summary_json["calibration_inputs"]["graph_size_cap"], 24)
            self.assertIn("Calibration progress: stage=filter_sampling", stdout.getvalue())
            self.assertIn("Calibration progress: stage=pair_sampling", stdout.getvalue())
            self.assertIn("Calibration progress: stage=thermo_model", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
