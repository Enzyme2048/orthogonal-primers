# Orthogonal Primer Generator

A Python tool for generating pools of DNA primers that are locally well-behaved and thermodynamically orthogonal to one another's reverse-complement binding sites.

The generator is designed for applications where many primer binding sites are present in a shared template pool, and each primer should avoid strong off-target extension from its 3' end. It uses a streaming random generator, a configurable local filter funnel, final thermodynamic validation, graph pruning, and optional primer pairing.

## Status

This is a research-oriented generator, not a clinical or production assay-design package. The thermodynamic model and defaults should be reviewed for your chemistry, polymerase, assay format, and risk tolerance before using generated primers experimentally.

## Features

- Config-file-first workflow using `config.yaml`.
- GC-balanced random primer proposal.
- Cheapest-useful-first local filter funnel.
- Intrinsic nearest-neighbour Tm window filtering.
- Forbidden motif filtering, including optional reverse-complement motif bans.
- Homopolymer, tandem repeat, near-palindrome, and self-complementarity filters.
- 3'-anchored reverse-complement mispriming score for pool pruning.
- K-mer prefilter and cached reverse complements to reduce thermodynamic validation work.
- Optional precomputation of the shortest sequence length that can exceed each Tm threshold.
- Conflict-graph pruning with a deterministic greedy maximal independent set heuristic.
- Optional final primer pairing with separate sense-vs-sense Tm limits.
- Checkpointing and resumability.
- Calibration mode for estimating search difficulty before running large jobs.

## Installation

Requirements:

- Python 3.10 or newer is recommended.
- PyYAML
- Biopython

Install dependencies from the project root:

```powershell
python -m pip install -r requirements.txt
```

The dependency file currently contains:

```text
PyYAML>=6.0
biopython>=1.87
```

## Quick Start

Generate primers using the default root-level config:

```powershell
python orthogonal_primers.py
```

Use an alternate config file:

```powershell
python orthogonal_primers.py --config path\to\config.yaml
```

Resume the most recent run in the configured output directory:

```powershell
python orthogonal_primers.py --resume
```

Run calibration instead of generation:

```powershell
python orthogonal_primers.py --calibrate
```

Run calibration with temporary sampling overrides:

```powershell
python orthogonal_primers.py --calibrate --filtered-sample-size 1000 --pair-sample-size 10000
```

## Repository Layout

```text
.
|-- config.yaml                 Example run configuration
|-- pyproject.toml              Package metadata and CLI script entry
|-- orthogonal_primers.py       Main CLI entrypoint
|-- requirements.txt            Runtime dependencies
|-- primer_generator/
|   |-- calibration.py          Thermodynamic search calibration
|   |-- cli.py                  Command-line interface
|   |-- config.py               YAML loading, defaults, validation
|   |-- encoding.py             Reverse-complement and compact encoding helpers
|   |-- filters.py              Local primer filters
|   |-- generator.py            Streaming generation, pruning, pairing
|   |-- graph.py                Conflict graph and greedy pruning
|   |-- reporting.py            CSV, JSON, YAML, checkpoint I/O
|   `-- thermo.py               Biopython nearest-neighbour wrappers
`-- tests/
    `-- test_generator.py       Unit and integration tests
```

Generated outputs are written under `outputs/<timestamp>/` by default. The `outputs/` directory should usually be ignored by Git.

## How Generation Works

Generation has three main phases.

1. Candidate generation and local filtering

Candidates are sampled with GC-balanced random generation. They are then passed through this filter order:

```text
duplicate
homopolymer
trinucleotide_repeat
intrinsic_tm
dinucleotide_repeat
forbidden_motif
self_complementarity
near_palindrome
```

The filter order is intentionally performance-oriented. Expensive filters with low rejection rates are kept late, while fast filters and highly selective filters are used earlier.

2. Thermodynamic validation and pruning

After a candidate pool is accepted, the program validates pairwise interactions using a 3'-anchored mispriming score. For each primer pair, the primer is compared against the other primer's reverse complement. Only complementary runs close to the 3' end of the primer are considered detrimental for pool orthogonality, controlled by `thermodynamics.three_prime_window_nt`.

Pairs above `thermodynamics.pair_tm_max_c` become conflict edges in a graph. The graph is pruned using a deterministic greedy maximal independent set heuristic. The final validated pool may be smaller than the requested target size unless `generator.force_target_size` is enabled.

3. Optional pairing

If `pairing.enabled` is true, the validated primer pool is grouped into primer pairs. Pairing uses a sense-vs-sense Tm rule because paired primers will coexist in the same reaction. Unlike the main pool pruning rule, sense pairing scans complementarity across the full alignment range.

When pairing is enabled and pruning produces more validated primers than requested, the pairing step gets access to the full validated pool for extra diversity. Once the required number of pairs is reached, unused primers are discarded.

## Configuration

The normal interface is the root-level `config.yaml`. The current example config is intentionally small enough for demonstration. For large production-like pools, increase `generator.target_size` and `generator.max_attempts`.

### `generator`

Controls primer length, target size, randomness, checkpointing, and force-target cycling.

Important fields:

- `primer_length`: fixed primer length in nt.
- `target_size`: requested number of final primers.
- `random_seed`: seed for reproducibility. Use `null` for non-deterministic runs.
- `max_attempts`: total random candidate attempt budget.
- `checkpoint_interval`: save checkpoint every N accepted primers.
- `force_target_size`: if true, run additional generation/pruning cycles until the requested final size is reached or `max_attempts` is exhausted.
- `force_target_cycle_oversample_multiplier`: in force-target mode, each cycle may generate up to this multiple of the current shortfall.
- `force_target_post_target_seconds`: in force-target mode, continue sampling for this many seconds after the raw pool reaches the requested size, unless the cycle oversample limit is reached first.

### `composition`

Controls GC fraction of generated primers.

- `gc_min_fraction`
- `gc_max_fraction`

Candidate generation itself samples within this GC range, so these limits shape the proposal distribution rather than acting only as a late filter.

### `filters`

Controls local primer quality filters.

- `intrinsic_tm_min_c`: minimum intrinsic primer Tm.
- `intrinsic_tm_max_c`: maximum intrinsic primer Tm.
- `max_homopolymer`: maximum allowed same-base run.
- `max_dinucleotide_repeats`: maximum dinucleotide tandem repeat count.
- `max_trinucleotide_repeats`: maximum trinucleotide tandem repeat count.
- `near_palindrome_min_match`: threshold for near-palindrome rejection.
- `self_complementarity_run_threshold`: threshold for self-complementarity rejection.

### `forbidden_motifs`

Controls exact motif exclusions.

- `motifs`: inline motifs to ban.
- `motif_file`: optional path to a text file of motifs.
- `include_reverse_complements`: if true, reverse complements of banned motifs are also banned.

### `thermodynamics`

Controls thermodynamic validation.

- `pair_tm_max_c`: maximum allowed reverse-complement interaction Tm.
- `strand_concentration_nM`: strand concentration passed to Biopython.
- `monovalent_mM`: monovalent salt concentration passed to Biopython.
- `magnesium_mM`: magnesium concentration passed to Biopython.
- `temperature_unit`: currently expected to be `C`.
- `min_alignment_run`: minimum complementary run length before a pair is considered for Tm calculation.
- `three_prime_window_nt`: number of bases from the primer 3' end that define the anchored mispriming zone.
- `precompute_shortest_threshold_seq`: if true, estimate the shortest possible sequence length capable of exceeding configured Tm thresholds, then use that as a runtime shortcut.

### `pairing`

Controls optional final pairing.

- `enabled`: enable or disable final primer pairing.
- `sense_tm_max_c`: maximum allowed sense-vs-sense pair Tm. If omitted, the main pool Tm threshold is used.
- `rejection_window`: number of recent pairing attempts used to estimate rejection rate.
- `requeue_probability_scale`: scales the probability of recycling an accepted pair back into the pairing pool when rejections are high.
- `max_attempts_multiplier`: pairing attempt cap is based on this multiplier and the pool size.
- `stagnation_attempt_window`: stop pairing after this many attempts without improving the unpaired count.

### `calibration`

Optional calibration defaults used by `--calibrate`. These can also be overridden temporarily from the CLI.

- `filtered_sample_size`: number of locally valid primers to sample.
- `max_filter_attempts`: maximum candidate attempts while building the filtered calibration sample.
- `pair_sample_size`: number of primer pairs to sample for thermodynamic conflict estimation.
- `graph_trials`: number of random graph-pruning simulations used for retention estimates.
- `graph_size_cap`: maximum simulated graph size for retention estimates.
- `random_seed_offset`: offset added to `generator.random_seed` for deterministic calibration sampling.

### `output`

Controls output paths relative to each timestamped run directory.

- `root_dir`: base output directory.
- `final_csv`: final delivered primer table.
- `pairings_csv`: primer pair table, written only when pairing is enabled.
- `unpaired_csv`: unpaired primer table, written only when pairing is enabled and unpaired primers remain.
- `summary_json`: run summary and diagnostics.
- `checkpoint_file`: checkpoint used for resume.
- `resolved_config_yaml`: copy of the effective config for reproducibility.

## Outputs

Each generation run writes to:

```text
outputs/<timestamp>/
```

Typical generation artifacts:

- `final_primers.csv`: delivered primers. Columns include `sequence`, `reverse_complement`, `gc_fraction`, and `intrinsic_tm_c`.
- `primer_pairs.csv`: written when pairing is enabled. Contains paired primer sequences and sense-pair Tm scores.
- `unpaired_primers.csv`: written only when pairing is enabled and unpaired primers remain.
- `run_summary.json`: run metadata, generation diagnostics, pruning statistics, conflict information, pairing summary, output paths, and wallclock time.
- `resolved_config.yaml`: effective config used for the run.
- `checkpoint.pkl`: resume checkpoint.

The standalone generation diagnostics JSON file is not written; generation diagnostics are included inside `run_summary.json`.

## Calibration Mode

Calibration mode samples filtered primers and estimates thermodynamic search behavior under the current config.

```powershell
python orthogonal_primers.py --calibrate
```

Optional overrides:

```powershell
python orthogonal_primers.py --calibrate `
  --filtered-sample-size 1000 `
  --max-filter-attempts 100000 `
  --pair-sample-size 10000 `
  --graph-trials 8 `
  --graph-size-cap 300
```

Calibration writes to:

```text
outputs/<timestamp>_calibration/
```

Calibration artifacts:

- `calibration_summary.json`: recommended estimate and calibration inputs.
- `thermo_search_estimate.csv`: predicted search size, retained fraction, raw attempts, validation pairs, runtime estimate, and feasibility estimate.
- `pair_3prime_run_histogram.csv`: distribution of 3'-anchored complementary run lengths and observed conflict rates.
- `calibration_resolved_config.yaml`: effective config used for calibration.

Calibration is approximate. It is most useful for comparing parameter sets and choosing attempt budgets before running large jobs.

## Resume and Interrupt Behavior

Generation writes checkpoints periodically and at the end of each generation cycle.

Resume the latest run recorded in the configured output root:

```powershell
python orthogonal_primers.py --resume
```

Resume checks that the current effective config matches the checkpointed config. If they differ, resume fails early to avoid silently mixing incompatible runs.

During generation or precompute, `Ctrl+C` is handled gracefully where possible. In generation, an interrupt moves the run to validation/pruning with the primers accepted so far. In precompute, an interrupt uses the longest explored length as the runtime shortcut.

## Testing

Run the test suite from the project root:

```powershell
python -m unittest discover -s tests -v
```

The tests cover config parsing, local filters, thermodynamic scoring behavior, generation, resume, pruning, pairing, calibration, and output behavior.

## Performance Notes

For large runs, the expensive stages are usually thermodynamic validation and graph pruning rather than random candidate generation. The implementation includes several shortcuts:

- GC-balanced proposal to avoid wasting candidates outside composition bounds.
- Filter ordering based on measured reject rate and cost.
- Cached reverse complements.
- K-mer prefiltering before pairwise Tm evaluation.
- Optional shortest-threshold sequence precompute to avoid Tm calculations for short runs that cannot exceed the configured cutoff.
- Incremental validation in force-target cycles so already-validated pairs are not recalculated.

Large target sizes can still require substantial CPU time because final validation depends on candidate pair density and thermodynamic conflict rates.

## Reproducibility

For deterministic runs:

- Set `generator.random_seed`.
- Keep `config.yaml` unchanged.
- Use the emitted `resolved_config.yaml` and `run_summary.json` with the final primer table.

The final graph pruning heuristic is deterministic for a fixed accepted pool.

## License

See [LICENSE.htm](LICENSE.htm) for license terms.
