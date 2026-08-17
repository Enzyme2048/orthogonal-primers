from __future__ import annotations

from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt

from .encoding import reverse_complement


def _tm_kwargs(thermo_config: dict[str, float]) -> dict[str, float]:
    concentration = float(thermo_config["strand_concentration_nM"])
    return {
        "dnac1": concentration,
        "dnac2": concentration,
        "Na": float(thermo_config["monovalent_mM"]),
        "Mg": float(thermo_config.get("magnesium_mM", 0.0)),
    }


def intrinsic_tm_c(sequence: str) -> float:
    return intrinsic_tm_with_config_c(
        sequence,
        {
            "strand_concentration_nM": 250.0,
            "monovalent_mM": 50.0,
            "magnesium_mM": 0.0,
        },
    )


def intrinsic_tm_with_config_c(sequence: str, thermo_config: dict[str, float]) -> float:
    if len(sequence) < 2:
        return float("-inf")
    return mt.Tm_NN(sequence, **_tm_kwargs(thermo_config))


def _three_prime_window_nt(thermo_config: dict[str, float]) -> int:
    return max(1, int(thermo_config.get("three_prime_window_nt", 3)))


def _max_3prime_anchored_run_at_offset(
    sequence: str,
    partner_sequence: str,
    offset: int,
    three_prime_window_nt: int,
) -> tuple[int, int]:
    n = len(sequence)
    best_start = 0
    best_length = 0
    current_start = 0
    current_length = 0
    three_prime_start = max(0, n - three_prime_window_nt)
    for i in range(n):
        j = i + offset
        if j < 0 or j >= n:
            if (
                current_length
                and current_start + current_length - 1 >= three_prime_start
                and current_length > best_length
            ):
                best_start = current_start
                best_length = current_length
            current_length = 0
            current_start = i + 1
            continue
        if sequence[i] == partner_sequence[j]:
            if not current_length:
                current_start = i
            current_length += 1
        else:
            if (
                current_length
                and current_start + current_length - 1 >= three_prime_start
                and current_length > best_length
            ):
                best_start = current_start
                best_length = current_length
            current_length = 0
            current_start = i + 1
    if current_length and current_start + current_length - 1 >= three_prime_start and current_length > best_length:
        best_start = current_start
        best_length = current_length
    return best_start, best_length


def _max_run_at_offset(
    sequence: str,
    partner_sequence: str,
    offset: int,
) -> tuple[int, int]:
    n = len(sequence)
    best_start = 0
    best_length = 0
    current_start = 0
    current_length = 0
    for i in range(n):
        j = i + offset
        if j < 0 or j >= n:
            if current_length > best_length:
                best_start = current_start
                best_length = current_length
            current_length = 0
            continue
        if sequence[i] == partner_sequence[j]:
            if not current_length:
                current_start = i
            current_length += 1
        else:
            if current_length > best_length:
                best_start = current_start
                best_length = current_length
            current_length = 0
    if current_length > best_length:
        best_start = current_start
        best_length = current_length
    return best_start, best_length


def _best_3prime_mispriming_run_against_partner(
    sequence_a: str,
    partner_sequence: str,
    thermo_config: dict[str, float],
) -> tuple[int, int]:
    best_start = 0
    best_length = 0
    three_prime_window_nt = _three_prime_window_nt(thermo_config)
    for offset in range(-len(sequence_a) + 1, len(sequence_a)):
        run_start, run_length = _max_3prime_anchored_run_at_offset(sequence_a, partner_sequence, offset, three_prime_window_nt)
        if run_length > best_length:
            best_start = run_start
            best_length = run_length
    return best_start, best_length


def _best_run_against_partner(
    sequence_a: str,
    partner_sequence: str,
) -> tuple[int, int]:
    best_start = 0
    best_length = 0
    for offset in range(-len(sequence_a) + 1, len(sequence_a)):
        run_start, run_length = _max_run_at_offset(sequence_a, partner_sequence, offset)
        if run_length > best_length:
            best_start = run_start
            best_length = run_length
    return best_start, best_length


def best_3prime_mispriming_run(sequence_a: str, sequence_b: str, thermo_config: dict[str, float]) -> str:
    run_start, run_length = _best_3prime_mispriming_run_against_partner(
        sequence_a,
        reverse_complement(sequence_b),
        thermo_config,
    )
    return sequence_a[run_start : run_start + run_length]


def _interaction_score_details_from_partner(
    sequence_a: str,
    partner_sequence: str,
    thermo_config: dict[str, float],
    best_run_start: int,
    best_run_length: int,
    precomputed_min_run: int,
) -> dict[str, float | int | str]:
    min_run = int(thermo_config.get("min_alignment_run", 4))
    tm_calculation_min_run = max(min_run, precomputed_min_run)
    anchored_run_sequence = sequence_a[best_run_start : best_run_start + best_run_length] if best_run_length else ""
    below_min_alignment_run = best_run_length < min_run
    tm_shortcircuited = precomputed_min_run > min_run and min_run <= best_run_length < tm_calculation_min_run
    if best_run_length < tm_calculation_min_run:
        return {
            "anchored_run_length": best_run_length,
            "anchored_run_sequence": anchored_run_sequence,
            "tm_c": float("-inf"),
            "below_min_alignment_run": below_min_alignment_run,
            "tm_shortcircuited": tm_shortcircuited,
            "tm_evaluated": False,
        }
    run_partner = str(Seq(anchored_run_sequence).complement())
    return {
        "anchored_run_length": best_run_length,
        "anchored_run_sequence": anchored_run_sequence,
        "tm_c": mt.Tm_NN(anchored_run_sequence, c_seq=run_partner, **_tm_kwargs(thermo_config)),
        "below_min_alignment_run": False,
        "tm_shortcircuited": False,
        "tm_evaluated": True,
    }


def heterodimer_score_details_from_partner(
    sequence_a: str,
    partner_sequence: str,
    thermo_config: dict[str, float],
) -> dict[str, float | int | str]:
    best_run_start, best_run_length = _best_3prime_mispriming_run_against_partner(sequence_a, partner_sequence, thermo_config)
    precomputed_min_run = max(
        0,
        int(
            thermo_config.get(
                "min_tm_calculation_run_length_reverse_complement",
                thermo_config.get("min_tm_calculation_run_length", 0),
            )
        ),
    )
    return _interaction_score_details_from_partner(
        sequence_a,
        partner_sequence,
        thermo_config,
        best_run_start,
        best_run_length,
        precomputed_min_run,
    )


def heterodimer_score_details(sequence_a: str, sequence_b: str, thermo_config: dict[str, float]) -> dict[str, float | int | str]:
    partner_sequence = reverse_complement(sequence_b)
    return heterodimer_score_details_from_partner(sequence_a, partner_sequence, thermo_config)


def sense_score_details(sequence_a: str, sequence_b: str, thermo_config: dict[str, float]) -> dict[str, float | int | str]:
    best_run_start, best_run_length = _best_run_against_partner(sequence_a, sequence_b)
    precomputed_min_run = max(
        0,
        int(
            thermo_config.get(
                "min_tm_calculation_run_length_sense",
                thermo_config.get("min_tm_calculation_run_length", 0),
            )
        ),
    )
    return _interaction_score_details_from_partner(
        sequence_a,
        sequence_b,
        thermo_config,
        best_run_start,
        best_run_length,
        precomputed_min_run,
    )


def heterodimer_tm_c(sequence_a: str, sequence_b: str, thermo_config: dict[str, float]) -> float:
    return float(heterodimer_score_details(sequence_a, sequence_b, thermo_config)["tm_c"])
