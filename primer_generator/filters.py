from __future__ import annotations

from dataclasses import dataclass

from .encoding import reverse_complement
from .thermo import intrinsic_tm_with_config_c


DNA_ALPHABET = {"A", "C", "G", "T"}


@dataclass(frozen=True)
class FilterSettings:
    primer_length: int
    gc_min_fraction: float
    gc_max_fraction: float
    intrinsic_tm_min_c: float
    intrinsic_tm_max_c: float
    thermo_config: dict[str, float]
    max_homopolymer: int
    max_dinucleotide_repeats: int
    max_trinucleotide_repeats: int
    near_palindrome_min_match: int
    self_complementarity_run_threshold: int
    forbidden_motifs: tuple[str, ...]


def validate_alphabet_and_length(sequence: str, length: int) -> bool:
    return len(sequence) == length and all(base in DNA_ALPHABET for base in sequence)


def contains_forbidden_motif(sequence: str, motifs: tuple[str, ...]) -> bool:
    return any(motif and motif in sequence for motif in motifs)


def exceeds_homopolymer_limit(sequence: str, max_homopolymer: int) -> bool:
    run = 1
    previous = sequence[0]
    for base in sequence[1:]:
        if base == previous:
            run += 1
            if run > max_homopolymer:
                return True
        else:
            run = 1
            previous = base
    return False


def exceeds_repeat_limit(sequence: str, motif_len: int, max_repeats: int) -> bool:
    if motif_len <= 0 or max_repeats <= 0:
        return False
    window = motif_len * (max_repeats + 1)
    if len(sequence) < window:
        return False
    for start in range(len(sequence) - window + 1):
        motif = sequence[start : start + motif_len]
        if motif * (max_repeats + 1) == sequence[start : start + window]:
            return True
    return False


def gc_fraction(sequence: str) -> float:
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def longest_reverse_complement_match(sequence: str, min_gap: int = 0) -> int:
    rev_comp = reverse_complement(sequence)
    n = len(sequence)
    longest = 0
    for offset in range(-n + 1, n):
        run = 0
        for i in range(n):
            j = i + offset
            if j < 0 or j >= n:
                run = 0
                continue
            if abs((n - 1 - j) - i) <= min_gap:
                run = 0
                continue
            if sequence[i] == rev_comp[j]:
                run += 1
                if run > longest:
                    longest = run
            else:
                run = 0
    return longest


def is_near_palindrome(sequence: str, min_match: int) -> bool:
    if min_match <= 0:
        return False
    return longest_reverse_complement_match(sequence, min_gap=0) >= min_match


def has_self_complementarity_risk(sequence: str, threshold: int) -> bool:
    if threshold <= 0:
        return False
    return longest_reverse_complement_match(sequence, min_gap=2) >= threshold


def filter_candidate(sequence: str, settings: FilterSettings) -> str | None:
    if exceeds_homopolymer_limit(sequence, settings.max_homopolymer):
        return "homopolymer"
    if exceeds_repeat_limit(sequence, 3, settings.max_trinucleotide_repeats):
        return "trinucleotide_repeat"
    intrinsic_tm_c = intrinsic_tm_with_config_c(sequence, settings.thermo_config)
    if intrinsic_tm_c < settings.intrinsic_tm_min_c or intrinsic_tm_c > settings.intrinsic_tm_max_c:
        return "intrinsic_tm"
    if exceeds_repeat_limit(sequence, 2, settings.max_dinucleotide_repeats):
        return "dinucleotide_repeat"
    if contains_forbidden_motif(sequence, settings.forbidden_motifs):
        return "forbidden_motif"
    if has_self_complementarity_risk(sequence, settings.self_complementarity_run_threshold):
        return "self_complementarity"
    if is_near_palindrome(sequence, settings.near_palindrome_min_match):
        return "near_palindrome"
    return None
