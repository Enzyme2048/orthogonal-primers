from __future__ import annotations

BASE_TO_BITS = {"A": 0b00, "C": 0b01, "G": 0b10, "T": 0b11}
BITS_TO_BASE = {value: key for key, value in BASE_TO_BITS.items()}
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def encode_sequence(sequence: str) -> int:
    encoded = 0
    for base in sequence:
        encoded = (encoded << 2) | BASE_TO_BITS[base]
    return encoded


def decode_sequence(encoded: int, length: int) -> str:
    chars = ["A"] * length
    value = encoded
    for index in range(length - 1, -1, -1):
        chars[index] = BITS_TO_BASE[value & 0b11]
        value >>= 2
    return "".join(chars)


def encode_reverse_complement(sequence: str) -> int:
    return encode_sequence(reverse_complement(sequence))
