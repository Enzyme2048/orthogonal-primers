from __future__ import annotations

from collections import defaultdict
from typing import Callable

PROGRESS_BATCH_SIZE = 10_000


def build_conflict_graph(
    sequences: list[str],
    pair_tm_max_c: float,
    tm_fn,
    existing_validated_count: int = 0,
    candidate_indices_by_left: dict[int, set[int]] | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> tuple[dict[int, set[int]], list[dict[str, float | int | str]]]:
    graph: dict[int, set[int]] = defaultdict(set)
    conflicts: list[dict[str, float | int | str]] = []
    seed_count = max(0, min(existing_validated_count, len(sequences)))
    pairs_checked = 0
    candidate_edges: list[tuple[int, int]] = []
    if candidate_indices_by_left is None:
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                if i < seed_count and j < seed_count:
                    continue
                candidate_edges.append((i, j))
    else:
        for i in range(len(sequences)):
            for j in sorted(candidate_indices_by_left.get(i, set())):
                if j <= i:
                    continue
                if i < seed_count and j < seed_count:
                    continue
                candidate_edges.append((i, j))
    total_pairs = len(candidate_edges)
    for i, j in candidate_edges:
        pairs_checked += 1
        tm_value = tm_fn(sequences[i], sequences[j])
        if tm_value > pair_tm_max_c:
            graph[i].add(j)
            graph[j].add(i)
            conflicts.append({"left_index": i, "right_index": j, "tm_c": tm_value})
        if progress_callback is not None and (pairs_checked % PROGRESS_BATCH_SIZE == 0 or pairs_checked == total_pairs):
            progress_callback(pairs_checked, total_pairs, len(conflicts))
    return graph, conflicts


def greedy_maximal_independent_set(
    sequences: list[str],
    graph: dict[int, set[int]],
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> list[int]:
    remaining = set(range(len(sequences)))
    adjacency = {index: set(graph.get(index, set())) for index in range(len(sequences))}
    removed = 0
    total_nodes = len(sequences)
    while True:
        conflicted = [index for index in remaining if adjacency.get(index)]
        if not conflicted:
            break
        victim = max(
            conflicted,
            key=lambda idx: (len(adjacency[idx] & remaining), sequences[idx]),
        )
        remaining.remove(victim)
        removed += 1
        for neighbor in adjacency.get(victim, set()):
            adjacency[neighbor].discard(victim)
        adjacency[victim].clear()
        if progress_callback is not None and (removed % PROGRESS_BATCH_SIZE == 0 or removed == total_nodes):
            progress_callback(removed, total_nodes, len(remaining))
    return sorted(remaining, key=lambda idx: sequences[idx])
