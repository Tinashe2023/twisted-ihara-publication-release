"""Search random regular Tanner graphs for ordinary determinant collisions."""

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import numpy as np

from src.analysis.graph_descriptors import graph_descriptor_record
from src.graphs.parity_check_families import (
    bipartition_preserving_isomorphic,
)
from src.graphs.random_biregular import (
    random_square_regular_parity_check,
)
from src.graphs.tanner_graphs import build_tanner_graph
from src.ihara.nonbacktracking import build_nonbacktracking_matrix
from src.ihara.numerical_characteristic import (
    determinant_polynomial_coefficients,
    trim_small_coefficients,
)


SEED = 20260924
TARGET_REPRESENTATIVES = 80
ROUNDING_DECIMALS = 7


def ordinary_fingerprint(graph: nx.Graph) -> np.ndarray:
    """Return cleaned coefficients of det(I-uB)."""
    matrix = build_nonbacktracking_matrix(graph)

    return trim_small_coefficients(
        determinant_polynomial_coefficients(matrix),
        tolerance=1e-8,
    )


def fingerprint_key(coefficients: np.ndarray) -> tuple[float, ...]:
    """
    Construct a hashable rounded key for numerical collision screening.

    Candidate collisions must later be verified exactly.
    """
    real_part = tuple(
        np.round(coefficients.real, ROUNDING_DECIMALS)
    )

    imaginary_part = tuple(
        np.round(coefficients.imag, ROUNDING_DECIMALS)
    )

    return real_part + imaginary_part


generator = np.random.default_rng(SEED)

representatives: list[dict] = []
representatives_by_hash: dict[str, list[dict]] = {}

attempt_count = 0
maximum_attempts = 20_000

while (
    len(representatives) < TARGET_REPRESENTATIVES
    and attempt_count < maximum_attempts
):
    attempt_count += 1

    parity_check = random_square_regular_parity_check(
        size=6,
        degree=3,
        generator=generator,
    )

    graph = build_tanner_graph(parity_check)

    if not nx.is_connected(graph):
        continue

    graph_hash = nx.weisfeiler_lehman_graph_hash(
        graph,
        node_attr="bipartite",
    )

    candidates = representatives_by_hash.setdefault(
        graph_hash,
        [],
    )

    if any(
        bipartition_preserving_isomorphic(
            graph,
            candidate["graph"],
        )
        for candidate in candidates
    ):
        continue

    fingerprint = ordinary_fingerprint(graph)

    record = {
        "index": len(representatives) + 1,
        "parity_check_matrix": parity_check.tolist(),
        "graph": graph,
        "ordinary_fingerprint": fingerprint,
        "descriptors": graph_descriptor_record(graph),
    }

    candidates.append(record)
    representatives.append(record)

print("Random (3,3)-regular Tanner collision search")
print()
print(f"Random seed: {SEED}")
print(f"Attempts: {attempt_count}")
print(f"Nonisomorphic connected representatives: {len(representatives)}")
print()

collision_buckets: dict[tuple[float, ...], list[dict]] = {}

for representative in representatives:
    key = fingerprint_key(
        representative["ordinary_fingerprint"]
    )

    collision_buckets.setdefault(key, []).append(representative)

candidate_collisions = [
    bucket
    for bucket in collision_buckets.values()
    if len(bucket) >= 2
]

print(
    "Numerical ordinary-fingerprint collision buckets: "
    f"{len(candidate_collisions)}"
)
print()

serializable_records = []

for representative in representatives:
    serializable_records.append(
        {
            "index": representative["index"],
            "parity_check_matrix": representative[
                "parity_check_matrix"
            ],
            "descriptors": representative["descriptors"],
        }
    )

for bucket_index, bucket in enumerate(
    candidate_collisions,
    start=1,
):
    indices = [representative["index"] for representative in bucket]

    print(f"Candidate collision bucket {bucket_index}: {indices}")

output_path = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "random_6x6_regular_tanner_search.json"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as output_file:
    json.dump(
        {
            "seed": SEED,
            "attempt_count": attempt_count,
            "representative_count": len(representatives),
            "candidate_collision_buckets": [
                [
                    representative["index"]
                    for representative in bucket
                ]
                for bucket in candidate_collisions
            ],
            "representatives": serializable_records,
        },
        output_file,
        indent=2,
    )

print()
print(f"Saved search summary to: {output_path}")
