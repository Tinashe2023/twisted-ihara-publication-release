import numpy as np
import pytest

from src.analysis.su2_probe_families import (
    holonomies_from_axis_angle_specification,
    seven_chord_probe_specifications,
)
from src.graphs.fundamental_holonomies import (
    deterministic_spanning_tree,
    fundamental_chords,
)
from src.graphs.random_biregular import (
    random_square_regular_parity_check,
)
from src.graphs.tanner_graphs import build_tanner_graph


def seven_chord_graph():
    generator = np.random.default_rng(20261007)

    parity_check = random_square_regular_parity_check(
        size=6,
        degree=3,
        generator=generator,
    )

    return build_tanner_graph(parity_check)


def test_every_probe_has_seven_entries() -> None:
    probes = seven_chord_probe_specifications()

    assert len(probes) == 3

    for specification in probes.values():
        assert len(specification) == 7


def test_probe_holonomies_match_chord_count() -> None:
    graph = seven_chord_graph()
    tree = deterministic_spanning_tree(graph)
    chords = fundamental_chords(graph, tree)

    specification = seven_chord_probe_specifications()[
        "mixed_noncommuting_a"
    ]

    holonomies = holonomies_from_axis_angle_specification(
        chords,
        specification,
    )

    assert len(chords) == 7
    assert set(holonomies) == set(chords)

    for matrix in holonomies.values():
        assert matrix.shape == (2, 2)


def test_incorrect_specification_length_is_rejected() -> None:
    graph = seven_chord_graph()
    tree = deterministic_spanning_tree(graph)
    chords = fundamental_chords(graph, tree)

    with pytest.raises(ValueError):
        holonomies_from_axis_angle_specification(
            chords,
            (("z", 0.1),),
        )
