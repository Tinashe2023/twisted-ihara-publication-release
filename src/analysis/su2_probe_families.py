"""Controlled SU(2) probe families for fundamental graph holonomies."""

from __future__ import annotations

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.representations.su2_representations import (
    su2_x_rotation,
    su2_z_rotation,
)


def holonomies_from_axis_angle_specification(
    chords,
    specification: tuple[tuple[str, float], ...],
) -> dict:
    r"""
    Assign SU(2) matrices to ordered fundamental chords.

    Each specification entry has the form:

    $$
    (\text{axis},\theta),
    $$

    where axis is either ``"x"`` or ``"z"``.

    The chord ordering must be the deterministic ordering returned by
    ``fundamental_chords``.
    """
    if len(chords) != len(specification):
        raise ValueError(
            "The number of chord specifications must equal the number "
            "of fundamental chords."
        )

    holonomies = {}

    for chord, (axis, angle) in zip(chords, specification):
        if axis == "x":
            holonomies[chord] = su2_x_rotation(angle)

        elif axis == "z":
            holonomies[chord] = su2_z_rotation(angle)

        else:
            raise ValueError(
                f"Unsupported SU(2) rotation axis: {axis!r}. "
                "Use 'x' or 'z'."
            )

    return holonomies


def seven_chord_probe_specifications() -> dict[str, tuple[tuple[str, float], ...]]:
    r"""
    Return controlled seven-chord SU(2) probe specifications.

    The first probe is commuting because all rotations use the $z$ axis.
    The other probes contain both $x$- and $z$-axis rotations and are
    generically noncommuting.
    """
    return {
        "commuting_z": (
            ("z", 0.11),
            ("z", 0.23),
            ("z", 0.37),
            ("z", 0.47),
            ("z", 0.59),
            ("z", 0.71),
            ("z", 0.83),
        ),
        "mixed_noncommuting_a": (
            ("z", 0.11),
            ("x", 0.23),
            ("z", 0.37),
            ("x", 0.47),
            ("z", 0.59),
            ("x", 0.71),
            ("z", 0.83),
        ),
        "mixed_noncommuting_b": (
            ("x", 0.17),
            ("z", 0.29),
            ("x", 0.41),
            ("z", 0.53),
            ("z", 0.67),
            ("x", 0.79),
            ("z", 0.91),
        ),
    }
