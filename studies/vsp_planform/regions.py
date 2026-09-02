"""Spanwise region detection for the VSP planform baselines.

Both baselines are built from the same three-region recipe:

* region A -- an inboard bay of constant chord,
* region B -- a straight tapered panel with constant leading-edge sweep and
  constant dihedral,
* region C -- a winglet, where sweep and dihedral both ramp up sharply.

The regions are what the optimizer parameterizes, so they have to be recovered
from the geometry rather than hardcoded per baseline. Everything here works on
the DegenStick (one entry per spanwise section), so the indices returned are
native section indices that can be used directly against ``stick.chord``,
``stick.le`` and friends.

Index convention
----------------
``idx_a_end`` is the *last* section of region A and ``idx_c_start`` is the
*first* section of region C, so the two interior sections are shared: region B
spans ``idx_a_end .. idx_c_start`` inclusive. Panel ``i`` (from
``stick.dihedral()`` / ``stick.le_sweep()``) joins sections ``i`` and ``i+1``,
so region C owns panels ``idx_c_start`` onwards.
"""

from dataclasses import dataclass

import numpy as np

# A chord is "still the root chord" while it is within this relative tolerance.
# The VSP loft wobbles by ~6e-5 through the constant-chord bay of the
# ConstChord baseline, and the first real taper step is ~1.6e-2, so anything in
# between separates them cleanly.
CHORD_TOL = 1.0e-3

# Degrees of dihedral or leading-edge sweep departure that mark the winglet.
# Region B holds both angles constant to ~1e-12 deg, and the first winglet panel
# jumps by at least 5.7 deg in dihedral and 6.2 deg in sweep on both baselines.
ANGLE_TOL = 2.0


@dataclass
class Regions:
    """Section indices and spanwise stations of the three planform regions."""

    idx_a_end: int
    idx_c_start: int
    y_a_end: float
    y_c_start: float

    @property
    def slice_a(self):
        """Sections of the constant-chord region, including its outboard end."""
        return slice(0, self.idx_a_end + 1)

    @property
    def slice_b(self):
        """Sections of the tapered region, including both shared ends."""
        return slice(self.idx_a_end, self.idx_c_start + 1)

    @property
    def slice_c(self):
        """Sections of the winglet, including its inboard end."""
        return slice(self.idx_c_start, None)

    def as_tuple(self):
        """The two index breakpoints, in the form accepted as an override."""
        return (self.idx_a_end, self.idx_c_start)


def _constant_chord_end(chord, tol):
    """Last index over which the chord is still the root chord to within ``tol``."""
    dev = np.abs(chord / chord[0] - 1.0)
    departed = np.flatnonzero(dev > tol)
    if departed.size == 0:
        # A wing with no taper at all: region B is empty, region A is everything.
        return chord.size - 1
    return int(departed[0]) - 1


def _winglet_start(dihedral, sweep, idx_a_end, tol):
    """First section index of the trailing run of panels that break region B.

    The reference dihedral and sweep are the medians over everything outboard of
    region A. Region B is by far the longest run on both baselines, so the median
    picks it out without knowing where it ends. We then walk *inboard* from the
    tip for as long as the panels keep deviating, which ignores the inboard
    blend panels near the wing root fairing -- those also deviate from the
    region-B values, but they are not part of the trailing run.
    """
    panels = np.arange(idx_a_end, dihedral.size)
    if panels.size == 0:
        raise ValueError("no panels outboard of the constant-chord region")

    dihedral_ref = np.median(dihedral[panels])
    sweep_ref = np.median(sweep[panels])

    deviation = np.maximum(
        np.abs(dihedral[panels] - dihedral_ref),
        np.abs(sweep[panels] - sweep_ref),
    )

    if deviation[-1] <= tol:
        raise ValueError("no winglet found: the outboard-most panel matches the tapered region")

    idx = panels[-1]
    while idx > idx_a_end and deviation[idx - idx_a_end - 1] > tol:
        idx -= 1
    return int(idx)


def detect_regions(stick, override=None):
    """Locate the constant-chord/taper/winglet breakpoints of one half-wing.

    Parameters
    ----------
    stick : DegenStick
        The stick of the ``surf_index == 0`` half, sections ordered root to tip.
    override : tuple or None
        ``(idx_a_end, idx_c_start)`` to bypass detection. The spanwise stations
        are still read from the geometry.

    Returns
    -------
    Regions
    """
    y = stick.le[:, 1]

    if override is not None:
        idx_a_end, idx_c_start = (int(v) for v in override)
    else:
        idx_a_end = _constant_chord_end(np.asarray(stick.chord, dtype=float), CHORD_TOL)
        idx_c_start = _winglet_start(stick.dihedral(), stick.le_sweep(), idx_a_end, ANGLE_TOL)

    if not 0 <= idx_a_end < idx_c_start < stick.num_secs:
        raise ValueError(f"nonsensical region breakpoints ({idx_a_end}, {idx_c_start}) for {stick.num_secs} sections")

    return Regions(
        idx_a_end=idx_a_end,
        idx_c_start=idx_c_start,
        y_a_end=float(y[idx_a_end]),
        y_c_start=float(y[idx_c_start]),
    )
