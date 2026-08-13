"""Tests for spanwise region detection on the two VSP baselines."""

import sys
import unittest
from pathlib import Path

import numpy as np

# studies/ is not an installed package, so make the repo root importable by path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from studies.vsp_planform import config  # noqa: E402
from studies.vsp_planform.degen_csv import read_degen_csv  # noqa: E402
from studies.vsp_planform.regions import detect_regions  # noqa: E402


def half_stick(name):
    """Stick of the surf_index == 0 half of one baseline."""
    return right_half(name).stick


def right_half(name):
    """The surf_index == 0 component of one baseline."""
    components = read_degen_csv(config.BASELINES[name])
    return next(comp for comp in components if comp.surf_index == 0)


class TestParser(unittest.TestCase):
    """The pieces the region detection depends on."""

    def test_two_components(self):
        for name in ("plan_l", "const_chord"):
            with self.subTest(name=name):
                components = read_degen_csv(config.BASELINES[name])
                self.assertEqual(len(components), 2)
                self.assertEqual([c.surf_index for c in components], [0, 1])
                self.assertEqual([c.flip_normal for c in components], [True, False])

    def test_plate_shapes(self):
        expected = {"plan_l": (27, 29), "const_chord": (95, 29)}
        for name, (num_secs, num_pnts) in expected.items():
            with self.subTest(name=name):
                plate = right_half(name).plate
                self.assertEqual((plate.num_secs, plate.num_pnts), (num_secs, num_pnts))
                for array in plate.camber_surface():
                    self.assertEqual(array.shape, (num_secs, num_pnts))

    def test_half_span_native(self):
        expected = {"plan_l": 694.111, "const_chord": 708.000}
        for name, half_span in expected.items():
            with self.subTest(name=name):
                stick = half_stick(name)
                self.assertAlmostEqual(stick.le[-1, 1], half_span, places=2)
                self.assertAlmostEqual(stick.le[0, 1], 0.0, places=6)


class TestDetectRegions(unittest.TestCase):
    def test_indices(self):
        # NOTE: the winglet of Plan_L starts at section 22, not 21. Section 22
        # (chord 52.00) is still reached by a region-B panel: sweep 4.868 deg and
        # dihedral 4.39 deg, identical to every other panel inboard of it. The
        # first panel that breaks those values is panel 22, so section 22 is the
        # inboard end of the winglet, exactly as section 90 is on ConstChord.
        expected = {"plan_l": (2, 22), "const_chord": (51, 90)}
        for name, indices in expected.items():
            with self.subTest(name=name):
                regions = detect_regions(half_stick(name))
                self.assertEqual(regions.as_tuple(), indices)

    def test_stations(self):
        expected = {"plan_l": (50.0, 661.706), "const_chord": (361.705, 674.946)}
        for name, (y_a_end, y_c_start) in expected.items():
            with self.subTest(name=name):
                regions = detect_regions(half_stick(name))
                self.assertAlmostEqual(regions.y_a_end, y_a_end, places=2)
                self.assertAlmostEqual(regions.y_c_start, y_c_start, places=2)

    def test_region_a_is_constant_chord(self):
        for name in ("plan_l", "const_chord"):
            with self.subTest(name=name):
                stick = half_stick(name)
                regions = detect_regions(stick)
                chord = stick.chord[regions.slice_a]
                np.testing.assert_allclose(chord, chord[0], rtol=1e-3)
                # And the very next section has actually started to taper.
                self.assertLess(stick.chord[regions.idx_a_end + 1], 0.999 * chord[0])

    def test_region_b_is_straight(self):
        """Region B holds one sweep and one dihedral out to the winglet.

        Only the outboard part is checked: on Plan_L the first four panels of
        region B are a blend out of the constant-chord bay (dihedral 3.1 -> 8.9
        -> 4.39 deg) before the straight panel settles in. That blend is why the
        winglet is found by walking inboard from the tip rather than outboard
        from region A.
        """
        for name in ("plan_l", "const_chord"):
            with self.subTest(name=name):
                stick = half_stick(name)
                regions = detect_regions(stick)
                panels = slice(regions.idx_a_end + 5, regions.idx_c_start)
                for angles in (stick.le_sweep()[panels], stick.dihedral()[panels]):
                    np.testing.assert_allclose(angles, angles[0], atol=1e-6)

    def test_region_c_departs(self):
        """Every winglet panel is bent well away from the region-B values."""
        for name in ("plan_l", "const_chord"):
            with self.subTest(name=name):
                stick = half_stick(name)
                regions = detect_regions(stick)
                sweep = stick.le_sweep()
                dihedral = stick.dihedral()
                reference_sweep = sweep[regions.idx_c_start - 1]
                reference_dihedral = dihedral[regions.idx_c_start - 1]
                for panel in range(regions.idx_c_start, sweep.size):
                    deviation = max(
                        abs(sweep[panel] - reference_sweep),
                        abs(dihedral[panel] - reference_dihedral),
                    )
                    self.assertGreater(deviation, 2.0)

    def test_slices_partition_the_span(self):
        for name in ("plan_l", "const_chord"):
            with self.subTest(name=name):
                stick = half_stick(name)
                regions = detect_regions(stick)
                indices = np.arange(stick.num_secs)
                covered = np.concatenate([indices[regions.slice_a], indices[regions.slice_b], indices[regions.slice_c]])
                np.testing.assert_array_equal(np.unique(covered), indices)

    def test_override(self):
        stick = half_stick("plan_l")
        regions = detect_regions(stick, override=(4, 20))
        self.assertEqual(regions.as_tuple(), (4, 20))
        self.assertAlmostEqual(regions.y_c_start, stick.le[20, 1])

    def test_bad_override_raises(self):
        stick = half_stick("plan_l")
        with self.assertRaises(ValueError):
            detect_regions(stick, override=(20, 4))
        with self.assertRaises(ValueError):
            detect_regions(stick, override=(2, 999))


if __name__ == "__main__":
    unittest.main()
