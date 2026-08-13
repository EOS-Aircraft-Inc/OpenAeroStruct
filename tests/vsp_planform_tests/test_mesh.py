"""Tests for the VSP camber-surface meshes and the spanwise/chordwise re-lofting."""

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
from studies.vsp_planform.mesh import (  # noqa: E402
    N_CHORDWISE,
    chordwise_stations,
    full_mesh,
    half_mesh,
    plate_to_mesh,
    resample,
    resample_chordwise,
    resample_spanwise,
    spanwise_stations,
)
from studies.vsp_planform.regions import detect_regions  # noqa: E402

# Native half-spans in inches, and the section counts they come with.
HALF_SPAN_IN = {"plan_l": 694.111, "const_chord": 708.000}
NUM_SECS = {"plan_l": 27, "const_chord": 95}

# nx is the number of camber points per section. The CSV PLATE header reports
# it directly (29), unlike the OpenVSP API, which reports the surface point
# count (57) and makes OAS halve it.
NX = 29


class TestPlateToMesh(unittest.TestCase):
    def test_shapes(self):
        for name, num_secs in NUM_SECS.items():
            with self.subTest(name=name):
                for comp in read_degen_csv(config.BASELINES[name]):
                    self.assertEqual(plate_to_mesh(comp).shape, (NX, num_secs, 3))

    def test_leading_edge_first(self):
        """mesh[0] is the leading edge and mesh[-1] the trailing edge."""
        for name in HALF_SPAN_IN:
            with self.subTest(name=name):
                mesh, _ = half_mesh(name and config.BASELINES[name])
                self.assertLess(mesh[0, 0, 0], mesh[-1, 0, 0])
                self.assertTrue(np.all(mesh[0, :, 0] < mesh[-1, :, 0]))

    def test_half_runs_root_to_tip(self):
        for name, half_span in HALF_SPAN_IN.items():
            with self.subTest(name=name):
                mesh, _ = half_mesh(config.BASELINES[name])
                y = mesh[0, :, 1]
                self.assertTrue(np.all(np.diff(y) > 0))
                self.assertAlmostEqual(y[0], 0.0, places=6)
                self.assertAlmostEqual(y[-1], half_span * config.SCALE, places=3)

    def test_half_span_si(self):
        for name, half_span in HALF_SPAN_IN.items():
            with self.subTest(name=name):
                mesh, stick = half_mesh(config.BASELINES[name])
                self.assertAlmostEqual(stick.le[-1, 1], half_span, places=2)
                self.assertAlmostEqual(mesh[0, -1, 1], half_span * 0.0254, places=4)

    def test_half_mesh_scale(self):
        """The scale factor is the only difference from the native mesh."""
        components = read_degen_csv(config.BASELINES["plan_l"])
        mesh, _ = half_mesh(components, scale=1.0)
        scaled, _ = half_mesh(components)
        np.testing.assert_allclose(scaled, mesh * config.SCALE, rtol=1e-12)

    def test_full_mesh(self):
        for name, num_secs in NUM_SECS.items():
            with self.subTest(name=name):
                mesh = full_mesh(config.BASELINES[name])
                # The shared root section appears once: utils.py:536.
                self.assertEqual(mesh.shape, (NX, 2 * num_secs - 1, 3))
                y = mesh[0, :, 1]
                self.assertTrue(np.all(np.diff(y) > 0))
                self.assertAlmostEqual(y[0], -y[-1], places=9)
                self.assertAlmostEqual(y[-1], HALF_SPAN_IN[name] * config.SCALE, places=3)

    def test_full_mesh_is_symmetric(self):
        """The two halves mirror each other in y about the root."""
        mesh = full_mesh(config.BASELINES["plan_l"])
        right = mesh[:, NUM_SECS["plan_l"] - 1 :, :]
        left = mesh[:, : NUM_SECS["plan_l"], :][:, ::-1, :]
        np.testing.assert_allclose(right[:, :, [0, 2]], left[:, :, [0, 2]], atol=1e-9)
        np.testing.assert_allclose(right[:, :, 1], -left[:, :, 1], atol=1e-9)


class TestSpanwiseStations(unittest.TestCase):
    def test_distribution(self):
        for name in HALF_SPAN_IN:
            with self.subTest(name=name):
                mesh, stick = half_mesh(config.BASELINES[name])
                regions = detect_regions(stick)
                y_native = mesh[0, :, 1]
                y_c_start = regions.y_c_start * config.SCALE

                y = spanwise_stations(y_native, config.N_SPANWISE_HALF, y_c_start, config.WINGLET_STATION_FRACTION)

                self.assertEqual(y.size, config.N_SPANWISE_HALF)
                self.assertTrue(np.all(np.diff(y) > 0))
                self.assertAlmostEqual(y[0], y_native[0], places=9)
                self.assertAlmostEqual(y[-1], y_native[-1], places=9)

                # The winglet start is a node, and it gets its share of stations.
                self.assertTrue(np.any(np.isclose(y, y_c_start)))
                n_winglet = int(np.sum(y > y_c_start + 1e-9))
                self.assertEqual(n_winglet, round(config.WINGLET_STATION_FRACTION * config.N_SPANWISE_HALF))

                # Cosine clustering: panels shrink toward the tip inboard of the winglet.
                inboard = y[y <= y_c_start + 1e-9]
                spacing = np.diff(inboard)
                self.assertTrue(np.all(np.diff(spacing) < 0))

    def test_rejects_winglet_outside_span(self):
        y_native = np.linspace(0.0, 10.0, 20)
        with self.assertRaises(ValueError):
            spanwise_stations(y_native, 20, 11.0, 0.2)


class TestResampleSpanwise(unittest.TestCase):
    def _resample(self, name, n_total=None):
        mesh, stick = half_mesh(config.BASELINES[name])
        regions = detect_regions(stick)
        y = spanwise_stations(
            mesh[0, :, 1],
            n_total or config.N_SPANWISE_HALF,
            regions.y_c_start * config.SCALE,
            config.WINGLET_STATION_FRACTION,
        )
        return mesh, y, resample_spanwise(mesh, y)

    def test_shape_and_ends(self):
        for name in HALF_SPAN_IN:
            with self.subTest(name=name):
                mesh, y, (new_mesh, _) = self._resample(name)
                self.assertEqual(new_mesh.shape, (NX, config.N_SPANWISE_HALF, 3))
                # Root and tip sections are reproduced exactly.
                np.testing.assert_allclose(new_mesh[:, 0, :], mesh[:, 0, :], atol=1e-9)
                np.testing.assert_allclose(new_mesh[:, -1, :], mesh[:, -1, :], atol=1e-9)
                # The requested stations are hit to within the arc-length remap.
                np.testing.assert_allclose(new_mesh[0, :, 1], y, atol=1e-6)

    def test_residual_is_returned_and_small(self):
        for name in HALF_SPAN_IN:
            with self.subTest(name=name):
                _, _, (_, residual) = self._resample(name)
                self.assertEqual(set(residual), {"max", "rms", "max_relative"})
                for value in residual.values():
                    self.assertGreater(value, 0.0)
                # Errors are millimetres on an ~18 m half-span.
                self.assertLess(residual["max"], 0.05)
                self.assertLess(residual["rms"], 0.01)
                self.assertLess(residual["max_relative"], 2e-3)

    def test_residual_shrinks_with_more_stations(self):
        for name in HALF_SPAN_IN:
            with self.subTest(name=name):
                _, _, (_, coarse) = self._resample(name, n_total=15)
                _, _, (_, fine) = self._resample(name, n_total=61)
                self.assertLess(fine["rms"], coarse["rms"])

    def test_rejects_unordered_mesh(self):
        mesh, _ = half_mesh(config.BASELINES["plan_l"])
        with self.assertRaises(ValueError):
            resample_spanwise(mesh[:, ::-1, :], mesh[0, :, 1])


class TestChordwiseStations(unittest.TestCase):
    def test_endpoints_and_monotonicity(self):
        for nx in (2, 5, 7, 9, 13):
            with self.subTest(nx=nx):
                for clustering in ("cosine", "uniform"):
                    t = chordwise_stations(nx, clustering)
                    self.assertEqual(t.size, nx)
                    self.assertTrue(np.all(np.diff(t) > 0))
                    self.assertEqual(t[0], 0.0)
                    self.assertEqual(t[-1], 1.0)

    def test_cosine_clusters_at_both_edges(self):
        spacing = np.diff(chordwise_stations(13))
        # Symmetric about mid-chord, smallest at both ends, largest in the middle.
        np.testing.assert_allclose(spacing, spacing[::-1], atol=1e-12)
        self.assertEqual(int(np.argmin(spacing)), 0)
        self.assertLess(spacing[0], np.diff(chordwise_stations(13, "uniform"))[0])
        self.assertTrue(np.all(np.diff(spacing[: spacing.size // 2]) > 0))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            chordwise_stations(1)
        with self.assertRaises(ValueError):
            chordwise_stations(5, "chebyshev")


class TestResampleChordwise(unittest.TestCase):
    def test_shape_and_edges(self):
        """The leading and trailing edges survive untouched."""
        for name in HALF_SPAN_IN:
            with self.subTest(name=name):
                mesh, _ = half_mesh(config.BASELINES[name])
                new_mesh, _ = resample_chordwise(mesh, N_CHORDWISE)
                self.assertEqual(new_mesh.shape, (N_CHORDWISE, mesh.shape[1], 3))
                np.testing.assert_allclose(new_mesh[0], mesh[0], atol=1e-12)
                np.testing.assert_allclose(new_mesh[-1], mesh[-1], atol=1e-12)
                # mesh[0] is still the leading edge.
                self.assertTrue(np.all(new_mesh[0, :, 0] < new_mesh[-1, :, 0]))

    def test_no_overshoot(self):
        """PCHIP keeps every new node inside the original chordwise x-range."""
        for name in HALF_SPAN_IN:
            with self.subTest(name=name):
                mesh, _ = half_mesh(config.BASELINES[name])
                new_mesh, _ = resample_chordwise(mesh, N_CHORDWISE)
                for axis in range(3):
                    self.assertTrue(np.all(new_mesh[:, :, axis] >= mesh[:, :, axis].min(axis=0) - 1e-12))
                    self.assertTrue(np.all(new_mesh[:, :, axis] <= mesh[:, :, axis].max(axis=0) + 1e-12))

    def test_residual_decreases_with_nx(self):
        for name in HALF_SPAN_IN:
            with self.subTest(name=name):
                mesh, _ = half_mesh(config.BASELINES[name])
                previous = None
                for nx in (5, 7, 9, 13):
                    _, residual = resample_chordwise(mesh, nx)
                    self.assertEqual(set(residual), {"max", "rms", "max_relative", "rms_relative"})
                    if previous is not None:
                        self.assertLess(residual["rms"], previous)
                    previous = residual["rms"]

    def test_default_nx_error_is_small(self):
        """N_CHORDWISE keeps the camber surface within 0.3% of local chord."""
        for name in HALF_SPAN_IN:
            with self.subTest(name=name):
                mesh, _ = half_mesh(config.BASELINES[name])
                _, residual = resample_chordwise(mesh)
                self.assertLess(residual["max_relative"], 3.0e-3)
                self.assertLess(residual["rms_relative"], 1.0e-3)

    def test_refining_is_rejected(self):
        mesh, _ = half_mesh(config.BASELINES["plan_l"])
        with self.assertRaises(ValueError):
            resample_chordwise(mesh, mesh.shape[0] + 2)


class TestResample(unittest.TestCase):
    def test_combined(self):
        for name in HALF_SPAN_IN:
            with self.subTest(name=name):
                mesh, stick = half_mesh(config.BASELINES[name])
                regions = detect_regions(stick)
                y = spanwise_stations(mesh[0, :, 1], config.N_SPANWISE_HALF, regions.y_c_start * config.SCALE)
                new_mesh, residual = resample(mesh, y, N_CHORDWISE)

                self.assertEqual(new_mesh.shape, (N_CHORDWISE, config.N_SPANWISE_HALF, 3))
                np.testing.assert_allclose(new_mesh[0, :, 1], y, atol=1e-9)

                # The two contributions are reported separately, not merged.
                self.assertEqual(set(residual), {"spanwise", "chordwise"})
                self.assertLess(residual["spanwise"]["max_relative"], 2.0e-3)
                self.assertLess(residual["chordwise"]["max_relative"], 6.0e-3)

                # Same answer as applying the two steps by hand.
                spanwise_mesh, _ = resample_spanwise(mesh, y)
                expected, _ = resample_chordwise(spanwise_mesh, N_CHORDWISE)
                np.testing.assert_allclose(new_mesh, expected, atol=1e-12)

    def test_panel_count_reduction(self):
        """The point of the exercise: far fewer VLM panels than the native plate."""
        mesh, _ = half_mesh(config.BASELINES["const_chord"])
        native = (mesh.shape[0] - 1) * (mesh.shape[1] - 1)
        reduced = (N_CHORDWISE - 1) * (config.N_SPANWISE_HALF - 1)
        self.assertLess(reduced, native / 3)


if __name__ == "__main__":
    unittest.main()
