"""Tests for the standalone AeroSandbox/NeuralFoil airfoil DOE.

These are sanity checks that the model is wired up correctly, not a validation
study, so the tolerances against published NACA data are deliberately loose.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# studies/ is not an installed package, so make it importable by path.
STUDY_DIR = Path(__file__).resolve().parents[2] / "studies" / "vsp_planform"
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import airfoil_doe  # noqa: E402

requires_asb = unittest.skipUnless(
    airfoil_doe.HAS_AEROSANDBOX,
    "aerosandbox is not installed; run `pip install aerosandbox`",
)


class TestNaca4Grid(unittest.TestCase):
    """The grid generator is pure Python and needs no aerosandbox."""

    def test_designation(self):
        self.assertEqual(airfoil_doe.naca4_name(2, 4, 12), "naca2412")
        self.assertEqual(airfoil_doe.naca4_name(4, 4, 12), "naca4412")
        self.assertEqual(airfoil_doe.naca4_name(2, 3, 5), "naca2305")
        self.assertEqual(airfoil_doe.naca4_name(6, 3, 18), "naca6318")

    def test_symmetric_ignores_camber_position(self):
        # An uncambered section is 00xx no matter what camber position is asked for.
        self.assertEqual(airfoil_doe.naca4_name(0, 4, 12), "naca0012")
        self.assertEqual(airfoil_doe.naca4_name(0, 6, 12), "naca0012")

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            airfoil_doe.naca4_name(10, 4, 12)
        with self.assertRaises(ValueError):
            airfoil_doe.naca4_name(2, 12, 12)
        with self.assertRaises(ValueError):
            airfoil_doe.naca4_name(2, 4, 0)

    def test_grid_contents_and_size(self):
        grid = airfoil_doe.naca4_grid([2, 4], [4], [10, 12])
        self.assertEqual(grid, ["naca2410", "naca4410", "naca2412", "naca4412"])

    def test_grid_deduplicates_symmetric_sections(self):
        grid = airfoil_doe.naca4_grid([0, 2], [2, 4, 6], [12, 18])
        # 2 thicknesses x (1 symmetric + 3 cambered)
        self.assertEqual(len(grid), 8)
        self.assertEqual(len(set(grid)), len(grid))
        self.assertEqual(grid.count("naca0012"), 1)

    def test_parse_roundtrip(self):
        for name in airfoil_doe.naca4_grid([0, 3], [4], [10, 18]):
            camber, camber_pos, thickness = airfoil_doe.parse_naca4(name)
            self.assertEqual(airfoil_doe.naca4_name(camber, camber_pos, thickness), name)

    def test_parse_rejects_junk(self):
        with self.assertRaises(ValueError):
            airfoil_doe.parse_naca4("naca63012")
        with self.assertRaises(ValueError):
            airfoil_doe.parse_naca4("e423")

    def test_default_thickness_range_covers_study_wings(self):
        # The real wings' t/c runs 0.100 to 0.178.
        self.assertLessEqual(min(airfoil_doe.DEFAULT_THICKNESSES), 10)
        self.assertGreaterEqual(max(airfoil_doe.DEFAULT_THICKNESSES), 18)


@requires_asb
class TestRunDoe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.airfoils = ["naca0012", "naca2412", "naca4415"]
        cls.alphas = np.arange(-4.0, 12.01, 1.0)
        cls.reynolds = [2.0e6, 1.0e7]
        cls.df = airfoil_doe.run_doe(cls.airfoils, cls.alphas, cls.reynolds, mach=airfoil_doe.MACH_INF)

    def test_columns(self):
        self.assertEqual(list(self.df.columns), list(airfoil_doe.DOE_COLUMNS))
        for col in ("CL", "CD", "CM", "camber", "camber_pos", "thickness"):
            self.assertIn(col, self.df.columns)

    def test_row_count(self):
        expected = len(self.airfoils) * len(self.alphas) * len(self.reynolds)
        self.assertEqual(len(self.df), expected)
        # One row per unique (airfoil, alpha, Re) triple.
        self.assertEqual(len(self.df.drop_duplicates(["airfoil", "alpha", "Re"])), expected)

    def test_parametric_decomposition(self):
        row = self.df[self.df["airfoil"] == "naca4415"].iloc[0]
        self.assertEqual(row["camber"], 4)
        self.assertEqual(row["camber_pos"], 4)
        self.assertEqual(row["thickness"], 15)
        self.assertAlmostEqual(row["t_over_c"], 0.15)

    def test_results_are_finite_and_sane(self):
        for col in ("CL", "CD", "CM"):
            self.assertTrue(np.all(np.isfinite(self.df[col])), f"{col} has non-finite values")
        self.assertTrue(np.all(self.df["CD"] > 0.0))
        self.assertTrue(np.all(self.df["CD"] < 1.0))

    def test_pivotable(self):
        pivot = self.df.pivot_table(index=["thickness", "alpha"], columns="Re", values="CD")
        self.assertEqual(len(pivot.columns), len(self.reynolds))


@requires_asb
class TestSummarize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.airfoils = ["naca0012", "naca2412"]
        cls.reynolds = [5.0e6]
        cls.df = airfoil_doe.run_doe(cls.airfoils, np.arange(-8.0, 18.01, 0.5), cls.reynolds)
        cls.summary = airfoil_doe.summarize(cls.df)

    def test_shape_and_columns(self):
        self.assertEqual(list(self.summary.columns), list(airfoil_doe.SUMMARY_COLUMNS))
        self.assertEqual(len(self.summary), len(self.airfoils) * len(self.reynolds))

    def test_values_consistent_with_raw_table(self):
        for name in self.airfoils:
            row = self.summary[self.summary["airfoil"] == name].iloc[0]
            raw = self.df[self.df["airfoil"] == name]
            self.assertAlmostEqual(row["CD_min"], raw["CD"].min(), places=6)
            self.assertLessEqual(row["CL_max"], raw["CL"].max() + 1e-9)
            self.assertGreater(row["LD_max"], 20.0)
            self.assertTrue(np.isfinite(row["CM_at_zero_lift"]))

    def test_cambered_section_has_higher_clmax_than_symmetric(self):
        clmax = self.summary.set_index("airfoil")["CL_max"]
        self.assertGreater(clmax["naca2412"], clmax["naca0012"])


@requires_asb
class TestAgainstPublishedData(unittest.TestCase):
    """Spot-check NACA 0012 against textbook thin-airfoil expectations.

    Loose tolerances on purpose: this confirms the NeuralFoil call is wired up
    with the right units and sign conventions, nothing more.
    """

    @classmethod
    def setUpClass(cls):
        # Low Mach so there is no Prandtl-Glauert amplification to account for.
        cls.df = airfoil_doe.run_doe(["naca0012"], np.arange(-5.0, 5.01, 1.0), [3.0e6], mach=0.1)

    def test_lift_curve_slope_near_two_pi(self):
        fit = np.polyfit(self.df["alpha"], self.df["CL"], 1)
        cl_alpha_per_deg = fit[0]
        cl_alpha_per_rad = np.degrees(cl_alpha_per_deg)
        self.assertAlmostEqual(cl_alpha_per_deg, 0.11, delta=0.02)
        self.assertAlmostEqual(cl_alpha_per_rad, 2 * np.pi, delta=1.0)

    def test_symmetric_section_has_zero_lift_at_zero_alpha(self):
        cl0 = float(np.interp(0.0, self.df["alpha"], self.df["CL"]))
        self.assertAlmostEqual(cl0, 0.0, delta=0.02)

    def test_symmetric_section_has_near_zero_moment(self):
        cm0 = float(np.interp(0.0, self.df["alpha"], self.df["CM"]))
        self.assertAlmostEqual(cm0, 0.0, delta=0.02)

    def test_drag_bucket_magnitude(self):
        # Published NACA 0012 at Re = 3e6 has minimum section CD around 0.006.
        self.assertAlmostEqual(float(self.df["CD"].min()), 0.006, delta=0.003)


class TestMissingDependencyMessage(unittest.TestCase):
    def test_message_is_actionable(self):
        self.assertIn("pip install aerosandbox", airfoil_doe.AEROSANDBOX_INSTALL_HINT)

    def test_run_doe_raises_runtime_error_without_aerosandbox(self):
        if airfoil_doe.HAS_AEROSANDBOX:
            self.skipTest("aerosandbox is installed")
        with self.assertRaises(RuntimeError):
            airfoil_doe.run_doe(["naca0012"], [0.0], [1.0e6])


if __name__ == "__main__":
    unittest.main()
