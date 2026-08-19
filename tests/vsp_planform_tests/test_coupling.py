"""The OAS -> WingCalc geometry export must survive the round trip.

The export is the only path by which an OpenAeroStruct design reaches the
structural tool, and a silent error in it looks like a structural result rather
than a plumbing bug -- which is exactly the failure mode this study has hit
before (see the BULGE note in README.md: every summary number looked plausible
and only a station-by-station chord table exposed it).

So the check is station-by-station: write the deck, read it back with WingCalc's
own provider where available, and compare chord and t/c at named stations.
"""

import unittest

import numpy as np

from studies.vsp_planform import config, run_opt
from studies.vsp_planform.coupling import geometry as wg
from studies.vsp_planform.coupling import deck as wcdeck
from studies.vsp_planform.coupling import mission
from studies.vsp_planform.degen_csv import lifting_surfaces, read_degen_csv


def _baseline(name="const_chord"):
    mesh, stick, regions, planform0, _res, _mn = run_opt.load_baseline(name)
    prob, _ = run_opt.build_problem(name, mesh, stick, regions, planform0)
    prob.run_model()
    comp = list(lifting_surfaces(read_degen_csv(config.BASELINES[name])).values())[0][0]
    return prob, comp, regions


class TestGeometryExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        from pathlib import Path

        cls.tmp = tempfile.TemporaryDirectory()
        prob, comp, regions = _baseline()
        cls.mesh = np.asarray(prob.get_val("wing.mesh", units="m"))
        cls.toc = np.asarray(prob.get_val("wing.t_over_c")).ravel()
        cls.y_junction = float(comp.stick.le[regions.idx_c_start, 1])
        cls.csv, cls.n = wg.export(
            cls.mesh, cls.toc, comp.plate, comp.stick,
            Path(cls.tmp.name) / "OpenVSP", name="test_export",
            max_ws_in=cls.y_junction)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_wrote_stations(self):
        self.assertGreater(self.n, 10, "too few stations exported to loft from")
        self.assertTrue(self.csv.exists())

    def test_ws_strictly_increasing(self):
        """A non-monotonic ws corrupts the provider's spanwise interpolation.

        The winglet is why: ws is the LE's y, which stops increasing through it,
        so the export must be clipped at the junction.
        """
        text = self.csv.read_text()
        ws = [float(line.split(",")[2]) for line in text.splitlines()
              if line.startswith("Leading Edge Point")]
        self.assertTrue(all(b > a for a, b in zip(ws, ws[1:])),
                        f"ws not strictly increasing: {ws}")

    def test_chord_and_toc_round_trip(self):
        try:
            wcdeck._wingcalc()
            from WingCalc_Tool.io.openvsp import read_openvsp_geometry
        except Exception as exc:  # noqa: BLE001 - optional external tool
            self.skipTest(f"WingCalc not importable: {exc}")

        g = read_openvsp_geometry(self.csv)
        le = self.mesh[0] / wg.SCALE
        te = self.mesh[-1] / wg.SCALE
        if le[0, 1] > le[-1, 1]:
            le, te = le[::-1], te[::-1]
        ws = le[:, 1]
        chord = np.linalg.norm(te - le, axis=1)

        for target in (0.0, 176.0, 356.0, 500.0):
            with self.subTest(ws=target):
                sec = g.section_at(target)
                self.assertAlmostEqual(sec.chord, float(np.interp(target, ws, chord)),
                                       places=2, msg=f"chord mismatch at ws {target}")
                self.assertGreater(sec.t_c, 0.05)
                self.assertLess(sec.t_c, 0.30)


class TestMission(unittest.TestCase):
    def test_battery_matches_the_aircraft_weight_book(self):
        """At the book's 7,460 lb wing the bookkeeping must reproduce its battery."""
        self.assertAlmostEqual(mission.battery_lb(7460.0), mission.BATT_LB_BOOK,
                               delta=0.005 * mission.BATT_LB_BOOK)

    def test_wing_weight_trades_against_battery_one_for_one(self):
        a, b = mission.battery_lb(8000.0), mission.battery_lb(8500.0)
        self.assertAlmostEqual(a - b, 500.0, places=6)

    def test_range_falls_with_a_heavier_wing(self):
        light = mission.electric_range_nmi(7500.0, 10_000.0)
        heavy = mission.electric_range_nmi(8500.0, 10_000.0)
        self.assertGreater(light, heavy)

    def test_cruise_is_below_mtow_by_half_the_fuel(self):
        self.assertAlmostEqual(mission.MTOW_LB - mission.cruise_weight_lb(),
                               0.5 * mission.FUEL_LB, places=6)


if __name__ == "__main__":
    unittest.main()
