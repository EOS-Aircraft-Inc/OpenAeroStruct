"""The OAS weight build-up has to BE WingCalc's, or the comparison means nothing.

``studies/vsp_planform/weight`` ports WingCalc's ``compute_wing_weight`` so OAS can
weigh a wing on its own. A port drifts silently: a constant edited on one side, a
term summed in a different order, a unit changed. These tests hand the port
WingCalc's OWN geometry and sized box, frozen from a WingCalc 0.63.0 run
(``data/wingcalc_0.63.0_arcA_weight.json``), and require WingCalc's own answer
back, term by term. What differs between the two tools after that is geometry
and sizing -- modelling -- and never the formula.

They read the deck from the WingCalc checkout (``WINGCALC_ROOT``) with the two
overrides ``write_deck`` applies, which is itself under test: the reference run
was made on a deck ``write_deck`` wrote, so if the overrides drift, so does
MTOW and every Torenbeek term with it.
"""

import json
import unittest
from pathlib import Path

import numpy as np

from studies.vsp_planform.coupling import deck as wcdeck

_FIX = Path(__file__).parent / "data" / "wingcalc_0.63.0_arcA_weight.json"
_HAVE_DECK = (wcdeck.WC_DECK / "planformIn.csv").is_file()


def _port_on_wingcalc_geometry(match_plies=True, extra_box_lb=0.0):
    from studies.vsp_planform.coupling import geometry as wg
    from studies.vsp_planform.weight import buildup as bu
    from studies.vsp_planform.weight.deck_inputs import read_weight_deck

    fx = json.loads(_FIX.read_text())
    G = {k: np.array([np.nan if v is None else v for v in vals], dtype=float)
         for k, vals in fx["geometry"].items()}
    deck = read_weight_deck()
    ws = G["WS"]
    xg = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 201)))
    prof = wg.blended_profile(fx["section_blend"], xg)
    q = [bu.section_quantities(c, t, f, a, *prof(w), xg)
         for c, t, f, a, w in zip(G["Chord"], G["t/c"], G["Fwd Spar Ratio"],
                                  G["Aft Spar Ratio"], ws)]
    # Local EA sweep read back off WingCalc's own developed rib pitch, so the test
    # isolates the build-up from how the sweep is estimated.
    d = np.r_[np.diff(ws), ws[-1] - ws[-2]]
    pw = np.where(np.isnan(G["rib_pitch"]), d, G["rib_pitch"])
    g = bu.RibGeometry(
        ws=ws, rib_type=tuple(bu.classify(w, deck) for w in ws), chord=G["Chord"],
        x_le=G["LE_X_global"], z_qc=G["LE_Z_global"] + 0.25 * (G["TE_Z_global"] - G["LE_Z_global"]),
        z_ea=G["EA_Z_global"], t_c=G["t/c"], fwd=G["Fwd Spar Ratio"], aft=G["Aft Spar Ratio"],
        spar_h_fwd=G["Spar Height Fwd"], spar_h_aft=G["Spar Height Aft"],
        z_up_fwd=np.array([x["z_up_fwd"] for x in q]), z_up_aft=np.array([x["z_up_aft"] for x in q]),
        box_perimeter=np.array([x["box_perimeter"] for x in q]),
        oml_perimeter=np.array([x["oml_perimeter"] for x in q]),
        ea_sweep_deg=np.degrees(np.arccos(np.clip(d / pw, -1, 1))),
        seg_sweep_deg=G["Bay EA Sweep deg"][:-1])
    match = ({tuple(k.split("/")): v for k, v in fx["web_match_plies"].items()}
             if match_plies else None)
    splice = G["A_geom"][deck.splice_bay - 1] * deck.materials[deck.skin_material].density
    r = bu.wing_weight(g, deck, fx["terms"]["W_bays_full"] + extra_box_lb, G["Am"], splice, match)
    return r, fx["terms"], deck, bu


@unittest.skipUnless(_HAVE_DECK, "needs the WingCalc checkout under WINGCALC_ROOT")
class TestPortReproducesWingCalc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r, cls.ref, cls.deck, cls.bu = _port_on_wingcalc_geometry()

    def test_rib_layout_comes_from_the_rules(self):
        """The 20 stations are derived from planformIn.csv, not copied from the output."""
        ws = json.loads(_FIX.read_text())["geometry"]["WS"]
        np.testing.assert_allclose(self.bu.rib_stations(self.deck), ws, atol=1e-9)

    def test_every_term_matches(self):
        """Within 0.05% and 0.05 lb -- the residual is polyline vs boom perimeters."""
        for k, v in self.ref.items():
            if k not in self.r:
                continue
            with self.subTest(term=k):
                self.assertLess(abs(self.r[k] - v), max(0.05, 5e-4 * abs(v)),
                                f"{k}: port {self.r[k]:.4f} vs WingCalc {v:.4f}")

    def test_wing_total(self):
        self.assertAlmostEqual(self.r["W_wing"], self.ref["W_wing"], delta=0.1)

    def test_exactly_reproduced_terms(self):
        """Terms with no perimeter in them must match to rounding, not to tolerance."""
        for k in ("W_ribs_full", "W_splices", "W_nacelle_reinforcements",
                  "W_landing_gear_reinforcements", "W_wing_fuselage_attach",
                  "W_leading_edge_fixed", "W_fixed_trailing_edge", "W_trailing_edge_flaps",
                  "W_aileron", "W_mfs", "W_wingtip", "W_ply_drops_clips"):
            with self.subTest(term=k):
                self.assertAlmostEqual(self.r[k], self.ref[k], delta=5e-3)

    def test_nacelle_fittings(self):
        """Governing loads, conditions and ply counts, straight off WingCalc's table."""
        want = {"Inboard": ((18171.8, 18171.9), ("6g down", "6g down"), (10, 10, 5)),
                "Outboard": ((18222.4, 18120.7), ("6g down", "6g down"), (10, 10, 5))}
        for n in self.r["detail"]["nacelle_reinforcement"]:
            loads, conds, plies = want[n["nacelle"]]
            self.assertEqual(tuple(x["condition"] for x in n["ribs"]), conds)
            np.testing.assert_allclose([x["load"] for x in n["ribs"]], loads, atol=0.06)
            self.assertEqual(tuple(p["pad_plies"] for p in n["pads"]), plies)


@unittest.skipUnless(_HAVE_DECK, "needs the WingCalc checkout under WINGCALC_ROOT")
class TestWhatOASCannotCompute(unittest.TestCase):
    def test_web_match_to_cap_is_the_only_gap_and_is_reported(self):
        """Without WingCalc's cap plies the nacelle term drops by exactly that rule."""
        with_m, ref, _d, _b = _port_on_wingcalc_geometry(match_plies=True)
        without, _r, _d, _b = _port_on_wingcalc_geometry(match_plies=False)
        gap = with_m["W_nacelle_reinforcements"] - without["W_nacelle_reinforcements"]
        self.assertAlmostEqual(gap, 34.07, delta=0.05)
        self.assertIn("not computed", without["detail"]["web_match_to_cap"])


@unittest.skipUnless(_HAVE_DECK, "needs the WingCalc checkout under WINGCALC_ROOT")
class TestDeckOverrides(unittest.TestCase):
    """The overrides write_deck applies before WingCalc ever sees the deck."""

    def test_case_weights(self):
        from studies.vsp_planform.weight.deck_inputs import read_weight_deck

        d = read_weight_deck()
        by = {r["Load_case"]: float(r["AC_Weight"]) for r in d.load_cases}
        self.assertAlmostEqual(by["2.5g Upbend1 flight"], 81920.0)     # dry wing tanks
        self.assertAlmostEqual(by["1-wheel landing"], 86000.0)         # full
        self.assertAlmostEqual(d.half_wingbox_span, 678.0)

    def test_gear_rides_in_the_inboard_nacelle(self):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "studies" / "vsp_planform" / "scripts"))
        import arc_aerostruct as aa

        (ib, m_ib, y_ib, xc_ib, dz_ib), (ob, m_ob, y_ob, xc_ob, dz_ob) = aa.NACELLES
        self.assertAlmostEqual(m_ib, 7002.004286 + 1203.055644, places=4)
        self.assertAlmostEqual(m_ob, 7002.004286, places=4)
        # Forward of WingCalc's mid-spar axis (0.435c on Arc A) -- the nose-down
        # torque the old on-axis placement zeroed.
        self.assertLess(xc_ib, 0.435)
        self.assertLess(xc_ob, 0.435)


class TestPartialsAreTheChainRule(unittest.TestCase):
    def test_box_multiplier(self):
        """Box-linear terms reach W_wing through ply drops then paint on k_misc."""
        from studies.vsp_planform.weight import buildup as bu

        self.assertAlmostEqual(bu.dW_wing_dW_box(), 1.02 * (1 + 0.02 * 1.04), places=12)

    @unittest.skipUnless(_HAVE_DECK, "needs the WingCalc checkout under WINGCALC_ROOT")
    def test_multiplier_is_what_the_sum_does(self):
        """Add 100 lb of box: W_wing must move by exactly 100 x dW_wing_dW_box.

        W_wing is affine in the box, so a finite nudge is exact. This is the claim
        the OpenMDAO component's analytic partials are built on.
        """
        r0, _ref, _d, bu = _port_on_wingcalc_geometry()
        r1, _ref, _d, _b = _port_on_wingcalc_geometry(extra_box_lb=100.0)
        self.assertAlmostEqual(r1["W_wing"] - r0["W_wing"], 100.0 * bu.dW_wing_dW_box(), places=8)


if __name__ == "__main__":
    unittest.main()
