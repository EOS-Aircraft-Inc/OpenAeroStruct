"""Unit tests for the region-wise planform parameterization.

These run on a synthetic three-region wing built to the same recipe as the two
VSP baselines, so they do not depend on the CSV parser or on the meshing module.
"""

import unittest
import warnings

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from openaerostruct.utils.check_surface_dict import check_surface_dict_keys
from openaerostruct.utils.testing import run_test

from studies.vsp_planform.degen_csv import DegenStick
from studies.vsp_planform.param import (
    RegionPlanform,
    _region_coords,
    baseline_planform,
    baseline_twist,
    baseline_wingbox_pct,
    build_geometry_group,
    build_surface,
    rear_spar_fraction,
    reloft_region_a,
    twist_cp_bounds,
)
from studies.vsp_planform.regions import Regions

# Synthetic "ConstChord"-like wing, in inches.
Y_A = 361.70
Y_C = 674.95
Y_TIP = 708.00
C_ROOT = 105.0
C_B_TIP = 40.0
C_TIP = 14.0
SWEEP_C = 71.0
# Dihedral: flat inboard, a break part way through region A, then region B, then
# the winglet. The break is what the real ConstChord loft does.
DIHEDRAL_A_BREAK = 6.70
DIHEDRAL_B = 4.39
DIHEDRAL_C = 42.0
TWIST_ROOT = 4.0
TWIST_TIP = -1.0
SCALE = 0.0254

# The straight-spar rule fixes region B's sweep, so the synthetic wing is built
# from a chosen spar fraction rather than from a chosen sweep.
PCT = 0.68087
SWEEP_B = np.degrees(np.arctan(PCT * (C_ROOT - C_B_TIP) / (Y_C - Y_A)))


def _sections(n_a=9, n_b=13, n_c=7):
    """Spanwise stations, leading edge, chord and twist of the synthetic wing."""
    y = np.concatenate(
        [
            np.linspace(0.0, Y_A, n_a, endpoint=False),
            np.linspace(Y_A, Y_C, n_b, endpoint=False),
            np.linspace(Y_C, Y_TIP, n_c),
        ]
    )

    x_le = np.zeros_like(y)
    chord = np.full_like(y, C_ROOT)

    in_b = (y > Y_A) & (y <= Y_C)
    t_b = (y[in_b] - Y_A) / (Y_C - Y_A)
    chord[in_b] = C_ROOT + (C_B_TIP - C_ROOT) * t_b
    # Straight spar at constant x: x_le = x_spar - p * c.
    x_le[in_b] = PCT * (C_ROOT - chord[in_b])

    # Region C: sweep ramps with span, integrated stationwise.
    in_c = y > Y_C
    t_c = (y[in_c] - Y_C) / (Y_TIP - Y_C)
    dy_c = np.diff(np.concatenate([[Y_C], y[in_c]]))
    sweep_c = SWEEP_B + (SWEEP_C - SWEEP_B) * t_c
    x_le[in_c] = PCT * (C_ROOT - C_B_TIP) + np.cumsum(dy_c * np.tan(np.radians(sweep_c)))
    chord[in_c] = C_B_TIP + (C_TIP - C_B_TIP) * t_c

    # Dihedral, integrated over the whole span. Region A is deliberately NOT
    # planar: the real ConstChord loft is flat for its first nine sections, then
    # breaks to 6.70 deg, then settles to region B's 4.39 deg. Nothing in the
    # parameterization may assume otherwise -- dihedral is frozen and the z
    # distribution is carried through verbatim, whatever shape it has.
    dihedral = np.where(y < 0.5 * Y_A, 0.0, DIHEDRAL_A_BREAK)
    dihedral[in_b] = DIHEDRAL_B
    dihedral[in_c] = DIHEDRAL_B + (DIHEDRAL_C - DIHEDRAL_B) * t_c
    z_le = np.concatenate([[0.0], np.cumsum(np.diff(y) * np.tan(np.radians(dihedral[1:])))])

    twist = TWIST_ROOT + (TWIST_TIP - TWIST_ROOT) * (y / Y_TIP)
    return y, x_le, z_le, chord, twist


def synthetic_stick():
    """A DegenStick whose chord is measured along x, i.e. untwisted sections.

    The twist is carried in the mesh instead, so ``chord`` and the x-extent
    agree and the analytic spar fraction is exactly recoverable.
    """
    y, x_le, z_le, chord, _ = _sections()
    columns = {
        "lex": x_le,
        "ley": y,
        "lez": z_le,
        "tex": x_le + chord,
        "tey": y.copy(),
        "tez": z_le.copy(),
        "chord": chord,
        "toc": 0.178 + (0.100 - 0.178) * (y / Y_TIP),
        "tLoc": np.full_like(y, 0.299),
    }
    return DegenStick(num_secs=y.size, columns=columns)


def synthetic_mesh(nx=5):
    """Right-hand half mesh in metres, root first, ordered LE -> TE."""
    y, x_le, z_le, chord, twist = _sections()
    xi = np.linspace(0.0, 1.0, nx)
    mesh = np.zeros((nx, y.size, 3))
    mesh[:, :, 0] = x_le + np.outer(xi, chord)
    mesh[:, :, 1] = y
    mesh[:, :, 2] = z_le - np.outer(xi, chord * np.tan(np.radians(twist)))
    return mesh * SCALE


def synthetic_regions(stick):
    y = stick.le[:, 1]
    idx_a_end = int(np.flatnonzero(np.isclose(y, Y_A))[0])
    idx_c_start = int(np.flatnonzero(np.isclose(y, Y_C))[0])
    return Regions(idx_a_end, idx_c_start, float(y[idx_a_end]), float(y[idx_c_start]))


def _planform_comp(mesh, regions, planform0, **kwargs):
    t, span_b = _region_coords(mesh, regions, SCALE)
    opts = dict(
        t=t,
        cx0=np.asarray(mesh[-1, :, 0] - mesh[0, :, 0]),
        span_b=span_b,
        c_a0=planform0["c_a0"] * SCALE,
        taper_B0=planform0["taper_B"],
        wingbox_pct0=planform0["wingbox_pct"],
        rule=planform0["rule"],
        width_t=0.15,
        width_c0=2.0,
    )
    opts.update(kwargs)
    return RegionPlanform(**opts)


class TestBaselineMeasurements(unittest.TestCase):
    def setUp(self):
        self.stick = synthetic_stick()
        self.regions = synthetic_regions(self.stick)

    def test_wingbox_pct_is_straight(self):
        """The fitted spar fraction must make the spar line genuinely straight."""
        pct, x_spar, max_dev = baseline_wingbox_pct(self.stick, self.regions)
        self.assertAlmostEqual(pct, PCT, places=10)
        self.assertAlmostEqual(x_spar, PCT * C_ROOT, places=8)
        self.assertLess(max_dev, 1e-9 * C_ROOT)

    def test_baseline_planform_recovers_inputs(self):
        p0 = baseline_planform(self.stick, self.regions, rule="preserved")
        self.assertAlmostEqual(p0["sweep_B"], SWEEP_B, places=10)
        self.assertAlmostEqual(p0["taper_B"], C_B_TIP / C_ROOT, places=10)
        self.assertAlmostEqual(p0["c_a0"], C_ROOT, places=10)

    def test_baseline_twist_from_mesh(self):
        tw = baseline_twist(synthetic_mesh())
        self.assertAlmostEqual(tw[0], TWIST_ROOT, places=10)
        self.assertAlmostEqual(tw[-1], TWIST_TIP, places=10)

    def test_twist_cp_bounds_are_shifted(self):
        lower, upper = twist_cp_bounds(synthetic_mesh(), 5, bounds=(-1.0, 5.0))
        self.assertEqual(lower.size, 5)
        np.testing.assert_allclose(upper - lower, 6.0)
        # The root sits at +4 deg baseline, so only +1 deg of headroom is left.
        self.assertAlmostEqual(upper[0], 1.0, places=10)


class TestRegionPlanformPartials(unittest.TestCase):
    """Analytic partials, checked against complex step at and away from baseline."""

    def _comp(self, rule):
        stick = synthetic_stick()
        regions = synthetic_regions(stick)
        p0 = baseline_planform(stick, regions, rule=rule)
        return _planform_comp(synthetic_mesh(), regions, p0)

    def test_partials_preserved(self):
        run_test(self, self._comp("preserved"), complex_flag=True, method="cs")

    def test_partials_root_le_fixed(self):
        run_test(self, self._comp("root_le_fixed"), complex_flag=True, method="cs")

    def test_partials_off_baseline(self):
        for rule in ("preserved", "root_le_fixed"):
            with self.subTest(rule=rule):
                prob = om.Problem(reports=False)
                prob.model.add_subsystem("comp", self._comp(rule), promotes=["*"])
                prob.setup(force_alloc_complex=True)
                prob.set_val("taper_B", 0.27)
                prob.set_val("wingbox_pct", 0.52)
                prob.run_model()
                data = prob.check_partials(method="cs", compact_print=True, out_stream=None)
                assert_check_partials(data, atol=1e-9, rtol=1e-9)


class TestPlanformMap(unittest.TestCase):
    def setUp(self):
        self.stick = synthetic_stick()
        self.regions = synthetic_regions(self.stick)
        self.mesh = synthetic_mesh()

    def _run(self, rule="preserved", **dvs):
        p0 = baseline_planform(self.stick, self.regions, rule=rule)
        prob = om.Problem(reports=False)
        prob.model.add_subsystem("comp", _planform_comp(self.mesh, self.regions, p0), promotes=["*"])
        prob.setup()
        for key, val in dvs.items():
            prob.set_val(key, val)
        prob.run_model()
        return prob

    def _deform(self, rule="preserved", **dvs):
        p0 = baseline_planform(self.stick, self.regions, rule=rule)
        surface = build_surface(self.mesh, self.stick, self.regions)
        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            "geom",
            build_geometry_group(surface, self.regions, p0, scale=SCALE),
            promotes=["*"],
        )
        prob.setup()
        for key, val in dvs.items():
            prob.set_val(key, val)
        prob.run_model()
        return prob

    def test_baseline_is_identity(self):
        for rule in ("preserved", "root_le_fixed"):
            with self.subTest(rule=rule):
                prob = self._run(rule)
                np.testing.assert_allclose(prob.get_val("chord"), 1.0, atol=1e-14)
                np.testing.assert_allclose(prob.get_val("xshear"), 0.0, atol=1e-14)

    def test_geometry_group_reproduces_baseline_mesh(self):
        err = np.abs(self._deform().get_val("mesh") - self.mesh)
        self.assertLess(err.max(), 1e-12, f"max node error {err.max():.3e} m")

    def test_oas_geometry_mesh_would_not_reproduce_the_baseline(self):
        """Why MeshTransforms exists rather than OAS's GeometryMesh.

        ``GeometryMesh`` builds ``Rotate`` with its ``rotate_x=True`` default
        (the ``self.rotate_x`` it sets is never passed through). That applies
        ``Rx(dihedral)`` to every section even at zero twist, which re-tilts a
        winglet whose sections are already lofted in the winglet's plane. If
        this test ever starts passing, ``Rotate`` has been fixed upstream and
        ``MeshTransforms`` can be reconsidered.
        """
        from openaerostruct.geometry.geometry_mesh import GeometryMesh

        surface = build_surface(self.mesh, self.stick, self.regions)
        surface = {**surface, "chord_cp": np.ones(2), "xshear_cp": np.zeros(2)}
        prob = om.Problem(reports=False)
        prob.model.add_subsystem("mesh", GeometryMesh(surface=surface), promotes=["*"])
        prob.setup()
        prob.run_model()
        err = np.abs(prob.get_val("mesh") - self.mesh).max()
        self.assertGreater(err, 1e-6, "OAS GeometryMesh is now identity at zero twist; MeshTransforms may be moot")

    def test_derived_sweep_matches_the_baseline(self):
        prob = self._run("preserved")
        self.assertAlmostEqual(prob.get_val("sweep_B", units="deg")[0], SWEEP_B, places=8)

    def test_spar_stays_straight_and_put(self):
        """For any design, x_le + p * cx must be the baseline constant over A+B."""
        for rule, pct in (("preserved", 0.50), ("root_le_fixed", 0.72)):
            with self.subTest(rule=rule):
                p0 = baseline_planform(self.stick, self.regions, rule=rule)
                prob = self._deform(rule, wingbox_pct=pct, taper_B=0.25)
                mesh_new = prob.get_val("mesh")
                ic = self.regions.idx_c_start
                cx = mesh_new[-1, : ic + 1, 0] - mesh_new[0, : ic + 1, 0]
                spar = mesh_new[0, : ic + 1, 0] + pct * cx
                np.testing.assert_allclose(spar, p0["x_spar"] * SCALE, atol=1e-12)

    def test_region_a_chord_rules(self):
        pct = baseline_planform(self.stick, self.regions, rule="preserved")["wingbox_pct"]
        ia = self.regions.idx_a_end

        chord = self._run("preserved", wingbox_pct=0.9 * pct).get_val("chord")
        np.testing.assert_allclose(chord[: ia + 1], 1.0, rtol=1e-13)

        chord = self._run("root_le_fixed", wingbox_pct=0.9 * pct).get_val("chord")
        np.testing.assert_allclose(chord[: ia + 1], 1.0 / 0.9, rtol=1e-13)

    def test_root_le_fixed_holds_the_root_leading_edge(self):
        mesh_new = self._deform("root_le_fixed", wingbox_pct=0.55).get_val("mesh")
        self.assertAlmostEqual(mesh_new[0, 0, 0], self.mesh[0, 0, 0], places=12)

    def test_winglet_is_welded_to_region_b(self):
        prob = self._deform("preserved", taper_B=0.25)
        chord = prob.get_val("chord")
        mesh_new = prob.get_val("mesh")
        ic = self.regions.idx_c_start
        # Every winglet station shares region B's tip chord factor ...
        np.testing.assert_allclose(chord[ic:], chord[ic], rtol=1e-13)
        # ... and the leading edge translates rigidly.
        dx = mesh_new[0, ic:, 0] - self.mesh[0, ic:, 0]
        np.testing.assert_allclose(dx, dx[0], atol=1e-12)

    def test_taper_does_not_move_region_a(self):
        mesh_new = self._deform("preserved", taper_B=0.30).get_val("mesh")
        ia = self.regions.idx_a_end
        np.testing.assert_allclose(mesh_new[:, : ia + 1, :], self.mesh[:, : ia + 1, :], atol=1e-12)

    def test_region_a_is_not_planar(self):
        """Guard the fixture: region A must contain a dihedral break.

        If this ever stops holding, ``test_dihedral_and_span_are_frozen`` has
        quietly stopped testing the case that matters.
        """
        ia = self.regions.idx_a_end
        z_ref = 0.25 * self.mesh[-1, : ia + 1, 2] + 0.75 * self.mesh[0, : ia + 1, 2]
        y = self.mesh[0, : ia + 1, 1]
        dihedral = np.degrees(np.arctan2(np.diff(z_ref), np.diff(y)))
        self.assertGreater(dihedral.max() - dihedral.min(), 1.0, "region A came out planar")

    def test_dihedral_and_span_are_frozen(self):
        """The reference-axis z line must not move for any planform change.

        Region A of these wings has a dihedral break in it, so "frozen" has to
        mean the z distribution is carried through verbatim, not that some
        single dihedral angle is held.
        """
        mesh_new = self._deform("root_le_fixed", taper_B=0.3, wingbox_pct=0.5).get_val("mesh")
        z_ref_old = 0.25 * self.mesh[-1, :, 2] + 0.75 * self.mesh[0, :, 2]
        z_ref_new = 0.25 * mesh_new[-1, :, 2] + 0.75 * mesh_new[0, :, 2]
        np.testing.assert_allclose(z_ref_new, z_ref_old, atol=1e-12)
        np.testing.assert_allclose(mesh_new[:, :, 1], self.mesh[:, :, 1], atol=1e-12)


class TestRearSparSchedule(unittest.TestCase):
    """The rear spar is a spanwise schedule, so it can kink."""

    # The wing 2 design point: constant 0.75c inboard, kinking forward to 0.499c
    # at the winglet junction.
    SCHEDULE = ((356.0, 0.750), (674.9, 0.499))

    def test_constant_schedule_is_the_old_single_fraction(self):
        np.testing.assert_allclose(rear_spar_fraction([0.0, 100.0, 700.0], ((0.0, 0.75),)), 0.75)

    def test_breakpoints_are_hit_exactly_and_held_outside(self):
        y = np.array([0.0, 176.0, 356.0, 674.9, 708.0])
        rear = rear_spar_fraction(y, self.SCHEDULE)
        # Flat inboard of the first knot and outboard of the last.
        np.testing.assert_allclose(rear[:3], 0.750)
        np.testing.assert_allclose(rear[3:], 0.499)

    def test_kink_is_linear_in_span(self):
        y_mid = 0.5 * (356.0 + 674.9)
        self.assertAlmostEqual(
            float(rear_spar_fraction(y_mid, self.SCHEDULE)), 0.5 * (0.750 + 0.499), places=12
        )

    def test_unsorted_breakpoints_are_accepted(self):
        shuffled = tuple(reversed(self.SCHEDULE))
        np.testing.assert_allclose(
            rear_spar_fraction([0.0, 500.0, 700.0], shuffled),
            rear_spar_fraction([0.0, 500.0, 700.0], self.SCHEDULE),
        )

    def test_bad_schedule_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            rear_spar_fraction(0.0, (0.75, 0.5))


class TestWidthStations(unittest.TestCase):
    """``wingbox_width`` is a vector over the constraint stations."""

    def setUp(self):
        self.stick = synthetic_stick()
        self.regions = synthetic_regions(self.stick)
        self.mesh = synthetic_mesh()
        self.p0 = baseline_planform(self.stick, self.regions, rule="root_le_fixed")

    def _run(self, **kwargs):
        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            "comp", _planform_comp(self.mesh, self.regions, self.p0, **kwargs), promotes=["*"]
        )
        prob.setup(force_alloc_complex=True)
        return prob

    def test_width_is_one_entry_per_station(self):
        prob = self._run(
            width_t=np.array([0.0, 0.3, 1.0]),
            width_c0=np.array([2.0, 1.5, 1.0]),
            front_pct=0.125,
            rear_pct=np.array([0.75, 0.70, 0.499]),
        )
        prob.run_model()
        chord = prob.get_val("station_chord")
        width = prob.get_val("wingbox_width")
        self.assertEqual(width.shape, (3,))
        # width = (rear - front) * chord, station by station.
        np.testing.assert_allclose(width, (np.array([0.75, 0.70, 0.499]) - 0.125) * chord, rtol=1e-12)

    def test_kinked_rear_spar_narrows_the_box_outboard(self):
        """A forward-kinking rear spar must cut the outboard box, not the inboard."""
        straight = self._run(
            width_t=np.array([0.0, 1.0]), width_c0=np.array([2.0, 1.0]), front_pct=0.125, rear_pct=0.75
        )
        straight.run_model()
        kinked = self._run(
            width_t=np.array([0.0, 1.0]),
            width_c0=np.array([2.0, 1.0]),
            front_pct=0.125,
            rear_pct=np.array([0.75, 0.499]),
        )
        kinked.run_model()

        w_straight = straight.get_val("wingbox_width")
        w_kinked = kinked.get_val("wingbox_width")
        self.assertAlmostEqual(w_kinked[0], w_straight[0], places=12)
        self.assertLess(w_kinked[1], w_straight[1])
        # The chord itself is untouched: the schedule moves an edge of the box,
        # not the planform.
        np.testing.assert_allclose(kinked.get_val("station_chord"), straight.get_val("station_chord"), rtol=1e-12)

    def test_partials_with_vector_stations(self):
        for rule in ("preserved", "root_le_fixed"):
            with self.subTest(rule=rule):
                p0 = baseline_planform(self.stick, self.regions, rule=rule)
                prob = om.Problem(reports=False)
                prob.model.add_subsystem(
                    "comp",
                    _planform_comp(
                        self.mesh,
                        self.regions,
                        p0,
                        width_t=np.array([0.0, 0.3, 1.0]),
                        width_c0=np.array([2.0, 1.5, 1.0]),
                        front_pct=0.125,
                        rear_pct=np.array([0.75, 0.70, 0.499]),
                    ),
                    promotes=["*"],
                )
                prob.setup(force_alloc_complex=True)
                prob.set_val("taper_B", 0.31)
                prob.set_val("wingbox_pct", 0.61)
                prob.run_model()
                data = prob.check_partials(method="cs", compact_print=True, out_stream=None)
                assert_check_partials(data, atol=1e-9, rtol=1e-9)

    def test_geometry_group_samples_the_schedule(self):
        """The group turns (stations, schedule) into the per-station rear edge."""
        surface = build_surface(self.mesh, self.stick, self.regions)
        stations = [100.0, 361.7, 674.95]
        schedule = ((361.7, 0.75), (674.95, 0.499))

        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            "wing",
            build_geometry_group(
                surface, self.regions, self.p0, width_stations=stations, rear_schedule=schedule
            ),
            promotes=["*"],
        )
        prob.setup()
        prob.run_model()

        chord = prob.get_val("station_chord", units="m")
        width = prob.get_val("wingbox_width", units="m")
        rear = rear_spar_fraction(stations, schedule)
        np.testing.assert_allclose(width, (rear - 0.125) * chord, rtol=1e-12)
        # Baseline design vector, so the chords are the mesh's own.
        y_abs = np.abs(self.mesh[0, :, 1])
        cx0 = self.mesh[-1, :, 0] - self.mesh[0, :, 0]
        np.testing.assert_allclose(chord, np.interp(np.array(stations) * SCALE, y_abs, cx0), rtol=1e-12)


class TestRelofting(unittest.TestCase):
    """Moving the A|B breakpoint by rebuilding the chord distribution."""

    def setUp(self):
        self.stick = synthetic_stick()
        self.regions = synthetic_regions(self.stick)
        self.mesh = synthetic_mesh()
        self.p0 = baseline_planform(self.stick, self.regions, rule="root_le_fixed")

    @staticmethod
    def _chord_at(mesh, y_in):
        y = np.abs(mesh[0, :, 1])
        return np.interp(np.asarray(y_in) * SCALE, y, mesh[-1, :, 0] - mesh[0, :, 0]) / SCALE

    def test_same_breakpoint_is_the_identity(self):
        """Asking for the breakpoint the baseline already has must change nothing."""
        mesh, stick, regions = reloft_region_a(self.mesh, self.stick, self.regions, Y_A, self.p0)
        np.testing.assert_allclose(mesh, self.mesh, atol=1e-12)
        np.testing.assert_allclose(stick.le, self.stick.le, atol=1e-10)
        np.testing.assert_allclose(stick.te, self.stick.te, atol=1e-10)
        self.assertEqual(regions.as_tuple(), self.regions.as_tuple())

    def test_region_a_and_the_winglet_are_untouched(self):
        y_new = 176.0
        mesh, stick, regions = reloft_region_a(self.mesh, self.stick, self.regions, y_new, self.p0)
        y = np.abs(self.mesh[0, :, 1]) / SCALE
        outside = (y <= regions.y_a_end) | (y >= Y_C)
        np.testing.assert_allclose(mesh[:, outside, :], self.mesh[:, outside, :], atol=1e-12)

    def test_region_b_becomes_a_straight_taper_from_the_new_breakpoint(self):
        y_new = 176.0
        mesh, stick, regions = reloft_region_a(self.mesh, self.stick, self.regions, y_new, self.p0)
        y_a = regions.y_a_end

        stations = np.linspace(y_a, Y_C, 9)
        chord = self._chord_at(mesh, stations)
        # Linear in y between the breakpoint chord and the junction chord.
        expected = np.linspace(chord[0], chord[-1], 9)
        np.testing.assert_allclose(chord, expected, rtol=1e-9)

        # And it is a taper, not the bulge the un-relofted map produces: the
        # chord must fall monotonically from the breakpoint outboard.
        self.assertTrue(np.all(np.diff(chord) < 0.0))
        self.assertLess(chord[-1], chord[0])

    def test_the_baseline_spar_line_is_preserved(self):
        """The re-loft moves the leading edge along the baseline's own spar."""
        mesh, stick, regions = reloft_region_a(self.mesh, self.stick, self.regions, 176.0, self.p0)
        p = self.p0["wingbox_pct"]

        def spar(m):
            le, te = m[0, :, 0], m[-1, :, 0]
            return le + p * (te - le)

        np.testing.assert_allclose(spar(mesh), spar(self.mesh), atol=1e-12)

    def test_twist_is_carried_through(self):
        mesh, _, _ = reloft_region_a(self.mesh, self.stick, self.regions, 176.0, self.p0)
        np.testing.assert_allclose(baseline_twist(mesh), baseline_twist(self.mesh), atol=1e-10)

    def test_span_and_dihedral_are_frozen(self):
        mesh, _, _ = reloft_region_a(self.mesh, self.stick, self.regions, 176.0, self.p0)
        np.testing.assert_allclose(mesh[:, :, 1], self.mesh[:, :, 1], atol=1e-12)
        # The quarter-chord z line is the dihedral, and nothing here may move it.
        def ref_z(m):
            return 0.75 * m[0, :, 2] + 0.25 * m[-1, :, 2]

        np.testing.assert_allclose(ref_z(mesh), ref_z(self.mesh), atol=1e-12)

    def test_matches_the_oas_transform_chain(self):
        """The hand-rolled ScaleX/ShearX must equal what MeshTransforms does."""
        from openaerostruct.geometry.geometry_mesh_transformations import ScaleX, ShearX

        from studies.vsp_planform.param import _reloft_factor

        y_new = 176.0
        mesh, _, regions = reloft_region_a(self.mesh, self.stick, self.regions, y_new, self.p0)

        y_s = np.abs(self.stick.le[:, 1])
        cx_s = self.stick.te[:, 0] - self.stick.le[:, 0]
        f = _reloft_factor(np.abs(self.mesh[0, :, 1]) / SCALE, regions.y_a_end, Y_C, y_s, cx_s)
        cx0 = self.mesh[-1, :, 0] - self.mesh[0, :, 0]
        p = self.p0["wingbox_pct"]

        ny = self.mesh.shape[1]
        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            "scale", ScaleX(val=np.ones(ny), mesh_shape=self.mesh.shape, ref_axis_pos=0.25), promotes=["*"]
        )
        prob.model.add_subsystem("shear", ShearX(val=np.zeros(ny), mesh_shape=self.mesh.shape), promotes=["xshear"])
        prob.model.connect("mesh", "shear.in_mesh")
        prob.setup()
        prob.set_val("in_mesh", self.mesh, units="m")
        prob.set_val("chord", f)
        prob.set_val("xshear", (p - 0.25) * cx0 * (1.0 - f), units="m")
        prob.run_model()

        np.testing.assert_allclose(mesh, prob.get_val("shear.mesh", units="m"), atol=1e-12)

    def test_relofted_baseline_still_round_trips_through_the_geometry_group(self):
        """The re-lofted mesh is a baseline like any other: identity at its own DVs."""
        mesh, stick, regions = reloft_region_a(self.mesh, self.stick, self.regions, 176.0, self.p0)
        planform0 = baseline_planform(stick, regions, rule="root_le_fixed")
        surface = build_surface(mesh, stick, regions)

        prob = om.Problem(reports=False)
        prob.model.add_subsystem(
            "wing", build_geometry_group(surface, regions, planform0), promotes=["*"]
        )
        prob.setup()
        prob.run_model()
        np.testing.assert_allclose(prob.get_val("mesh", units="m"), mesh, atol=1e-12)

    def test_measured_taper_matches_the_rebuilt_geometry(self):
        """baseline_planform must read the re-lofted wing, not the original."""
        mesh, stick, regions = reloft_region_a(self.mesh, self.stick, self.regions, 176.0, self.p0)
        planform0 = baseline_planform(stick, regions, rule="root_le_fixed")

        chord = self._chord_at(mesh, [regions.y_a_end, Y_C])
        self.assertAlmostEqual(planform0["taper_B"], chord[1] / chord[0], places=8)
        # Region B is now a genuinely straight taper, so its spar fit is exact.
        self.assertLess(planform0["spar_max_dev"], 1e-9 * C_ROOT)

    def test_breakpoint_outboard_of_the_junction_is_rejected(self):
        with self.assertRaises(ValueError):
            reloft_region_a(self.mesh, self.stick, self.regions, Y_TIP, self.p0)


class TestSurfaceDict(unittest.TestCase):
    def setUp(self):
        self.stick = synthetic_stick()
        self.regions = synthetic_regions(self.stick)
        self.surface = build_surface(synthetic_mesh(), self.stick, self.regions)

    def test_uses_t_over_c_cp_not_scalar(self):
        self.assertIn("t_over_c_cp", self.surface)
        for key in ("t_over_c", "dihedral", "sweep", "taper"):
            self.assertNotIn(key, self.surface)
        self.assertAlmostEqual(self.surface["c_max_t"], 0.299, places=10)
        self.assertTrue(np.all(self.surface["t_over_c_cp"] > 0.09))
        self.assertTrue(np.all(self.surface["t_over_c_cp"] < 0.19))

    def test_surface_dict_passes_oas_key_check(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_surface_dict_keys(self.surface)


if __name__ == "__main__":
    unittest.main()
