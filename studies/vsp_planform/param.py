"""Region-wise planform parameterization of a VSP wing for OpenAeroStruct.

Why not just use OAS's ``sweep`` / ``taper`` / ``dihedral`` surface keys
-----------------------------------------------------------------------
Those three keys are *scalars* that drive one linear transform applied to the
whole surface (``openaerostruct/geometry/geometry_mesh_transformations.py``,
chained in ``geometry/geometry_mesh.py``). There is no way to say "sweep region
B only", and a 42-degree raked winglet is not expressible as a single shear at
all. Only the per-section distributions (``chord``, ``xshear``, ``zshear``,
``twist``, ``t_over_c``) have enough freedom.

So this module keeps the OAS mesh-transformation chain but replaces the
B-splines that normally drive ``chord`` and ``xshear`` with
:class:`RegionPlanform`, an explicit component that maps two scalar design
variables onto the full-span ``chord`` (scale factors) and ``xshear`` (metres)
distributions.

The planform model
------------------
Three regions, split at the spanwise stations ``y_a`` (end of the constant
chord bay A) and ``y_c`` (start of the winglet C)::

    A: 0    <= y <= y_a     constant chord, unswept leading edge
    B: y_a  <= y <= y_c     linear taper, constant leading-edge sweep
    C: y_c  <= y            the winglet, welded to the tip of B

The governing structural rule is that the aft wingbox spar is a **straight,
unswept line at constant x from the root out to the winglet**, across regions A
and B alike. Both baselines already satisfy it to within a fraction of a
percent of the root chord. Written as a chord fraction ``p``::

    x_LE(y) = x_spar - p * c(y)          (regions A and B)

which is built into the parameterization rather than imposed as a constraint --
cheaper, better conditioned, and exact.

Design variables
----------------
``wingbox_pct`` (p)
    The rear-spar location as a fraction of the local chord.

``taper_B`` (lambda)
    Tip/root chord ratio across region B.

Leading-edge sweep is **not** a design variable. The straight-spar rule makes it
a dependent quantity::

    tan(sweep_LE_B) = p * c_A * (1 - lambda) / span_B

which reproduces both baselines to ~0.02 deg. :class:`RegionPlanform` emits it
as an output for reporting only.

Region A's chord follows a per-geometry rule (``region_a_rule``):

``"preserved"`` (ConstChord)
    ``c_A`` is frozen at its baseline value. Changing ``p`` then slides region A
    fore/aft rigidly (no sweep, since the chord is constant there) and sweeps
    region B via the relation above.

``"root_le_fixed"`` (Plan_L)
    The root leading edge is held instead, so ``c_A = c_A0 * p0 / p``: the same
    physical wingbox occupying a smaller chord fraction means a longer chord, a
    trailing-edge extension. This is the geometry whose inboard wingbox width
    ``p * c(y)`` has to be constrained, since nothing else stops the chord from
    collapsing.

Region C is *welded* to region B: it translates rigidly with the moving B tip
and its chords scale by the same factor as B's tip chord, so the planform stays
continuous and the winglet keeps its shape.

Dihedral is frozen. Nothing here touches ``y`` or the reference-axis ``z``:
``ScaleX`` scales the mesh about the local reference axis, so the quarter-chord
z-line -- the dihedral -- is carried through from the parsed VSP mesh verbatim.
There is deliberately no ``zshear_cp`` and no ``dihedral`` key.

Everything is written as a *perturbation from the baseline*: the component
outputs ratios and offsets relative to the parsed mesh rather than absolute
chords and x-stations. That makes the baseline design vector an exact identity
map (machine-precision round trip) regardless of how well the real VSP loft
matches the idealized three-region model, and it means the spar line moves
exactly as the baseline spar did rather than being re-idealized.
"""

import numpy as np
import openmdao.api as om

from openaerostruct.geometry.geometry_mesh_transformations import Rotate, ScaleX, ShearX
from openaerostruct.utils.interpolation import get_normalized_span_coords

from studies.vsp_planform import config
from studies.vsp_planform.degen_csv import DegenStick
from studies.vsp_planform.regions import Regions

__all__ = [
    "RegionPlanform",
    "REGION_A_RULE",
    "baseline_planform",
    "rear_spar_fraction",
    "baseline_twist",
    "baseline_wingbox_pct",
    "reloft_region_a",
    "build_geometry_group",
    "build_surface",
    "mac_quarter_chord",
    "twist_cp_bounds",
]

# Per-geometry rule for what happens to region A's chord. See the module
# docstring.
#
# Both are "root_le_fixed": the wingbox is a fixed physical structure that stays
# straight and unswept, and the chord is free to grow aft of it as a
# trailing-edge extension. ConstChord was previously "preserved" (chord frozen),
# which forced every bit of an area change into the outboard taper -- that is
# what over-tapered it to 0.163 and drove the tip washin.
REGION_A_RULE = {
    "const_chord": "root_le_fixed",
    "plan_l": "root_le_fixed",
}

# Inboard wingbox width requirement for Plan_L: at least 65 in of box, from the
# root out to y = 100 in. The chord falls monotonically outboard, so only the
# outboard end of that run can bind. Kept as the default single station; the
# general form is ``config.WINGBOX_WIDTH_STATIONS``.
WINGBOX_WIDTH_MIN_IN = 65.0
WINGBOX_WIDTH_STATION_IN = 100.0

_RULE_EXPONENT = {"preserved": 0.0, "root_le_fixed": 1.0}


def rear_spar_fraction(y_in, schedule=None):
    """Rear-spar chord fraction at spanwise stations ``y_in``, in inches.

    ``schedule`` is a sequence of ``(y_in, fraction)`` breakpoints, linearly
    interpolated between them and held flat outside. The wing 2 design point is
    the two-breakpoint case ``((356.0, 0.75), (674.9, 0.499))``: constant
    inboard, kinking forward across the outer half of region B so that the box
    still fits at the winglet junction.
    """
    if schedule is None:
        schedule = config.WINGBOX_REAR_SCHEDULE

    knots = np.asarray(schedule, dtype=float)
    if knots.ndim != 2 or knots.shape[1] != 2:
        raise ValueError("rear spar schedule must be a sequence of (y_in, fraction) pairs")
    order = np.argsort(knots[:, 0])
    return np.interp(np.asarray(y_in, dtype=float), knots[order, 0], knots[order, 1])

_RAD2DEG = 180.0 / np.pi


# ---------------------------------------------------------------------------
# Measurements taken off the baseline geometry
# ---------------------------------------------------------------------------


def baseline_wingbox_pct(stick, regions):
    """Measure the chord fraction at which the baseline's spar is straight.

    A straight, unswept spar is a line ``x_le(y) + p * cx(y) = const`` where
    ``cx`` is the chordwise x-extent. Over region A the chord is constant and
    the leading edge is unswept, so *every* ``p`` gives a straight line there
    and the fraction is not identifiable; region B, where the wing both sweeps
    and tapers, pins it down. Fitting ``p`` by least squares over regions A and
    B together is the same thing as minimizing the variance of the spar's x
    station:

    .. math:: p = -\\mathrm{cov}(c_x,\\ x_{le}) / \\mathrm{var}(c_x)

    Returns
    -------
    pct : float
        The fitted spar chord fraction.
    x_spar : float
        Mean x station of the fitted spar, in the stick's own length units.
    max_dev : float
        Largest departure of the fitted spar from straight, same units. Near
        zero confirms the baseline really does have a straight unswept spar.
    """
    sl = slice(0, regions.idx_c_start + 1)
    x_le = np.asarray(stick.le[:, 0], dtype=float)[sl]
    cx = np.asarray(stick.te[:, 0], dtype=float)[sl] - x_le

    var_c = np.var(cx)
    if var_c <= 0.0:
        raise ValueError("chord is constant over regions A+B; the spar fraction is not identifiable")

    pct = -np.cov(cx, x_le, bias=True)[0, 1] / var_c
    spar = x_le + pct * cx
    return float(pct), float(np.mean(spar)), float(np.max(np.abs(spar - np.mean(spar))))


def baseline_twist(mesh):
    """Geometric twist of every spanwise station of a mesh, in degrees.

    Taken from the mesh rather than the stick so that it survives spanwise
    resampling. Positive is leading edge up, matching OAS's ``Rotate``.
    """
    le = mesh[0, :, :]
    te = mesh[-1, :, :]
    return np.degrees(np.arctan2(-(te[:, 2] - le[:, 2]), te[:, 0] - le[:, 0]))


def mac_quarter_chord(mesh):
    """Quarter-chord point of the mean aerodynamic chord, in mesh coordinates.

    The VSP models are not referenced to the wing root or to anything
    aerodynamic -- both sit around x = 24 to 27 m in VSP global coordinates --
    so leaving a moment reference at the origin would put a 25 m arm on every
    moment. This gives ``AeroPoint`` a reference that means something.

    MAC is the chord-weighted mean chord over the half span. The reference point
    takes its x from the quarter chord of the section at the MAC's spanwise
    station, but sits on the plane of symmetry (y = 0) at the root's
    reference-axis height -- the conventional choice for a symmetric aircraft.
    Putting it out at the MAC station instead would hang a 7 m arm on the
    rolling and yawing moments of a wing that has none by symmetry.

    Returns
    -------
    point : ndarray
        The (3,) moment reference point, metres.
    mac : float
        Mean aerodynamic chord, metres.
    y_mac : float
        Spanwise station of the MAC, metres.
    """
    le, te = mesh[0, :, :], mesh[-1, :, :]
    chord = np.linalg.norm(te - le, axis=1)
    y = np.abs(le[:, 1])

    weight = np.trapezoid(chord, y)
    mac = np.trapezoid(chord**2, y) / weight
    y_mac = np.trapezoid(chord * y, y) / weight

    ref_axis = 0.75 * le + 0.25 * te
    order = np.argsort(y)
    root = int(order[0])
    x = float(np.interp(y_mac, y[order], ref_axis[order, 0]))
    point = np.array([x, 0.0, ref_axis[root, 2]])
    return point, float(mac), float(y_mac)


def _reloft_factor(y_in, y_a_new, y_c, y_stick, cx_stick):
    """Chord scale factor that moves the A|B breakpoint to ``y_a_new``.

    The target is the baseline's own chord everywhere except between the new
    breakpoint and the winglet junction, where it is the straight taper joining
    them. So the factor is exactly 1 inboard of ``y_a_new`` and outboard of
    ``y_c``, and continuous at both.
    """
    y_in = np.asarray(y_in, dtype=float)
    base = np.interp(y_in, y_stick, cx_stick)

    c_a = float(np.interp(y_a_new, y_stick, cx_stick))
    c_j = float(np.interp(y_c, y_stick, cx_stick))
    taper = c_a + (c_j - c_a) * (y_in - y_a_new) / (y_c - y_a_new)

    inside_b = (y_in > y_a_new) & (y_in < y_c)
    return np.where(inside_b, taper, base) / base


def reloft_region_a(mesh, stick, regions, y_a_new_in, planform0=None, ref_axis_pos=0.25, scale=None):
    """Move the region A|B breakpoint by rebuilding the chord distribution.

    :class:`RegionPlanform` emits a chord *scale factor* on the baseline mesh,
    which is what makes the baseline design vector an exact identity. The price
    is that it can only rescale the chord distribution it is given, never
    reshape it: pointing it at a breakpoint inboard of the baseline's own leaves
    the constant-chord bay constant out to where the loft actually ends it, and
    asking for a smaller junction chord then *bulges* the middle of the span
    rather than tapering it. Measured on the wing 2 design point, region A moved
    361.7 -> 179.2 in: chord at y = 356 came out 115.7 in against a 105 in root.

    So a moved breakpoint has to be baked into the baseline before the
    parameterization sees it. This returns a new ``(mesh, stick, regions)`` whose
    region B is the straight taper from ``y_a_new_in`` to the winglet junction,
    ready to be handed to :func:`baseline_planform` and
    :func:`build_geometry_group` exactly as a parsed baseline would be.

    What is and is not re-idealized
    -------------------------------
    Only the span between the new breakpoint and the junction is rebuilt.
    Inboard of ``y_a_new_in`` and outboard of the junction the loft is carried
    through verbatim, wobble and all, so region A keeps whatever the real VSP
    model has and the winglet is untouched. The leading edge is placed by the
    baseline's *own* spar line -- ``x_le + p0 * cx`` evaluated station by station,
    not a fitted straight line -- so the spar moves exactly as it would under
    :class:`RegionPlanform`. Passing the breakpoint the baseline already has is
    therefore the identity to the loft's own straightness.

    Thickness is treated as a fixed *ratio*: ``toc`` and ``tLoc`` are carried
    through unchanged, so the section stays similar and scales with the chord.
    That is the same convention the optimizer works in, where ``t_over_c`` is its
    own design variable rather than something the chord drags around.

    Parameters
    ----------
    y_a_new_in : float
        Requested spanwise station of the new A|B breakpoint, inches. Snapped to
        the nearest native section so the returned ``Regions`` indices and the
        rebuilt geometry agree exactly; the snapped value is in
        ``regions.y_a_end``.
    planform0 : dict or None
        Baseline planform, for its ``wingbox_pct``. Measured here if omitted.

    Returns
    -------
    mesh, stick, regions
        The re-lofted baseline, in the same units and conventions as the inputs.
    """
    if scale is None:
        scale = config.SCALE
    if planform0 is None:
        planform0 = baseline_planform(stick, regions, rule="preserved")
    p0 = planform0["wingbox_pct"]

    le_s = np.asarray(stick.le, dtype=float)
    te_s = np.asarray(stick.te, dtype=float)
    y_signed = le_s[:, 1]
    y_s = np.abs(y_signed)
    cx_s = te_s[:, 0] - le_s[:, 0]

    # Snap to a native section: Regions carries indices, and a breakpoint that
    # fell between two sections would not be representable in them.
    idx_a_new = int(np.argmin(np.abs(y_s - abs(y_a_new_in))))
    y_a_new = float(y_s[idx_a_new])
    y_c = abs(float(regions.y_c_start))

    if not 0 <= idx_a_new < regions.idx_c_start:
        raise ValueError(f"new region A end (section {idx_a_new}) is not inboard of the winglet")
    if y_a_new >= y_c:
        raise ValueError(f"new region A end ({y_a_new} in) is not inboard of the junction ({y_c} in)")

    # --- the mesh. This is ScaleX followed by ShearX, spelled out: ScaleX
    # scales every coordinate about the reference axis, so the leading edge
    # moves by ref_axis_pos * cx0 * (1 - f) on its own, and the shear puts it
    # where the spar rule wants it. See MeshTransforms.
    mesh = np.asarray(mesh, dtype=float)
    y_mesh = np.abs(mesh[0, :, 1]) / scale
    f_mesh = _reloft_factor(y_mesh, y_a_new, y_c, y_s, cx_s)
    cx0_mesh = mesh[-1, :, 0] - mesh[0, :, 0]

    ref_axis = ref_axis_pos * mesh[-1] + (1.0 - ref_axis_pos) * mesh[0]
    mesh_new = np.einsum("ijk,j->ijk", mesh - ref_axis, f_mesh) + ref_axis
    mesh_new[:, :, 0] += (p0 - ref_axis_pos) * cx0_mesh * (1.0 - f_mesh)

    # --- the stick, by the same rule. Its x follows the spar line directly;
    # its z is scaled about the reference axis exactly as ScaleX does, so the
    # section twist is carried through unchanged.
    f_s = _reloft_factor(y_s, y_a_new, y_c, y_s, cx_s)
    cx_new = f_s * cx_s
    lex_new = le_s[:, 0] + p0 * cx_s * (1.0 - f_s)

    z_ref = ref_axis_pos * te_s[:, 2] + (1.0 - ref_axis_pos) * le_s[:, 2]
    columns = dict(stick.columns)
    columns["lex"] = lex_new
    columns["tex"] = lex_new + cx_new
    columns["lez"] = z_ref + (le_s[:, 2] - z_ref) * f_s
    columns["tez"] = z_ref + (te_s[:, 2] - z_ref) * f_s
    columns["chord"] = np.asarray(stick.chord, dtype=float) * f_s
    stick_new = DegenStick(num_secs=stick.num_secs, columns=columns)

    regions_new = Regions(
        idx_a_end=idx_a_new,
        idx_c_start=regions.idx_c_start,
        y_a_end=float(y_signed[idx_a_new]),
        y_c_start=regions.y_c_start,
    )
    return mesh_new, stick_new, regions_new


def _region_coords(mesh, regions, scale):
    """Normalized region coordinate ``t`` per mesh station, and region B's span.

    ``t`` is 0 through region A, ramps linearly 0 -> 1 across region B and is
    pinned at 1 through the winglet, which is what welds C to B's tip.
    """
    y = np.abs(np.asarray(mesh[0, :, 1], dtype=float))
    y_a = abs(regions.y_a_end * scale)
    y_c = abs(regions.y_c_start * scale)

    span_b = y_c - y_a
    if span_b <= 0.0:
        raise ValueError(f"region B has non-positive span ({span_b})")

    return np.clip((y - y_a) / span_b, 0.0, 1.0), span_b


def baseline_planform(stick, regions, name=None, rule=None):
    """Baseline values and fixed coefficients of the planform model.

    ``rule`` defaults to ``REGION_A_RULE[name]``.
    """
    if rule is None:
        if name is None:
            raise ValueError("pass either a baseline name or an explicit region A rule")
        rule = REGION_A_RULE[name]
    if rule not in _RULE_EXPONENT:
        raise ValueError(f"unknown region A rule {rule!r}")

    le = np.asarray(stick.le, dtype=float)
    te = np.asarray(stick.te, dtype=float)
    chord_x = te[:, 0] - le[:, 0]
    ia, ic = regions.idx_a_end, regions.idx_c_start

    dy = abs(le[ic, 1] - le[ia, 1])
    sweep_B0 = float(np.degrees(np.arctan2(le[ic, 0] - le[ia, 0], dy)))
    taper_B0 = float(chord_x[ic] / chord_x[ia])
    pct0, x_spar, max_dev = baseline_wingbox_pct(stick, regions)

    return {
        "rule": rule,
        "wingbox_pct": pct0,
        "taper_B": taper_B0,
        "sweep_B": sweep_B0,
        "c_a0": float(chord_x[ia]),
        "span_b": dy,
        "x_spar": x_spar,
        "spar_max_dev": max_dev,
    }


# ---------------------------------------------------------------------------
# The planform map
# ---------------------------------------------------------------------------


class RegionPlanform(om.ExplicitComponent):
    """Map the planform design variables onto ``chord`` / ``xshear``.

    Outputs are exactly what OAS's ``ScaleX`` and ``ShearX`` want: ``chord`` is
    a per-station multiplicative scale factor and ``xshear`` a per-station x
    translation in metres, both applied to the baseline mesh.

    ``ScaleX`` scales about the reference axis (quarter chord by default), so a
    scale factor ``f`` on its own would move the leading edge by
    ``r * cx0 * (1 - f)``. We want the leading edge to land exactly at
    ``x_le0 + dx``, so the shear has to undo that::

        xshear = dx - r * cx0 * (1 - f)

    with ``cx0`` the baseline chordwise x-extent of the station.

    Parameters
    ----------
    taper_B : float
        Tip/root chord ratio of region B.
    wingbox_pct : float
        Spar location as a fraction of chord.

    Returns
    -------
    chord[ny] : numpy array
        Chord scale factor per spanwise station.
    xshear[ny] : numpy array
        x translation per spanwise station, metres.
    sweep_B : float
        Resulting leading-edge sweep of region B, degrees. Reporting only.
    station_chord[n_width] : numpy array
        Local chord at each width-constraint station, metres.
    wingbox_width[n_width] : numpy array
        Width of the structural box, ``(rear_pct - front_pct) * c``, at each
        constraint station, metres. Does not move with ``wingbox_pct`` except
        through the chord: that is the straight line inside the box, not one of
        its edges. ``rear_pct`` is per station, so the rear spar may kink.
    """

    def initialize(self):
        self.options.declare("t", types=np.ndarray, desc="Region coordinate per station: 0 in A, 0->1 in B, 1 in C.")
        self.options.declare("cx0", types=np.ndarray, desc="Baseline chordwise x-extent per station, metres.")
        self.options.declare("span_b", types=float, desc="Spanwise extent of region B, metres.")
        self.options.declare("c_a0", types=float, desc="Baseline chordwise x-extent of region A, metres.")
        self.options.declare("taper_B0", types=float, desc="Baseline taper ratio of region B.")
        self.options.declare("wingbox_pct0", types=float, desc="Baseline spar chord fraction.")
        self.options.declare("rule", values=("preserved", "root_le_fixed"), desc="What holds region A's chord.")
        self.options.declare("width_t", default=0.0, desc="Region coordinate of each wingbox-width station.")
        self.options.declare("width_c0", default=1.0, desc="Baseline chord at each width station, metres.")
        self.options.declare("front_pct", default=0.0, desc="Box front edge as a fraction of chord.")
        self.options.declare("rear_pct", default=1.0, desc="Box rear edge at each width station, chord fraction.")
        self.options.declare("ref_axis_pos", default=0.25, desc="Fraction of chord used as the reference axis.")

    def setup(self):
        t = self.options["t"]
        cx0 = self.options["cx0"]

        # The width stations are a vector; scalars are accepted for the common
        # single-station case and broadcast against each other here.
        w_t = np.atleast_1d(np.asarray(self.options["width_t"], dtype=float))
        w_c0 = np.atleast_1d(np.asarray(self.options["width_c0"], dtype=float))
        rear = np.atleast_1d(np.asarray(self.options["rear_pct"], dtype=float))
        self._w_t, self._w_c0, self._rear = np.broadcast_arrays(w_t, w_c0, rear)
        n_w = self._w_t.size

        self.add_input("taper_B", val=self.options["taper_B0"])
        self.add_input("wingbox_pct", val=self.options["wingbox_pct0"])

        self.add_output("chord", val=np.ones(t.size))
        self.add_output("xshear", val=np.zeros(t.size), units="m")
        self.add_output("sweep_B", val=0.0, units="deg")
        self.add_output("station_chord", val=np.zeros(n_w), units="m")
        self.add_output("wingbox_width", val=np.zeros(n_w), units="m")

        # Which outputs actually respond to the spar fraction depends on the
        # region A rule, so declare only the structurally nonzero blocks.
        # "root_le_fixed" makes p * c invariant, which freezes the leading-edge
        # line, region B's sweep and the wingbox width; p then only scales chords.
        # "preserved" is the mirror image: p sweeps region B but leaves the chord
        # distribution alone.
        rule = self.options["rule"]
        self._root_le_fixed = rule == "root_le_fixed"
        self._width_varies = bool(np.any(self._w_t != 0.0))

        self.declare_partials(["chord", "xshear", "sweep_B"], "taper_B")
        self.declare_partials("xshear", "wingbox_pct")
        if self._root_le_fixed:
            self.declare_partials("chord", "wingbox_pct")
        else:
            self.declare_partials("sweep_B", "wingbox_pct")
        # The box spans fixed chord fractions *at each station*, so the width
        # tracks p only through the local chord. Under "preserved" the chord does
        # not depend on p at all, which makes that derivative identically zero.
        self._box_frac = self._rear - self.options["front_pct"]
        self._width_varies_with_p = self._root_le_fixed
        if self._width_varies_with_p:
            self.declare_partials(["station_chord", "wingbox_width"], "wingbox_pct")
        if self._width_varies:
            self.declare_partials(["station_chord", "wingbox_width"], "taper_B")

        # Constant coefficients. Everything below is a smooth closed-form
        # function of the two inputs with these frozen, which is what makes the
        # partials trivial and keeps the component complex-step safe.
        self._t = t
        self._e = _RULE_EXPONENT[self.options["rule"]]
        self._denom = 1.0 + (self.options["taper_B0"] - 1.0) * t
        self._r_cx0 = self.options["ref_axis_pos"] * cx0

        # The lever arm of the spar rule. Regions A and B use their own chord;
        # the winglet is welded to the junction, so it uses the junction's,
        # which makes its x offset a rigid translation.
        junction = int(np.argmax(t >= 1.0))
        self._cx_ref = np.where(t < 1.0, cx0, cx0[junction])

        self._width_denom = 1.0 + (self.options["taper_B0"] - 1.0) * self._w_t

    def _factors(self, lam, p):
        """Chord scale factor and its derivatives w.r.t. lambda and p."""
        p0 = self.options["wingbox_pct0"]
        scale = (p0 / p) ** self._e
        g = (1.0 + (lam - 1.0) * self._t) / self._denom
        f = scale * g
        return f, scale * self._t / self._denom, -self._e / p * f

    def compute(self, inputs, outputs):
        lam = inputs["taper_B"][0]
        p = inputs["wingbox_pct"][0]
        p0 = self.options["wingbox_pct0"]

        f, _, _ = self._factors(lam, p)

        # Straight spar: x_le = x_spar - p * c, so the leading edge offset is
        # minus the change in the spar's chordwise setback.
        dx = -self._cx_ref * (p * f - p0)

        outputs["chord"] = f
        outputs["xshear"] = dx - self._r_cx0 * (1.0 - f)
        # np.degrees is not complex-safe, so scale explicitly.
        outputs["sweep_B"] = np.arctan(self._tan_sweep(lam, p)) * _RAD2DEG

        g_w = (1.0 + (lam - 1.0) * self._w_t) / self._width_denom
        # The box edges are fixed chord fractions at each station, so its width
        # follows the local chord only -- it does not move with p, which is the
        # straight line inside the box, not an edge of it.
        c_w = (p0 / p) ** self._e * g_w * self._w_c0
        outputs["station_chord"] = c_w
        outputs["wingbox_width"] = self._box_frac * c_w

    def _tan_sweep(self, lam, p):
        p0 = self.options["wingbox_pct0"]
        return p * (p0 / p) ** self._e * self.options["c_a0"] * (1.0 - lam) / self.options["span_b"]

    def compute_partials(self, inputs, partials):
        lam = inputs["taper_B"][0]
        p = inputs["wingbox_pct"][0]
        p0 = self.options["wingbox_pct0"]

        f, df_dlam, df_dp = self._factors(lam, p)

        # d(dx)/d* from dx = -cx_ref * (p * f - p0)
        ddx_dlam = -self._cx_ref * p * df_dlam
        ddx_dp = -self._cx_ref * (f + p * df_dp)

        partials["chord", "taper_B"] = df_dlam
        partials["xshear", "taper_B"] = ddx_dlam + self._r_cx0 * df_dlam
        partials["xshear", "wingbox_pct"] = ddx_dp + self._r_cx0 * df_dp
        if self._root_le_fixed:
            partials["chord", "wingbox_pct"] = df_dp

        # sweep = atan(T), T = p^(1-e) * p0^e * c_a0 * (1 - lam) / span_b
        tan_s = self._tan_sweep(lam, p)
        d_atan = _RAD2DEG / (1.0 + tan_s**2)
        dT_dlam = -p * (p0 / p) ** self._e * self.options["c_a0"] / self.options["span_b"]
        partials["sweep_B", "taper_B"] = d_atan * dT_dlam

        # width = box_frac * c_w(p) * g_w(lam), with c_w = (p0 / p)^e * width_c0.
        t_w = self._w_t
        g_w = (1.0 + (lam - 1.0) * t_w) / self._width_denom
        c_w = (p0 / p) ** self._e * self._w_c0
        if self._width_varies:
            dc_dlam = c_w * t_w / self._width_denom
            partials["station_chord", "taper_B"] = dc_dlam
            partials["wingbox_width", "taper_B"] = self._box_frac * dc_dlam
        # d/dp [(p0/p)^e] = -e/p * (p0/p)^e
        if self._width_varies_with_p:
            dc_dp = -self._e / p * c_w * g_w
            partials["station_chord", "wingbox_pct"] = dc_dp
            partials["wingbox_width", "wingbox_pct"] = self._box_frac * dc_dp
        if not self._root_le_fixed:
            partials["sweep_B", "wingbox_pct"] = d_atan * tan_s / p


# ---------------------------------------------------------------------------
# Surface dict and geometry group
# ---------------------------------------------------------------------------


def _resample(x_src, y_src, x_dst):
    """Linear interpolation of a native-section quantity onto new coordinates."""
    order = np.argsort(x_src)
    return np.interp(x_dst, np.asarray(x_src)[order], np.asarray(y_src)[order])


def build_surface(mesh, stick, regions, name="wing", n_toc_cp=5, n_twist_cp=None):
    """Assemble the OAS surface dict for one VSP baseline.

    ``t_over_c`` is supplied as ``t_over_c_cp`` and never as a scalar
    ``t_over_c``: the scalar is *not* in the accepted key list in
    ``openaerostruct/utils/check_surface_dict.py``, so OAS silently drops it
    with a RuntimeWarning and the viscous/wave drag model keeps its
    ``np.arange(ny - 1)`` placeholder, i.e. nonsense thickness ratios growing
    along the span. (``openaerostruct/docs/advanced_features/scripts/
    run_vsp_777.py`` has exactly this bug.)

    ``twist_cp`` is carried only to record how many twist control points the
    surface has; the values start at zero because the parsed mesh already has
    the baseline twist baked in and OAS's ``Rotate`` *adds* to what it is given.
    There is no ``chord_cp`` or ``xshear_cp``: those distributions come from
    :class:`RegionPlanform`, not from B-splines.
    """
    if n_twist_cp is None:
        n_twist_cp = config.N_TWIST_CP

    # Normalized span of the native sections, so stick quantities can be
    # resampled onto whatever spanwise distribution the mesh ended up with.
    y_stick = np.abs(np.asarray(stick.le[:, 1], dtype=float))
    s_stick = (y_stick - y_stick[0]) / (y_stick[-1] - y_stick[0])

    y_mesh = np.abs(np.asarray(mesh[0, :, 1], dtype=float))
    s_panel = (y_mesh[:-1] + y_mesh[1:]) / 2.0
    s_panel = (s_panel - y_mesh[0]) / (y_mesh[-1] - y_mesh[0])

    # t_over_c control points, uniform in normalized span. The SplineComp
    # interpolates onto panel midpoints, so sample the panel values.
    toc_panel = _resample(s_stick, stick.toc, s_panel)
    t_over_c_cp = _resample(s_panel, toc_panel, np.linspace(0.0, 1.0, n_toc_cp))

    return {
        "name": name,
        "symmetry": True,
        "S_ref_type": "projected",
        "mesh": mesh,
        "ref_axis_pos": 0.25,
        "twist_cp": np.zeros(n_twist_cp),
        "CL0": 0.0,
        "CD0": 0.0,
        "k_lam": config.K_LAM,
        "t_over_c_cp": t_over_c_cp,
        "c_max_t": float(np.mean(stick.tLoc)),
        "with_viscous": config.WITH_VISCOUS,
        "with_wave": config.WITH_WAVE,
    }


def twist_cp_bounds(mesh, n_cp, bounds=None):
    """Bounds on the *incremental* twist control points.

    The parsed mesh already carries the baseline twist and OAS's ``Rotate`` adds
    to what is there, so ``twist_cp`` starts at zero and means "change in
    twist". The user's limits are absolute, so they have to be shifted by the
    baseline twist at each control point's spanwise station.

    These are a first-order guard only; the exact absolute limits are imposed as
    a constraint on ``twist_abs``, which :class:`RegionGeometry` exposes.
    """
    if bounds is None:
        bounds = config.TWIST_BOUNDS

    tw = baseline_twist(mesh)
    s = np.linspace(0.0, 1.0, tw.size)
    tw_cp = np.interp(np.linspace(0.0, 1.0, n_cp), s, tw)
    return bounds[0] - tw_cp, bounds[1] - tw_cp


class MeshTransforms(om.Group):
    """The subset of OAS's ``GeometryMesh`` chain that is not the identity here.

    ``GeometryMesh`` chains nine transforms -- taper, scale_x, sweep, shear_x,
    stretch, shear_y, dihedral, shear_z, rotate. With no ``taper`` / ``sweep`` /
    ``span`` / ``dihedral`` keys and no y/z shears, six of those are no-ops, so
    this group keeps only ``ScaleX`` -> ``ShearX`` -> ``Rotate``, in the same
    order. That is a third of the components and none of the behaviour, plus two
    fixes that matter for this geometry:

    1. ``Rotate`` is instantiated with ``rotate_x=False``. ``GeometryMesh`` sets
       a local ``self.rotate_x = True`` and then never passes it, so ``Rotate``
       silently keeps its ``rotate_x=True`` default. With that on, ``Rotate`` is
       **not** the identity at zero twist: it applies ``Rx(theta_x)`` about the
       reference axis, where ``theta_x`` is read off the local dihedral. On a
       flat wing that is harmless, but our winglet runs to 42 degrees of
       dihedral, and its sections are already lofted in the winglet's own plane,
       so the extra rotation tilts every one of them a second time. Measured on
       the synthetic winglet this moves the baseline mesh by ~4 mm at zero
       twist, which would put a floor under the round-trip error and bias every
       drag number. Turning it off makes the baseline design vector an exact
       identity. The cost is that twist on the winglet is then applied about the
       global y axis rather than perpendicular to the surface; that is the
       better trade here, since a reproducible baseline is worth more than the
       twist axis of the outer 5% of span.

    2. ``Stretch`` is dropped rather than fed its own span back. Re-normalizing
       ``y`` by a span recomputed from the mesh is only the identity to
       round-off, and it also assumes the reference-axis y is monotonic.
       Dropping it makes the semi-span exactly fixed, which is what was asked
       for anyway.
    """

    def initialize(self):
        self.options.declare("surface", types=dict)

    def setup(self):
        surface = self.options["surface"]
        mesh = surface["mesh"]
        ny = mesh.shape[1]
        ref_axis_pos = surface.get("ref_axis_pos", 0.25)

        self.add_subsystem(
            "scale_x",
            ScaleX(val=np.ones(ny), mesh_shape=mesh.shape, ref_axis_pos=ref_axis_pos),
            promotes_inputs=["chord"],
        )
        self.add_subsystem(
            "shear_x",
            ShearX(val=np.zeros(ny), mesh_shape=mesh.shape),
            promotes_inputs=["xshear"],
        )
        self.add_subsystem(
            "rotate",
            Rotate(
                val=np.zeros(ny),
                mesh_shape=mesh.shape,
                symmetry=surface["symmetry"],
                rotate_x=False,
                ref_axis_pos=ref_axis_pos,
            ),
            promotes_inputs=["twist"],
            promotes_outputs=["mesh"],
        )

        self.set_input_defaults("scale_x.in_mesh", val=mesh, units="m")
        self.connect("scale_x.mesh", "shear_x.in_mesh")
        self.connect("shear_x.mesh", "rotate.in_mesh")


class RegionGeometry(om.Group):
    """Geometry group: planform DVs + twist B-spline -> deformed mesh.

    Stands in for ``openaerostruct.geometry.geometry_group.Geometry``, which
    would insist on driving ``chord`` and ``xshear`` from B-splines.

    Inputs (promoted): ``taper_B``, ``wingbox_pct``, ``twist_cp``,
    ``t_over_c_cp``.
    Outputs (promoted): ``mesh``, ``t_over_c``, ``twist_abs``, ``sweep_B``,
    ``station_chord``, ``wingbox_width``. The last two are vectors over the
    ``width_stations``.
    """

    def initialize(self):
        self.options.declare("surface", types=dict)
        self.options.declare("regions")
        self.options.declare("planform0", types=dict)
        self.options.declare("scale", default=config.SCALE)
        self.options.declare(
            "width_stations",
            default=None,
            desc="Spanwise stations where the box width is evaluated, inches. "
            "Defaults to the y of config.WINGBOX_WIDTH_STATIONS.",
        )
        self.options.declare(
            "rear_schedule",
            default=None,
            desc="Rear-spar (y_in, chord fraction) breakpoints; defaults to config.WINGBOX_REAR_SCHEDULE.",
        )

    def setup(self):
        surface = self.options["surface"]
        mesh = surface["mesh"]
        ny = mesh.shape[1]
        scale = self.options["scale"]
        p0 = self.options["planform0"]

        n_cp = len(surface["twist_cp"])
        x_interp = get_normalized_span_coords(surface)
        twist_comp = self.add_subsystem(
            "twist_bsp",
            om.SplineComp(
                method="bsplines",
                x_interp_val=x_interp,
                num_cp=n_cp,
                interp_options={"order": min(n_cp, 4)},
            ),
            promotes_inputs=["twist_cp"],
            promotes_outputs=["twist"],
        )
        twist_comp.add_spline(y_cp_name="twist_cp", y_interp_name="twist", y_units="deg")
        self.set_input_defaults("twist_cp", val=surface["twist_cp"], units="deg")

        # Absolute twist, so the user's absolute bounds can be constrained exactly.
        self.add_subsystem(
            "twist_abs_comp",
            om.ExecComp(
                "twist_abs = twist + twist_base",
                twist_abs={"shape": (ny,), "units": "deg"},
                twist={"shape": (ny,), "units": "deg"},
                twist_base={"shape": (ny,), "units": "deg", "val": baseline_twist(mesh)},
            ),
            promotes_inputs=["twist"],
            promotes_outputs=["twist_abs"],
        )

        n_toc = len(surface["t_over_c_cp"])
        toc_comp = self.add_subsystem(
            "t_over_c_bsp",
            om.SplineComp(
                method="bsplines",
                x_interp_val=get_normalized_span_coords(surface, mid_panel=True),
                num_cp=n_toc,
                interp_options={"order": min(n_toc, 4), "x_cp_start": 0.0, "x_cp_end": 1.0},
            ),
            promotes_inputs=["t_over_c_cp"],
            promotes_outputs=["t_over_c"],
        )
        toc_comp.add_spline(y_cp_name="t_over_c_cp", y_interp_name="t_over_c")
        self.set_input_defaults("t_over_c_cp", val=surface["t_over_c_cp"])

        t, span_b = _region_coords(mesh, self.options["regions"], scale)
        cx0 = np.asarray(mesh[-1, :, 0] - mesh[0, :, 0], dtype=float)

        # Where the wingbox width is checked, and how wide the box is there. The
        # rear spar is a schedule in *inches* of true span, so it is sampled on
        # the requested stations directly rather than on the region coordinate.
        stations = self.options["width_stations"]
        if stations is None:
            stations = [y for y, _ in config.WINGBOX_WIDTH_STATIONS]
        y_stations = np.atleast_1d(np.asarray(stations, dtype=float))

        y_abs = np.abs(np.asarray(mesh[0, :, 1], dtype=float))
        y_w = y_stations * scale
        width_t = np.interp(y_w, y_abs, t)
        width_c0 = np.interp(y_w, y_abs, cx0)
        rear_pct = rear_spar_fraction(y_stations, self.options["rear_schedule"])

        self.add_subsystem(
            "planform",
            RegionPlanform(
                t=t,
                cx0=cx0,
                span_b=span_b,
                c_a0=p0["c_a0"] * scale,
                taper_B0=p0["taper_B"],
                wingbox_pct0=p0["wingbox_pct"],
                rule=p0["rule"],
                width_t=width_t,
                width_c0=width_c0,
                front_pct=config.WINGBOX_FRONT_PCT,
                rear_pct=rear_pct,
                ref_axis_pos=surface.get("ref_axis_pos", 0.25),
            ),
            promotes_inputs=["taper_B", "wingbox_pct"],
            promotes_outputs=["chord", "xshear", "sweep_B", "station_chord", "wingbox_width"],
        )

        self.add_subsystem(
            "mesh",
            MeshTransforms(surface=surface),
            promotes_inputs=["chord", "xshear", "twist"],
            promotes_outputs=["mesh"],
        )


def build_geometry_group(surface, regions, planform0, scale=config.SCALE, **kwargs):
    """Build the geometry group for a surface. See :class:`RegionGeometry`."""
    return RegionGeometry(surface=surface, regions=regions, planform0=planform0, scale=scale, **kwargs)
