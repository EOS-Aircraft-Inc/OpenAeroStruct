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

__all__ = [
    "RegionPlanform",
    "REGION_A_RULE",
    "baseline_planform",
    "baseline_twist",
    "baseline_wingbox_pct",
    "build_geometry_group",
    "build_surface",
    "mac_quarter_chord",
    "twist_cp_bounds",
]

# Per-geometry rule for what happens to region A's chord. See the module
# docstring; these are user-mandated and deliberately not unified.
REGION_A_RULE = {
    "const_chord": "preserved",
    "plan_l": "root_le_fixed",
}

# Inboard wingbox width requirement for Plan_L: at least 65 in of box, from the
# root out to y = 100 in. The chord falls monotonically outboard, so only the
# outboard end of that run can bind.
WINGBOX_WIDTH_MIN_IN = 65.0
WINGBOX_WIDTH_STATION_IN = 100.0

_RULE_EXPONENT = {"preserved": 0.0, "root_le_fixed": 1.0}

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
    wingbox_width : float
        Wingbox width ``(p - front_pct) * c`` at the constraint station, metres:
        the spar-to-spar extent of the structural box.
    """

    def initialize(self):
        self.options.declare("t", types=np.ndarray, desc="Region coordinate per station: 0 in A, 0->1 in B, 1 in C.")
        self.options.declare("cx0", types=np.ndarray, desc="Baseline chordwise x-extent per station, metres.")
        self.options.declare("span_b", types=float, desc="Spanwise extent of region B, metres.")
        self.options.declare("c_a0", types=float, desc="Baseline chordwise x-extent of region A, metres.")
        self.options.declare("taper_B0", types=float, desc="Baseline taper ratio of region B.")
        self.options.declare("wingbox_pct0", types=float, desc="Baseline spar chord fraction.")
        self.options.declare("rule", values=("preserved", "root_le_fixed"), desc="What holds region A's chord.")
        self.options.declare("width_t", default=0.0, desc="Region coordinate of the wingbox-width station.")
        self.options.declare("width_c0", default=1.0, desc="Baseline chord at the wingbox-width station, metres.")
        self.options.declare("front_pct", default=0.0, desc="Front-spar location as a fraction of chord.")
        self.options.declare("ref_axis_pos", default=0.25, desc="Fraction of chord used as the reference axis.")

    def setup(self):
        t = self.options["t"]
        cx0 = self.options["cx0"]

        self.add_input("taper_B", val=self.options["taper_B0"])
        self.add_input("wingbox_pct", val=self.options["wingbox_pct0"])

        self.add_output("chord", val=np.ones(t.size))
        self.add_output("xshear", val=np.zeros(t.size), units="m")
        self.add_output("sweep_B", val=0.0, units="deg")
        self.add_output("wingbox_width", val=0.0, units="m")

        # Which outputs actually respond to the spar fraction depends on the
        # region A rule, so declare only the structurally nonzero blocks.
        # "root_le_fixed" makes p * c invariant, which freezes the leading-edge
        # line, region B's sweep and the wingbox width; p then only scales chords.
        # "preserved" is the mirror image: p sweeps region B but leaves the chord
        # distribution alone.
        rule = self.options["rule"]
        self._root_le_fixed = rule == "root_le_fixed"
        self._width_varies = self.options["width_t"] != 0.0

        self.declare_partials(["chord", "xshear", "sweep_B"], "taper_B")
        self.declare_partials("xshear", "wingbox_pct")
        if self._root_le_fixed:
            self.declare_partials("chord", "wingbox_pct")
        else:
            self.declare_partials("sweep_B", "wingbox_pct")
        # Under "root_le_fixed" the product p * c is invariant, so a box measured
        # from the leading edge would not respond to p at all. A box that starts
        # at a fixed *front spar* does respond, because only its rear edge moves
        # with p -- but with front_pct = 0 the two definitions coincide and the
        # derivative really is zero, so only declare it where it can be nonzero.
        self._width_varies_with_p = (not self._root_le_fixed) or self.options["front_pct"] != 0.0
        if self._width_varies_with_p:
            self.declare_partials("wingbox_width", "wingbox_pct")
        if self._width_varies:
            self.declare_partials("wingbox_width", "taper_B")

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

        self._width_denom = 1.0 + (self.options["taper_B0"] - 1.0) * self.options["width_t"]

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

        g_w = (1.0 + (lam - 1.0) * self.options["width_t"]) / self._width_denom
        # Box width is spar to spar: (rear - front) * chord at the station.
        width = p - self.options["front_pct"]
        outputs["wingbox_width"] = width * (p0 / p) ** self._e * g_w * self.options["width_c0"]

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

        # width = (p - front) * c_w(p) * g_w(lam), with c_w = (p0 / p)^e * width_c0.
        t_w = self.options["width_t"]
        g_w = (1.0 + (lam - 1.0) * t_w) / self._width_denom
        c_w = (p0 / p) ** self._e * self.options["width_c0"]
        width = p - self.options["front_pct"]
        if self._width_varies:
            partials["wingbox_width", "taper_B"] = width * c_w * t_w / self._width_denom
        # d/dp [(p - front) * (p0/p)^e] = (p0/p)^e * (1 - e * (p - front) / p)
        if self._width_varies_with_p:
            partials["wingbox_width", "wingbox_pct"] = g_w * c_w * (1.0 - self._e * width / p)
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
    ``wingbox_width``.
    """

    def initialize(self):
        self.options.declare("surface", types=dict)
        self.options.declare("regions")
        self.options.declare("planform0", types=dict)
        self.options.declare("scale", default=config.SCALE)
        self.options.declare("width_station", default=WINGBOX_WIDTH_STATION_IN, desc="Width constraint station, in.")

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

        # Where the inboard wingbox width is checked.
        y_abs = np.abs(np.asarray(mesh[0, :, 1], dtype=float))
        y_w = self.options["width_station"] * scale
        width_t = float(np.interp(y_w, y_abs, t))
        width_c0 = float(np.interp(y_w, y_abs, cx0))

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
                ref_axis_pos=surface.get("ref_axis_pos", 0.25),
            ),
            promotes_inputs=["taper_B", "wingbox_pct"],
            promotes_outputs=["chord", "xshear", "sweep_B", "wingbox_width"],
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
