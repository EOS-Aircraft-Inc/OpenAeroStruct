"""WingCalc's wing weight build-up, computed from OpenAeroStruct geometry.

WHY THIS EXISTS
---------------
Until this module the OAS weight was ``F_BOX * W_OAS_box + W_FIXED_LB``: the sized
box, plus 3,437 lb of ribs, fasteners, reinforcements and secondary structure held
constant from ONE WingCalc run. That was correct only while the planform was frozen,
and it made WingCalc a hidden input to every OAS answer. Here the whole build-up is
recomputed from the OAS geometry, by WingCalc's own rules, so OAS stands alone.

WHAT KEPT WINGCALC OUT OF THE GRADIENT LOOP WAS NOT THIS
-------------------------------------------------------
The non-smooth part of WingCalc is its differential-evolution BAY SIZER. None of the
build-up comes from it. Ribs are a smeared plate over the enclosed cell; fasteners
are a count off developed lengths and rib perimeters; the Torenbeek terms are closed
form in areas and aircraft weight; ply drops and paint are fractions. Every one is
algebra on geometry, and ports as algebra.

SOURCE, TERM FOR TERM
---------------------
Structure follows ``Structures-WingCalc_Tool/weight/weight_calc.py:compute_wing_weight``
(tool version 0.63.0), and each function below names the WingCalc function it ports.
The ORDER of the sum matters and is kept: ply drops are a fraction of the box with
everything else already in it, and paint is a fraction of the wing with everything
else in it. Constants are copied, not imported, for the same reason the report
frontend is vendored: the OAS study must be able to compute a weight with the
WingCalc checkout absent. Drift is caught by ``tests/vsp_planform_tests/test_weight.py``,
which pins this module against a WingCalc run term by term.

WHAT CANNOT PORT, AND IS SAID SO
--------------------------------
* ``W_bays`` -- skins, stringers, spar caps, webs -- is the SIZED box. That is what
  OAS's FEM replaces, so it comes in as an input, never from here.
* The nacelle spar-web pad "brought up to its own cap" (``_spar_web_pad_up``'s
  ``match_plies``) needs WingCalc's sized CAP and WEB ply counts. OAS's wingbox has
  one smeared spar thickness and no cap at all, so the rule has nothing to act on.
  It is an explicit input, 0 by default, and the result reports that it was not
  computed. At the Arc A datum WingCalc puts 34.1 lb there (0.5% of the wing).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants -- copied from WingCalc 0.63.0, grouped by the module they come from
# ---------------------------------------------------------------------------
K_MISC = 1.04                                   # weight_results.K_MISCELLANEOUS
PLY_DROP_CLIP_FRACTION = 0.02                   # weight_nonoptimum
PAINT_WING_WEIGHT_FRACTION = 0.02               # weight_secondary
COPPER_MESH_AREAL_DENSITY_LB_FT2 = 0.035        # weight_secondary
RIB_MATERIAL = "Aluminum"                       # weight_wingbox.RIB_MATERIAL

# weight_wingbox.SMEARED_THICKNESS_IN
SMEARED_THICKNESS_IN = {"BL0": 0.20, "W2F_ATTACH": 0.20, "NACELLE_IB": 0.20,
                        "NACELLE_OB": 0.20, "TIP": 0.15, "LIGHT": 0.15}
WS_TOL_IN = 1e-3                                # geometry/topology._WS_TOL_IN

# weight_nonoptimum: 1/4 in countersunk Ti pins on a 5D pitch
_FASTENER_D = 0.2495
_FASTENER_PITCH = 5.0 * _FASTENER_D
_SPAR_CAP_ROWS = 4
_PERIPHERAL_ROWS = 2

# weight_secondary: Torenbeek Ch. 11 reference values (SI) and Table 11.1 credits
_QREF, _WREF, _BREF, _LREF, _OMEGA_REF, _S_REF_M2 = 30_000.0, 1_000_000.0, 50.0, 5.0, 56.0, 10.0
_TE_INC_FOWLER, _TE_INC_DOUBLE_FOWLER = 40.0, 100.0
_REDUCTION = {"fixed_edge": 0.20, "flaps": 0.15, "ailerons": 0.25, "spoilers": 0.10}
# The fixed LE is Aluminum in WingCalc (FIXED_LE_MATERIAL), so it takes no credit;
# the TE shroud, flaps, ailerons and MFS are "Composite" and do.

# geometry/control_surfaces
_MOVABLE_HINGE_OFFSET = 0.05
_MOVABLE_CG_FRACTION = 1.0 / 3.0

# weight_reinforcement: the baseline nacelle installation
_BASELINE_INTERFACE_LOAD = 15_000.0
_DMU_RIB_VOLUME = {("Inboard", "ib"): 485.0, ("Inboard", "ob"): 457.0,
                   ("Outboard", "ib"): 310.0, ("Outboard", "ob"): 292.0}
_PAD_PLIES_BASELINE = 8
_PAD_SPAN_RIB_PITCHES = 1.5
_OTHER_STRUCTURE_FRACTION = 0.003

# loading/nacelle_Interface.INERTIAL_CONDITIONS -- ultimate crash cases
INERTIAL_CONDITIONS = (("6g down", (0.0, 0.0, -6.0)),
                       ("4.5g forward", (-4.5, 0.0, 0.0)),
                       ("2g side outboard", (0.0, 2.0, 0.0)))

_IN_PER_M = 39.37007874015748
_IN2_PER_FT2 = 144.0
_M2_PER_FT2 = _IN2_PER_FT2 / (_IN_PER_M * _IN_PER_M)
_LBF_PER_N = 0.22480894387096
_N_PER_LBF = 1.0 / _LBF_PER_N
_SI_TO_LB_IN2 = _LBF_PER_N / (_IN_PER_M * _IN_PER_M)
_GC_FT_S2 = 32.17404855643044
_PSF_PER_N_M2 = 1.0 / 47.88025898033584


# ---------------------------------------------------------------------------
# Rib layout -- geometry/topology.compute_rib_stations / classify_rib_station
# ---------------------------------------------------------------------------
def _fill(ws_lo, ws_hi, target):
    if abs(ws_hi - ws_lo) < 1e-9:
        return [ws_lo]
    n = max(1, round((ws_hi - ws_lo) / target))
    step = (ws_hi - ws_lo) / n
    return [ws_lo + i * step for i in range(n + 1)]


def nacelle_ribs(deck):
    """(ib_lo, ib_hi, ob_lo, ob_hi): each nacelle is bracketed by a rib pair."""
    return (deck.inboard_nacelle_Y - 0.5 * deck.rib_pitch_inbrd_nacelle,
            deck.inboard_nacelle_Y + 0.5 * deck.rib_pitch_inbrd_nacelle,
            deck.outboard_nacelle_Y - 0.5 * deck.rib_pitch_outbrd_nacelle,
            deck.outboard_nacelle_Y + 0.5 * deck.rib_pitch_outbrd_nacelle)


def rib_stations(deck) -> list[float]:
    """Every rib, from the deck's placement rules -- NOT from WingCalc's output.

    The rules (root, W2F, the two nacelle pairs, the tip, and even fill at the
    average pitches between them) are what move the ribs when the planform moves,
    so they are what has to be ported. On the Arc A deck they reproduce WingCalc's
    20 stations exactly.
    """
    ib_lo, ib_hi, ob_lo, ob_hi = nacelle_ribs(deck)
    tip, w2f = deck.half_wingbox_span, deck.W2F_BL
    segs = ((0.0, w2f, deck.avg_pitch_inbrd), (w2f, ib_lo, deck.avg_pitch_inbrd),
            (ib_hi, ob_lo, deck.avg_pitch_mid), (ob_hi, tip, deck.avg_pitch_outbrd))
    out: list[float] = []
    for lo, hi, target in segs:
        seg = _fill(lo, hi, target)
        out.extend(seg[1:] if out and abs(out[-1] - seg[0]) < 1e-6 else seg)
    return sorted({0.0, w2f, tip, ib_lo, ib_hi, ob_lo, ob_hi}.union(out))


def classify(ws, deck) -> str:
    ib_lo, ib_hi, ob_lo, ob_hi = nacelle_ribs(deck)
    if abs(ws) <= WS_TOL_IN:
        return "BL0"
    if abs(ws - deck.W2F_BL) <= WS_TOL_IN:
        return "W2F_ATTACH"
    if abs(ws - deck.half_wingbox_span) <= WS_TOL_IN:
        return "TIP"
    if min(abs(ws - ib_lo), abs(ws - ib_hi)) <= WS_TOL_IN:
        return "NACELLE_IB"
    if min(abs(ws - ob_lo), abs(ws - ob_hi)) <= WS_TOL_IN:
        return "NACELLE_OB"
    return "LIGHT"


# ---------------------------------------------------------------------------
# Geometry at the ribs
# ---------------------------------------------------------------------------
@dataclass
class RibGeometry:
    """The wing at its rib stations -- everything the build-up reads. Inches.

    It is the one seam between "where the geometry came from" and "what it
    weighs": :func:`rib_geometry_from_oas` fills it from an OAS mesh, and the test
    suite fills it from WingCalc's own ``crossSectionData.csv`` to check that the
    formulas below reproduce WingCalc when handed WingCalc's geometry.
    """
    ws: np.ndarray
    rib_type: tuple
    chord: np.ndarray           # true section chord, |TE - LE|: twist does not shorten it
    x_le: np.ndarray            # global X of the leading edge
    z_qc: np.ndarray            # chord-line Z at the quarter chord
    z_ea: np.ndarray            # elastic-axis Z, the nacelle CG's vertical datum
    t_c: np.ndarray
    fwd: np.ndarray             # spar chord ratios
    aft: np.ndarray
    spar_h_fwd: np.ndarray      # OML depth at each spar
    spar_h_aft: np.ndarray
    z_up_fwd: np.ndarray        # upper OML above the chord line, at each spar
    z_up_aft: np.ndarray
    box_perimeter: np.ndarray   # closed OML box, for the rib fastener rows
    oml_perimeter: np.ndarray   # whole airfoil, for the lightning mesh
    ea_sweep_deg: np.ndarray    # local elastic-axis sweep at the rib (rib pitch)
    seg_sweep_deg: np.ndarray   # EA sweep of each bay, n-1 (structural span)

    @property
    def box_chord(self):
        return (self.aft - self.fwd) * self.chord


def _arc(x, z):
    return float(np.sum(np.hypot(np.diff(x), np.diff(z))))


def section_quantities(chord, t_c, fwd, aft, cam, thk, x_grid):
    """Spar depths, box and OML perimeters of one scaled section.

    Thickness carries the station's t/c and camber does not -- exactly how
    ``coupling.geometry.export`` builds the contours WingCalc is handed.
    """
    thk = thk * (t_c / float(np.max(thk)))
    up, lo = cam + 0.5 * thk, cam - 0.5 * thk
    xc = x_grid * chord
    u_f, u_a = np.interp(fwd, x_grid, up), np.interp(aft, x_grid, up)
    l_f, l_a = np.interp(fwd, x_grid, lo), np.interp(aft, x_grid, lo)
    inside = (x_grid > fwd) & (x_grid < aft)
    xb = np.r_[fwd, x_grid[inside], aft] * chord
    ub = np.r_[u_f, up[inside], u_a] * chord
    lb = np.r_[l_f, lo[inside], l_a] * chord
    h_f, h_a = (u_f - l_f) * chord, (u_a - l_a) * chord
    return {
        "spar_h_fwd": h_f, "spar_h_aft": h_a,
        "z_up_fwd": u_f * chord, "z_up_aft": u_a * chord,
        "box_perimeter": _arc(xb, ub) + _arc(xb, lb) + h_f + h_a,
        "oml_perimeter": _arc(xc, up * chord) + _arc(xc, lo * chord),
    }


def rib_geometry_from_oas(deck, ws_node, le_in, te_in, toc_panel, aft_at,
                          fwd_ratio, profile_at, x_grid, ea_fraction):
    """Fill :class:`RibGeometry` from an OAS mesh, ROOT -> TIP, inches.

    ``toc_panel`` goes to the nodes the way the export does it, so the t/c the
    build-up sees is the t/c WingCalc is given. ``ea_fraction`` places OAS's own
    elastic axis (the FEM node line) for the nacelle CG's vertical datum.
    """
    ws_r = np.array(rib_stations(deck))
    chord_n = np.linalg.norm(te_in - le_in, axis=1)
    toc_n = np.interp(np.arange(len(ws_node)), np.arange(len(toc_panel)) + 0.5, toc_panel)

    def at(v):
        return np.interp(ws_r, ws_node, v)

    chord, t_c = at(chord_n), at(toc_n)
    x_le, z_le, z_te = at(le_in[:, 0]), at(le_in[:, 2]), at(te_in[:, 2])
    fwd = np.full_like(ws_r, float(fwd_ratio))
    aft = np.array([float(aft_at(w)) for w in ws_r])

    # WingCalc's elastic axis is the mid-spar line (loading._elastic_axis_xc); its
    # sweep sets the developed rib pitch and the structural span.
    aft_n = np.array([float(aft_at(w)) for w in ws_node])
    x_ea_n = le_in[:, 0] + 0.5 * (fwd_ratio + aft_n) * chord_n
    local = np.degrees(np.arctan(np.gradient(x_ea_n, ws_node)))
    x_ea = x_le + 0.5 * (fwd + aft) * chord
    seg = np.degrees(np.arctan(np.diff(x_ea) / np.diff(ws_r)))

    q = [section_quantities(c, t, f, a, *profile_at(w), x_grid)
         for c, t, f, a, w in zip(chord, t_c, fwd, aft, ws_r)]

    return RibGeometry(
        ws=ws_r, rib_type=tuple(classify(w, deck) for w in ws_r),
        chord=chord, x_le=x_le, z_qc=z_le + 0.25 * (z_te - z_le),
        z_ea=z_le + ea_fraction * (z_te - z_le), t_c=t_c, fwd=fwd, aft=aft,
        **{k: np.array([d[k] for d in q]) for k in q[0]},
        ea_sweep_deg=np.interp(ws_r, ws_node, local), seg_sweep_deg=seg,
    )


# ---------------------------------------------------------------------------
# Spanwise lengths -- topology._rib_pitch_at_index, weight_inputs._structural_span_m
# ---------------------------------------------------------------------------
def rib_pitch(g, deck):
    """Developed rib spacing: ``dWS / cos(EA sweep)``, unswept inboard of W2F.

    A stringer runs along the swept axis, so it spans more than dWS between two
    ribs. The tip station has no bay outboard, so it takes the last interval.
    """
    d = np.r_[np.diff(g.ws), g.ws[-1] - g.ws[-2]]
    cos = np.cos(np.radians(g.ea_sweep_deg))
    return np.where(g.ws < deck.W2F_BL, d, np.where(cos > 0.0, d / cos, d))


def structural_span_m(g):
    """Developed span of both wings along the elastic axis, m."""
    dy = np.diff(g.ws)
    cos = np.cos(np.radians(g.seg_sweep_deg))
    return 2.0 * float(np.sum(np.where(cos > 0.0, dy / cos, dy))) / _IN_PER_M


# ---------------------------------------------------------------------------
# Reference areas -- geometry/reference_areas + geometry/control_surfaces
# ---------------------------------------------------------------------------
class _Areas:
    """Trapezoidal chord x chord-fraction integrals, broken at every rib.

    ``stations_for_range`` puts a node at every rib inside a range and at both
    ends, and the chord between ribs is linear in WS -- the same nodes WingCalc
    integrates on, so an edge that follows a moving spar is not averaged away.
    """

    def __init__(self, g, deck):
        self.g, self.deck = g, deck
        self.half = deck.half_wingbox_span

    def chord(self, ws):
        return float(np.interp(ws, self.g.ws, self.g.chord))

    def aft_spar(self, ws):
        return float(np.interp(ws, self.g.ws, self.g.aft))

    def fwd_spar(self, ws):
        return float(np.interp(ws, self.g.ws, self.g.fwd))

    def stations(self, y0, y1):
        y0 = max(0.0, min(float(y0), self.half))
        y1 = max(0.0, min(float(y1), self.half))
        if y1 <= y0:
            return []
        return sorted({y0, y1, *(w for w in self.g.ws if y0 < w < y1)})

    def integrate_ft2(self, y0, y1, ratio):
        st = self.stations(y0, y1)
        if len(st) < 2:
            return 0.0
        w = [self.chord(y) * max(float(ratio(y)), 0.0) for y in st]
        semi = sum(0.5 * (w[i] + w[i + 1]) * (st[i + 1] - st[i]) for i in range(len(st) - 1))
        return 2.0 * semi / _IN2_PER_FT2

    # -- the movables and the shroud, as chordwise edges ------------------------
    def hinge(self, ws):
        return min(max(self.aft_spar(ws) + _MOVABLE_HINGE_OFFSET, 0.0), 1.0)

    def movable_cg(self, ws):
        h = self.hinge(ws)
        return h + _MOVABLE_CG_FRACTION * (1.0 - h)

    def flap(self):
        d = self.deck
        if d.fuse_diameter <= 0.0 or d.flap_span_ratio <= 0.0:
            return None
        y0 = 0.5 * d.fuse_diameter
        y1 = min(y0 + 0.5 * (2.0 * self.half - d.fuse_diameter) * d.flap_span_ratio, self.half)
        return (y0, y1) if y1 > y0 else None

    def aileron(self):
        r = max(self.deck.aileron_span_ratio, 0.0)
        y0 = self.half * max(1.0 - r, 0.0)
        return (y0, self.half) if r > 0.0 and self.half > y0 else None

    def movables(self):
        out = [s for s in (self.flap(), self.aileron()) if s is not None]
        for (a0, a1), (b0, _b1) in zip(out, out[1:]):
            if b0 < a1:
                raise ValueError(f"flap runs to WS {a1:.1f} but the aileron starts at WS "
                                 f"{b0:.1f}: they overlap (planformIn.csv span ratios)")
        return out

    def movable_area(self, span):
        return 0.0 if span is None else self.integrate_ft2(*span, lambda w: 1.0 - self.hinge(w))

    def fixed_te(self, y0, y1):
        area = self.integrate_ft2(y0, y1, lambda w: max(1.0 - self.aft_spar(w), 0.0))
        for s0, s1 in self.movables():
            a, b = max(s0, y0), min(s1, y1)
            if b > a:
                area -= self.integrate_ft2(a, b, lambda w: max(1.0 - self.movable_cg(w), 0.0))
        return max(area, 0.0)

    def compute(self):
        d, exposed = self.deck, min(0.5 * max(self.deck.fuse_diameter, 0.0), self.half)
        flap = self.flap()
        mfs = sum(self.integrate_ft2(min(a, b), max(a, b),
                                     lambda w: max(self.movable_cg(w) - self.aft_spar(w), 0.0))
                  for a, b in d.mfs if max(a, b) > min(a, b))
        tip_chord = self.chord(self.half)
        s_winglet = (2.0 * 0.5 * (tip_chord + d.winglet_tip_chord) * d.winglet_span
                     / _IN2_PER_FT2) if d.winglet_span > 0.0 else 0.0
        s_wing = self.integrate_ft2(0.0, self.half, lambda w: 1.0)
        return {
            "Sref_ft2": s_wing,
            "S_wing_ft2": s_wing + s_winglet,
            "S_le_ft2": self.integrate_ft2(exposed, self.half, lambda w: max(self.fwd_spar(w), 0.0)),
            "S_fixed_te_ft2": self.fixed_te(exposed, self.half),
            "S_fixed_te_flap_aligned_ft2": self.fixed_te(*flap) if flap else 0.0,
            "S_flap_ft2": self.movable_area(flap),
            "S_aileron_ft2": self.movable_area(self.aileron()),
            "S_mfs_ft2": mfs,
            "S_winglet_ft2": s_winglet,
            "flap_ws": flap, "aileron_ws": self.aileron(),
        }


# ---------------------------------------------------------------------------
# Aircraft-level numbers -- weight/weight_inputs
# ---------------------------------------------------------------------------
def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def mtow_lb(deck):
    m = max((_f(r.get("AC_Weight")) for r in deck.load_cases), default=0.0)
    if m <= 0.0:
        raise ValueError("loadCasesIn.csv needs a positive AC_Weight on at least one case")
    return m


def mlw_lb(deck):
    w = [_f(r.get("AC_Weight")) for r in deck.load_cases
         if "landing" in r.get("Load_case", "").lower() and _f(r.get("AC_Weight")) > 0.0]
    return max(w) if w else mtow_lb(deck)


def two_wheel_landing_nz(deck, default=4.0):
    for r in deck.load_cases:
        if r.get("Load_case", "").lower() == "2-wheel landing":
            return _f(r.get("Nz"), default)
    return default


def manoeuvre_ultimate_nz(deck):
    n = max((_f(r.get("Nz_lift")) * _f(r.get("Ult_factor")) for r in deck.load_cases), default=0.0)
    if n <= 0.0:
        raise ValueError("loadCasesIn.csv needs a positive Nz_lift x Ult_factor")
    return n


def design_dynamic_pressure_n_m2(deck):
    rho, vd = deck.wing_loading.get("rho max", 0.0), deck.wing_loading.get("Vd", 0.0)
    if rho <= 0.0 or vd <= 0.0:
        return 0.0
    return (0.5 * rho * vd * vd / _GC_FT_S2) / _PSF_PER_N_M2


# ---------------------------------------------------------------------------
# Fasteners and splices -- weight/weight_nonoptimum
# ---------------------------------------------------------------------------
def fastener_weight(deck):
    """One pin and nut, less the laminate its hole displaces, lb."""
    rho = 4.43 * 0.036127292                  # Ti-6Al-4V, g/cc -> lb/in3
    rho_c = deck.materials["UD QI"].density
    d, Dh, D_nut, H_nut, H, grip = _FASTENER_D, 0.504, 0.425, 0.340, 0.107, 0.25
    P = H_nut + 0.05
    A_eff = 0.975 * math.pi * (d / 2) ** 2
    R, r = Dh / 2, d / 2
    v_head = math.pi * H * (R * R + R * r + r * r) / 3.0
    v_pin = v_head + A_eff * (grip + P - H)
    v_hole = v_head + A_eff * grip
    v_nut = math.pi / 4.0 * (D_nut ** 2 - d ** 2) * H_nut
    return v_pin * rho - v_hole * rho_c + v_nut * rho


def splice_fasteners(g, deck):
    n, length = max(0, deck.wing_splices), max(0.0, deck.splice_length)
    if n == 0 or length <= 0.0 or deck.splice_bay <= 0:
        return 0
    per = float(g.box_perimeter[deck.splice_bay - 1])
    return n * round(length / 1.5) * math.ceil(per / 1.5)


def joints(g, deck, pitch):
    spar_len = 2.0 * float(np.sum(pitch[:-1])) * _SPAR_CAP_ROWS
    rib_len = (float(g.box_perimeter[0]) + 2.0 * float(np.sum(g.box_perimeter[1:]))) * _PERIPHERAL_ROWS
    count = (spar_len + rib_len) / _FASTENER_PITCH + splice_fasteners(g, deck)
    return fastener_weight(deck) * count


def winglet_attach(g, deck):
    if deck.winglet_span <= 0.0:
        return 0.0
    return fastener_weight(deck) * 2.0 * float(g.box_perimeter[-1]) * _PERIPHERAL_ROWS / _FASTENER_PITCH


def splices(deck, splice_w_per_in):
    n, length = max(0, deck.wing_splices), max(0.0, deck.splice_length)
    if n == 0 or length <= 0.0 or deck.splice_bay <= 0:
        return 0.0
    return n * length * float(splice_w_per_in)


# ---------------------------------------------------------------------------
# Nacelle interface -- loading/nacelle_Interface + weight/weight_reinforcement
# ---------------------------------------------------------------------------
def _bolt_group(points, cg, applied):
    """Rigid bolt group: equilibrium in all six DOF (nacelle_Interface._bolt_group_reactions)."""
    p = np.asarray(points, dtype=float)
    c = p.mean(axis=0)
    dx, dy, dz = (p - c).T
    Px, Py, Pz = applied
    rx, ry, rz = np.asarray(cg, dtype=float) - c
    n = len(p)
    a, d, e = -Pz / n, -Px / n, -Py / n
    Ixx, Iyy, Ixy = float(dy @ dy), float(dx @ dx), float(dx @ dy)
    Szx, Szy = float(dz @ dx), float(dz @ dy)
    f = -(rx * Py - ry * Px) / (Ixx + Iyy)
    b, cc = np.linalg.solve(np.array([[Ixy, Ixx], [Iyy, Ixy]]),
                            np.array([f * Szx - ry * Pz + rz * Py,
                                      -f * Szy + rz * Px - rx * Pz]))
    return [(d - f * dyi, e + f * dxi, a + b * dxi + cc * dyi) for dxi, dyi in zip(dx, dy)]


def nacelle_fitting_envelope(g, deck):
    """Worst resultant per fitting over the three crash cases, per nacelle.

    The fittings are the four points where the nacelle ribs cross the spars, at the
    upper skin OML; the CG is ``x/c`` aft of the local LE and ``delta_Z`` above the
    elastic axis. The inboard nacelle is taken WITHOUT its gear here, as WingCalc
    does: the fitting loads are a gear-free condition.
    """
    wl = deck.wl
    ib_lo, ib_hi, ob_lo, ob_hi = nacelle_ribs(deck)
    specs = (("Inboard", (ib_lo, ib_hi), deck.inboard_nacelle_Y,
              wl("Inboard nacelle weight wo mlg"), wl("x/c ncl ib_no_mlg"), wl("delta_Z NCL ib_no_mlg")),
             ("Outboard", (ob_lo, ob_hi), deck.outboard_nacelle_Y,
              wl("Outboard nacelle weight"), wl("x/c ncl ob"), wl("delta_Z NCL ob")))
    out = []
    for name, ribs, y_cg, weight, xc, dz in specs:
        idx = [int(np.argmin(np.abs(g.ws - w))) for w in ribs]
        pts, labels = [], []
        for i, rib in zip(idx, ("ib", "ob")):
            for spar, ratio, zup in (("fwd", g.fwd[i], g.z_up_fwd[i]), ("aft", g.aft[i], g.z_up_aft[i])):
                pts.append((g.x_le[i] + ratio * g.chord[i], g.ws[i], g.z_qc[i] + zup))
                labels.append((spar, rib))
        cg = (np.interp(y_cg, g.ws, g.x_le) + xc * np.interp(y_cg, g.ws, g.chord),
              y_cg, np.interp(y_cg, g.ws, g.z_ea) + dz)
        worst = [(0.0, "") for _ in pts]
        for cond, (nx, ny, nz) in INERTIAL_CONDITIONS:
            for k, F in enumerate(_bolt_group(pts, cg, (nx * weight, ny * weight, nz * weight))):
                r = float(np.linalg.norm(F))
                if r > worst[k][0]:
                    worst[k] = (r, cond)
        out.append({"nacelle": name, "weight": weight, "rib_idx": idx, "cg": cg,
                    "fittings": [{"spar": s, "rib": rb, "load": w[0], "condition": w[1]}
                                 for (s, rb), w in zip(labels, worst)]})
    return out


def _worst(fittings, **keep):
    sel = [f for f in fittings if all(f[k] == v for k, v in keep.items())]
    return max(sel, key=lambda f: f["load"])


def nacelle_reinforcement(g, deck, am, pitch, web_match_plies=None):
    """What each nacelle adds to one wing, over the plain structure beneath it.

    ``web_match_plies`` is ``{(nacelle, spar): plies}``: WingCalc's "bring the web up
    to its own cap" step, which needs sized cap and web ply counts OAS does not have.
    Absent, it is 0 and the result says it was not computed.
    """
    rho_rib = deck.materials[RIB_MATERIAL].density
    skin, web = deck.materials[deck.skin_material], deck.materials[deck.web_material]
    rows = []
    for nac in nacelle_fitting_envelope(g, deck):
        name, (i_ib, i_ob), fits = nac["nacelle"], nac["rib_idx"], nac["fittings"]
        ribs = []
        for rib, i in (("ib", i_ib), ("ob", i_ob)):
            w = _worst(fits, rib=rib)
            v = _DMU_RIB_VOLUME[(name, rib)] * w["load"] / _BASELINE_INTERFACE_LOAD
            t_base = SMEARED_THICKNESS_IN[g.rib_type[i]]
            ribs.append({"rib": rib, "ws": float(g.ws[i]), "load": w["load"],
                         "condition": w["condition"], "V_scaled": v, "Am": float(am[i]),
                         "delta": (v - float(am[i]) * t_base) * rho_rib})
        span = _PAD_SPAN_RIB_PITCHES * float(pitch[i_ib])
        pads = []
        w_all = _worst(fits)
        n_sk = math.ceil(_PAD_PLIES_BASELINE * w_all["load"] / _BASELINE_INTERFACE_LOAD)
        pads.append({"panel": "Upper skin", "load": w_all["load"], "condition": w_all["condition"],
                     "match_plies": 0, "pad_plies": n_sk, "width": float(g.box_chord[i_ib]),
                     "span": span, "weight": n_sk * skin.ply_t * span * float(g.box_chord[i_ib]) * skin.density})
        for spar, h in (("fwd", g.spar_h_fwd), ("aft", g.spar_h_aft)):
            w = _worst(fits, spar=spar)
            n_pad = math.ceil(_PAD_PLIES_BASELINE * w["load"] / _BASELINE_INTERFACE_LOAD)
            match = 0 if web_match_plies is None else int(web_match_plies[(name, spar)])
            width = 0.5 * (float(h[i_ib]) + float(h[i_ob]))
            pads.append({"panel": f"{spar.capitalize()} spar web", "load": w["load"],
                         "condition": w["condition"], "match_plies": match, "pad_plies": n_pad,
                         "width": width, "span": span,
                         "weight": (match + n_pad) * web.ply_t * span * width * web.density})
        other = _OTHER_STRUCTURE_FRACTION * nac["weight"]
        rib_w, pad_w = sum(r["delta"] for r in ribs), sum(p["weight"] for p in pads)
        rows.append({"nacelle": name, "ribs": ribs, "pads": pads, "other": other,
                     "rib_weight": rib_w, "pad_weight": pad_w,
                     "total": rib_w + pad_w + other, "cg": nac["cg"]})
    return rows


# ---------------------------------------------------------------------------
# Secondary structure -- weight/weight_secondary (Torenbeek Ch. 11)
# ---------------------------------------------------------------------------
def secondary(g, deck, areas, bst_m):
    mtow_n = mtow_lb(deck) * _N_PER_LBF
    qd = design_dynamic_pressure_n_m2(deck)
    if qd <= 0.0:
        raise ValueError("wingLoadingIn.csv requires positive 'rho max' and 'Vd'")
    span_145 = (mtow_n * bst_m / (_WREF * _BREF)) ** 0.145
    out = {}
    # Eq 11.63 -- fixed LE, Aluminum, no composite credit
    out["W_leading_edge_fixed"] = (3.15 * _OMEGA_REF * (qd / _QREF) ** 0.25 * span_145
                                   * _SI_TO_LB_IN2 * areas["S_le_ft2"] * _IN2_PER_FT2)
    out["W_leading_edge_high_lift"] = 0.0
    # Eq 11.65 -- fixed TE shroud, composite
    ksup, kslot = deck.flap_Ksup, deck.flap_Kslot
    inc = (_TE_INC_DOUBLE_FOWLER if (ksup >= 1.6 and kslot >= 1.5)
           else _TE_INC_FOWLER if (ksup >= 1.6 or kslot >= 1.5) else 0.0)
    om_base = 2.6 * _OMEGA_REF * (mtow_n * bst_m / (_WREF * _BREF)) ** 0.0544 * _SI_TO_LB_IN2
    w_te = (om_base * areas["S_fixed_te_ft2"] + inc * _SI_TO_LB_IN2 * areas["S_fixed_te_flap_aligned_ft2"]) * _IN2_PER_FT2
    out["W_fixed_trailing_edge"] = w_te * (1.0 - _REDUCTION["fixed_edge"])
    # Eq 11.66 -- flaps
    om = 1.7 * ksup * kslot * _OMEGA_REF * (1.0 + (mtow_n / _WREF) ** 0.35) * _SI_TO_LB_IN2
    out["W_trailing_edge_flaps"] = om * areas["S_flap_ft2"] * _IN2_PER_FT2 * (1.0 - _REDUCTION["flaps"])
    # Eq 11.67 -- ailerons
    s = areas["S_aileron_ft2"]
    out["W_aileron"] = 0.0 if s <= 0.0 else (
        3.0 * deck.aileron_Kbal * _OMEGA_REF * (s * _M2_PER_FT2 / _S_REF_M2) ** 0.044
        * _SI_TO_LB_IN2 * s * _IN2_PER_FT2 * (1.0 - _REDUCTION["ailerons"]))
    # Eq 11.68 -- spoilers / MFS
    s = areas["S_mfs_ft2"]
    out["W_mfs"] = 0.0 if s <= 0.0 else (
        2.2 * _OMEGA_REF * (s * _M2_PER_FT2 / _S_REF_M2) ** 0.032
        * _SI_TO_LB_IN2 * s * _IN2_PER_FT2 * (1.0 - _REDUCTION["spoilers"]))
    # Eq 11.70 -- wingtip. l_tip is the winglet span x 1.5 IN INCHES, against
    # Torenbeek's l_ref of 5 m: WingCalc's own units, ported as they are (see the
    # study README -- this overstates the term, and the port must match first).
    l_tip = deck.winglet_span * 1.5
    s = areas["S_winglet_ft2"]
    out["W_wingtip"] = 0.0 if (l_tip <= 0.0 or s <= 0.0) else (
        2.5 * _OMEGA_REF * (mtow_n * l_tip / (_WREF * _LREF)) ** 0.145 * _SI_TO_LB_IN2 * s * _IN2_PER_FT2)
    # Lightning mesh over the painted surface, from the fuselage side outboard
    y_root = 0.5 * max(deck.fuse_diameter, 0.0)
    semi = 0.0
    for i in range(len(g.ws) - 1):
        w0, w1 = float(g.ws[i]), float(g.ws[i + 1])
        lo = max(y_root, w0)
        if w1 <= lo:
            continue
        p0, p1 = float(g.oml_perimeter[i]), float(g.oml_perimeter[i + 1])
        p_lo = p0 + (p1 - p0) * (lo - w0) / (w1 - w0)
        semi += 0.5 * (p_lo + p1) * (w1 - lo)
    painted_ft2 = 2.0 * semi / _IN2_PER_FT2 + 2.0 * areas["S_winglet_ft2"]
    out["W_copper_mesh"] = painted_ft2 * COPPER_MESH_AREAL_DENSITY_LB_FT2
    out["_painted_area_ft2"] = painted_ft2
    return out


# ---------------------------------------------------------------------------
# The sum -- weight/weight_calc.compute_wing_weight
# ---------------------------------------------------------------------------
BOX_TERMS = ("W_bays_full", "W_ribs_full", "W_joints", "W_winglet_attachment", "W_splices",
             "W_fail_safety_damage_tolerance", "W_nacelle_reinforcements",
             "W_landing_gear_reinforcements", "W_wing_fuselage_attach",
             "W_dynamic_over_swing", "W_torsional_stiffness", "W_ply_drops_clips")
SECONDARY_TERMS = ("W_leading_edge_fixed", "W_leading_edge_high_lift", "W_fixed_trailing_edge",
                   "W_trailing_edge_flaps", "W_aileron", "W_mfs", "W_wingtip",
                   "W_copper_mesh", "W_paint")


def wing_weight(g, deck, W_bays_full, am, splice_w_per_in, web_match_plies=None):
    """Full-wing weight by WingCalc's build-up, lb. Every term is returned.

    ``W_bays_full`` is the sized box (both wings), ``am`` the enclosed mid-line cell
    area at each rib (in2), ``splice_w_per_in`` the box running weight at the splice
    bay (lb/in). Those three are the only places the SIZED structure enters; all of
    it is linear, which is what lets the OpenMDAO wrapper give exact partials.
    """
    am = np.asarray(am, dtype=float)
    pitch = rib_pitch(g, deck)
    bst = structural_span_m(g)
    areas = _Areas(g, deck).compute()
    rho_rib = deck.materials[RIB_MATERIAL].density
    t = np.array([SMEARED_THICKNESS_IN[r] for r in g.rib_type])
    w_rib = am * t * rho_rib

    r = {"W_bays_full": float(W_bays_full)}
    r["W_ribs_full"] = float(w_rib[0] + 2.0 * np.sum(w_rib[1:]))
    r["W_wingbox_basic"] = r["W_bays_full"] + r["W_ribs_full"]
    r["W_joints"] = joints(g, deck, pitch)
    r["W_winglet_attachment"] = winglet_attach(g, deck)
    r["W_splices"] = splices(deck, splice_w_per_in)
    r["W_fail_safety_damage_tolerance"] = 0.0
    nac = nacelle_reinforcement(g, deck, am, pitch, web_match_plies)
    r["W_nacelle_reinforcements"] = 2.0 * sum(n["total"] for n in nac)
    r["W_landing_gear_reinforcements"] = 0.0015 * two_wheel_landing_nz(deck) * mlw_lb(deck)
    r["W_reinforcements"] = r["W_nacelle_reinforcements"] + r["W_landing_gear_reinforcements"]
    r["W_wing_fuselage_attach"] = 0.0003 * manoeuvre_ultimate_nz(deck) * mtow_lb(deck)
    r["W_dynamic_over_swing"] = 0.0
    r["W_torsional_stiffness"] = 0.0
    before_layup = (r["W_wingbox_basic"] + r["W_joints"] + r["W_winglet_attachment"]
                    + r["W_splices"] + r["W_fail_safety_damage_tolerance"]
                    + r["W_reinforcements"] + r["W_wing_fuselage_attach"]
                    + r["W_dynamic_over_swing"] + r["W_torsional_stiffness"])
    r["W_ply_drops_clips"] = PLY_DROP_CLIP_FRACTION * max(before_layup, 0.0)
    r["W_wingbox"] = before_layup + r["W_ply_drops_clips"]

    sec = secondary(g, deck, areas, bst)
    painted = sec.pop("_painted_area_ft2")
    r.update(sec)
    sec_before = sum(r[k] for k in SECONDARY_TERMS if k != "W_paint")
    r["W_paint"] = PAINT_WING_WEIGHT_FRACTION * max(r["W_wingbox"] + K_MISC * sec_before, 0.0)
    r["W_miscellaneous"] = r["W_paint"] + r["W_copper_mesh"]
    r["W_secondary"] = sec_before + r["W_paint"]
    r["k_misc"] = K_MISC
    r["W_wing"] = r["W_wingbox"] + K_MISC * r["W_secondary"]

    r["detail"] = {
        "reference_areas": areas, "structural_span_m": bst, "painted_area_ft2": painted,
        "nacelle_reinforcement": nac,
        "web_match_to_cap": "input" if web_match_plies is not None else "not computed (OAS has no spar caps)",
        "ribs": [{"ws": float(w), "rib_type": rt, "pitch": float(p), "Am": float(a),
                  "t_smeared": float(ti), "W_rib": float(wr)}
                 for w, rt, p, a, ti, wr in zip(g.ws, g.rib_type, pitch, am, t, w_rib)],
    }
    return r


def dW_wing_dW_box():
    """d(W_wing)/d(any box-linear term), exact: 1.02 from ply drops, then paint and k_misc.

    Every term that scales with the sized box enters ``W_wingbox`` before the lay-up
    allowance, so it is multiplied by ``1 + PLY_DROP_CLIP_FRACTION``; ``W_wing`` is then
    ``(1 + PAINT * K_MISC) * (W_wingbox + K_MISC * sec_before)``. Stated once, here,
    because the OpenMDAO component's analytic partials are built on it.
    """
    return (1.0 + PLY_DROP_CLIP_FRACTION) * (1.0 + PAINT_WING_WEIGHT_FRACTION * K_MISC)
