"""Apply the swept-optimum t/c profile to all three architectures, and size them.

The 24-point level x shape sweep (coupled_toc_profile.py) put the electric-range
optimum at root t/c 0.250 tapering to a 0.145 tip -- NOT the constant t/c that the
thin-root row suggested, and NOT the 22 -> 10 the crossover argument implied. The
band is flat: root 0.21-0.25 at ratios 0.45-0.75 all sit within 0.7% of the peak,
so a second profile is run at root 0.220 / ratio 0.75, which stays inside
conventional thickness and inside the Raymer form factor's calibration.

Applying a t/c profile is not just setting control points. The 6 in aileron depth
requirement is imposed as a box WIDTH at a chord c_req = 6/(retention * t/c), so
thickening the wing changes its own constraint. Taking c_req from the baseline
loft, as the study did, delivers 5.71 in rather than 6.00; here the requirement is
re-derived from the t/c the model actually delivers and iterated to convergence,
as depth_feasibility.py established.

Each design is then sized by WingCalc through a damped weight fixed point, so the
reported range carries a real structural weight rather than a seed.

  arc A  constant chord     region A at the loft's own 361.7 in breakpoint
  arc B  straight fwd spar  straight line pinned at 0.12c, `preserved` rule
  arc C  free               A|B re-lofted to 176 in, straight line free

Writes out/logs/arc_optimal_toc_<arc>_<profile>.json; merge with
merge_arc_toc.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

from studies.vsp_planform import config, param                  # noqa: E402
import studies.vsp_planform.run_opt as ro                       # noqa: E402
from studies.vsp_planform.run_opt import POINT, trim_alpha      # noqa: E402
from studies.vsp_planform.param import baseline_planform, rear_spar_fraction  # noqa: E402
from studies.vsp_planform.degen_csv import read_degen_csv, lifting_surfaces   # noqa: E402
from studies.vsp_planform.coupling import deck as wcdeck        # noqa: E402
from studies.vsp_planform.coupling import mission               # noqa: E402
import wing2_oas as w2                                          # noqa: E402
from doe_v3 import asbuilt                                      # noqa: E402
from wing8_constchord_toc import REGION_A_AS_BUILT_IN           # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")
q = 0.5 * config.RHO * config.V_MS**2
SEMI_IN, Y_AIL = 708.0, 0.90 * 708.0
# The aileron actuator depth floor. Raised 6 -> 7 in (user, 2026-08-28). With e694
# and a thick root the arcs deliver 7.65-8.83 in at a 0.574c spar, so 7 in still
# does not bind there -- but Arc A's straight aft spar sits at 0.750c, where the
# section keeps only 0.640 of its thickness, so on Arc A it binds hard.
DEPTH_REQ = 7.0
SCHEDULE = ((356.0, 0.750), (674.9, 0.550))
# Arc A carries its own: a CONSTANT 0.750c rear spar, so the aft spar is a straight
# line root to junction. It has to be its own schedule, because x_aft = x_le +
# rear(y)*chord(y) kinks wherever the ratio moves, and the shared schedule moves it
# from exactly y=356 -- the outboard nacelle, the point past which Arc A is meant
# to hold its aft spar straight. Arc A's forward spar is left to kink there instead
# (it sits at a fixed 0.12c, so it follows the chord breakpoint), which is the
# trade this architecture is making: one straight spar, and it is the aft one.
SCHEDULE_A = ((356.0, 0.750), (674.9, 0.750))
TOL_IN, MAX_PASS = 0.005, 5
W_SEED_LB, W_TOL_LB = 8000.0, 25.0
# The weight loop is a damped fixed point (w += 0.5 * residual) and it contracts by
# ~0.45 a pass, so 4 passes lands within ~80 lb -- close, but over the 25 lb
# tolerance, which is why those weights are reported flagged. 7-8 clears it. This is
# overridable rather than simply raised because more passes is the ONE lever that
# turns a flagged weight into a converged one and it costs only time, and because it
# does not help every case equally: the `optimal` profiles converge honestly
# (df/dw -0.17 to -0.23, the right sign for weight relief) while A_capped stalls at a
# residual ratio of 0.968 with a POSITIVE df/dw, meaning sizer noise of +-30-50 lb is
# already swamping the real sensitivity there. More passes would converge that one
# cosmetically, not honestly.
W_PASSES = int(os.environ.get("ARC_W_PASSES", "4"))

PROFILES = {                    # (root t/c, tip/root) -> the 5 spline control points
    "optimal": (0.250, 0.58),   # the sweep's peak, 344.0 nmi
    "capped":  (0.220, 0.75),   # inside conventional thickness, 343.0 nmi
}
ARCHS = {          # region A end, region A rule, pinned straight line, rear schedule
    # Arc A is "preserved", NOT "root_le_fixed". Under root_le_fixed the product
    # p * c is held invariant, so pinning p at 0.750 SHRINKS region A's chord from
    # 105 to 95.13 in and the box tops out at 71.49 * (1 - 0.12/p) <= 62.91 in --
    # under the 65 in the inboard nacelle needs, at ANY p (arc_a_constfrac derives
    # the bound). It is infeasible by construction, and the SLSQP run that looked
    # like it worked never reached feasibility: exit mode 9, three of five width
    # stations violated, and width_req_in was never written to the JSON so nothing
    # compared them. The shipped Arc A design point has always used "preserved".
    # --region-a-rule overrides this per run.
    "A": (REGION_A_AS_BUILT_IN, "preserved", None, SCHEDULE_A),
    "B": (w2.REGION_A_END_IN, "preserved", w2.FRONT_PCT, SCHEDULE),
    "C": (w2.REGION_A_END_IN, "root_le_fixed", None, SCHEDULE),
}

# Arcs whose AFT SPAR must be straight, which is a statement about geometry, not a
# design variable to be pinned.
#
# The parameterization already builds x_LE(y) = x_spar - p * c(y), so the line at
# the chord fraction p = `wingbox_pct` is straight at constant x by construction --
# measured on the exported Arc A geometry it holds to 0.16 in over 675 in of span.
# The rear spar used for the box WIDTH and DEPTH constraints is a separate,
# SCHEDULED fraction, and it sits at
#
#     x_aft(y) = x_spar + (rear(y) - p) * c(y)
#
# which is constant in y -- straight -- if and only if rear(y) == p. So a straight
# aft spar is not obtained by pinning p to a chosen number; it is obtained by making
# the SCHEDULE follow p wherever the optimizer puts it.
#
# Pinning was tried and is actively harmful: with bounds collapsed to (0.750, 0.750)
# SLSQP carries a zero-range design variable sitting on its own bound and fails with
# "positive directional derivative for linesearch" -- measured, with and without the
# planform rebuild that the pin branch also triggers. Leaving p free costs nothing
# and removes the degeneracy: p simply becomes the spar station, and the depth
# requirement reads the retention there.
# value the spar fraction is FIXED at, and removed from the design variables.
# rear(y) == p == this, both constant, so x_aft = x_spar + (rear - p) * c(y) is
# constant in y: the aft spar is straight. 0.750 is chosen because the box width it
# gives on Arc A's frozen 103.38 in region-A chord is (0.750 - 0.12) * 103.38 =
# 65.13 in, which meets the 65 in nacelle requirement almost exactly -- any less and
# the inboard box fails, any more and it is wasted depth.
#
# NOT a design variable and NOT pinned. Collapsing its bounds to (0.750, 0.750) was
# tried and fails: SLSQP carries a zero-range variable on its own bound and exits
# with "positive directional derivative for linesearch". Making the schedule chase
# the free variable instead was also tried and DIVERGES, because param.py:523 bakes
# _box_frac in at setup, so nothing inside the optimization ties the variable to the
# schedule it was built from (measured: p ran 0.681 -> 0.616 -> 0.544 -> 0.466 with
# drag climbing 11183 -> 11509 -> 12158 N).
#
# taper_B stays free: growing outboard chord to reach the aileron depth is a TAPER
# change, which is the mechanism that should pay for it.
STRAIGHT_AFT_PCT = {"A": 0.750}


AIRFOIL_NAME = None      # set by section(); the export needs the NAME, not just the curve


def section(name=None):
    """The section this design is built on: as-built, or a database airfoil.

    A section reaches OAS as exactly TWO scalars -- c_max_t and the t/c
    distribution -- through the Raymer form factor (viscous_drag.py:103). But it
    reaches the STRUCTURE through its thickness retention at the spar, which
    fixes c_req = depth / (retention * t/c) and therefore the chord, the area and
    the weight. That second path is much the larger one: e694 keeps 0.142 of
    chord at the spar against the as-built's 0.096, which is 20 in of chord not
    spent buying depth.
    """
    global AIRFOIL_NAME
    AIRFOIL_NAME = None if name in (None, "as-built") else name
    if name in (None, "as-built"):
        af = asbuilt()
    else:
        import aerosandbox as asb
        af = asb.Airfoil(name)
    xs = np.linspace(0.05, 0.95, 300)
    t = np.array([float(af.local_thickness(x_over_c=x)) for x in xs])
    ret = lambda x: float(np.interp(x, xs, t / t.max()))
    c_max_t = float(xs[int(np.argmax(t))])
    return ret, c_max_t


def blended_section(name_in, name_out, f0, f1, semi_in=SEMI_IN):
    """Two sections lofted along the span: `name_in` inboard, `name_out` outboard.

    The two requirements on the section live at opposite ends of the wing, which is
    why one section cannot serve both. Inboard carries nearly all the wetted area
    and has chord to spare, so it wants L/D. The aileron station wants THICKNESS AT
    THE SPAR -- depth = retention(spar) * t/c * chord -- and at a 0.750c spar e694
    keeps only 0.640 of its thickness, which is what forced a 29% chord growth.
    An aft-thick section keeps 0.81 there, and outboard of 80% semi-span there is
    only 11.8% of the planform area for its poorer L/D to be paid on.

    Linear loft on the THICKNESS distribution between f0 and f1 (fractions of
    semi-span), which is what a real lofted wing does between two defining
    sections. Both returned quantities therefore vary along the span:

      ret_at(y, spar)  thickness retention -- fixes the depth, and so the chord
      cmt_at(y)        c_max_t -- the only other thing OAS sees of a section

    OAS takes c_max_t as an array over panels with no change (verified: a constant
    array reproduces the scalar exactly, and the analytic partials still match
    complex-step to 1e-19), so a spanwise section costs nothing but this function.
    """
    xs = np.linspace(0.05, 0.95, 300)

    def prof(name):
        if name in (None, "", "as-built"):
            af = asbuilt()
        else:
            import aerosandbox as asb
            af = asb.Airfoil(name)
        return np.array([float(af.local_thickness(x_over_c=float(x))) for x in xs])

    t_in, t_out = prof(name_in), prof(name_out)

    def w_of(y_in):
        return float(np.clip((abs(y_in) / semi_in - f0) / (f1 - f0), 0.0, 1.0))

    def t_at(y_in):
        w = w_of(y_in)
        return (1.0 - w) * t_in + w * t_out

    def ret_at(y_in, spar):
        t = t_at(y_in)
        return float(np.interp(spar, xs, t / t.max()))

    def cmt_at(y_in):
        t = t_at(y_in)
        return float(xs[int(np.argmax(t))])

    return ret_at, cmt_at, w_of


# Arc A only. B and C keep their 0.574c spar, where e694 is already excellent
# (retention 0.935) and there is nothing to buy.
# Started as LATE as the requirement allows. The blend has to be COMPLETE by the
# aileron: full goe16k retains 0.8106 at a 0.750c spar and 0.8237 is what the 7 in
# floor needs at the delivered chord, so a partially blended section cannot reach
# it -- there is no credit for starting earlier, only cost. The start is therefore
# put on the last RIB inboard of the aileron that still leaves a buildable
# transition: y = 601.0 in (84.9% semi), one rib bay to 639.5, a 36.2 in loft
# (5.1% of semi-span) with only 8.3% of the planform area outboard of it.
# 562.5 in (79.4%, a 74.7 in loft) is the fallback if the aero side wants a gentler
# spanwise pressure transition; it costs L/D over 12.2% of the area instead of 8.3%.
# 524.0 in (74.0% semi, a rib station), NOT as late as possible. Starting the blend
# later fails the requirement: the 7 in depth holds from ROOT TO AILERON, not merely
# AT the aileron, and the straight aft spar pushes the spar aft in chord fraction as
# the chord shrinks -- 0.750c at the root, 0.784c by y = 600 -- where e694 retains only
# ~0.58. Between the point the spar gets too far aft and the point the blend arrives,
# depth collapses. Measured minima over root->aileron:
#       start 601.0 in  ->  5.80 in at y = 601   (73 in of span below 7)
#       start 562.5 in  ->  6.80 in at y = 565
#       start 524.0 in  ->  6.84 in at y = 637   the dip is gone; the AILERON binds
# Once the minimum lands at the aileron, taper_B can lift the whole curve. It costs
# L/D over 16.6% of the planform area instead of 8.3%, which is the price of the
# requirement being a span, not a station.
_BLEND_START_IN = 524.0
SECTION_BLEND = {
    "A": ("e694", "goe16k", _BLEND_START_IN / SEMI_IN, 0.90),
}

RET, C_MAX_T = section()          # replaced at run time by --airfoil
RET_AT = CMT_AT = None            # set per arc when that arc blends sections


def spar_at_aileron(schedule):
    """Aft-spar chord fraction at the aileron -- per arc, since it sets the depth.

    Not a constant: Arc A's straight aft spar puts it at 0.750c where the shared
    schedule puts it at 0.574c, and e694 keeps only 0.640 of its thickness at
    0.750c against 0.935 at 0.574c. The depth requirement is
    ``depth = retention(spar) * t/c * chord``, so the SAME 6 in needs about 46%
    more chord on Arc A than on B or C. That is the real price of a straight aft
    spar, and it is paid in area, not in the spar.
    """
    return float(rear_spar_fraction(Y_AIL, schedule))


def solve_toc_cp(prob, root_toc, ratio):
    """Control points whose spline OUTPUT is the linear ramp, not whose VALUES are.

    ``np.linspace`` on the control points does NOT give a linear t/c. ``SplineComp``
    with ``method="bsplines"`` approximates its control points rather than passing
    through them, so the delivered curve sags below the line inboard, crosses near
    mid-span and rides above it outboard -- 3.0% at worst, and +2.43% at the aileron,
    where it sets ``c_req = depth / (retention * t/c)`` and therefore the chord, the
    area and the weight. ``plot_toc_request.py`` measures it.

    The spline is LINEAR in its control points, so inverting it is a least-squares
    solve, and an order-4 basis represents a straight line exactly -- the residual
    below is machine noise, not a fit error.

    Both traps the study hit before are closed here. The basis is probed from the
    LIVE model, so it is sampled at the true mid-panel stations OAS interpolates at
    (``utils/interpolation.py``, ``mid_panel=True``); rebuilding it on evenly-spaced
    points silently solved a different problem. And the result is checked, because a
    target the basis cannot represent comes back as a quiet bad fit -- the earlier
    attempt returned negative t/c and nonsense drag.
    """
    saved = np.asarray(prob.get_val("wing.t_over_c_cp")).copy()
    n = saved.size
    # The mesh is an OUTPUT, so it does not exist until the model has run once --
    # reading it straight after setup() gives zeros and the span normalisation below
    # divides by zero, silently producing NaN control points.
    prob.run_model()
    ys = np.abs(np.asarray(prob.get_val("wing.mesh", units="m"))[0, :, 1])
    s_node = (ys - ys[0]) / (ys[-1] - ys[0])
    s = 0.5 * (s_node[:-1] + s_node[1:])           # mid-panel, as the spline is read
    target = root_toc + (root_toc * ratio - root_toc) * s

    basis = np.empty((target.size, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        prob.set_val("wing.t_over_c_cp", e)
        prob.run_model()
        basis[:, i] = np.asarray(prob.get_val("wing.t_over_c")).ravel()

    cp, *_ = np.linalg.lstsq(basis, target, rcond=None)
    resid = float(np.max(np.abs(basis @ cp - target)))
    prob.set_val("wing.t_over_c_cp", saved)
    if resid > 1e-6 or cp.min() <= 0.0:
        raise RuntimeError(
            "the t/c inversion did not close: residual {:.3e}, control points {}. A "
            "non-positive control point or a residual this size means the basis "
            "cannot represent the requested distribution."
            .format(resid, np.round(cp, 5)))
    return cp, resid


def width_stations(spar_ail, c_req):
    """The box-width requirements, in inches. ONE definition.

    The aileron entry is the depth requirement re-encoded: at a fixed spar station
    depth is strictly proportional to chord, so ``depth >= DEPTH_REQ`` is exactly
    ``chord >= c_req``, and the box width there is ``(spar - front) * chord``. The
    model and the closed-form inversion both read this, so they cannot disagree.
    """
    return ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0),
            (Y_AIL, (spar_ail - w2.FRONT_PCT) * c_req),
            (674.9, w2.JUNCTION_BOX_IN))


def probe_map(prob, y_ail=Y_AIL):
    """Measure width(taper_B) and the delivered t/c at the aileron. Two evaluations.

    ``RegionPlanform`` builds every chord as
    ``c0 * (p0/p)**e * (1 + (lam-1)*t) / (1 + (lam0-1)*t)`` (param.py:546), which is
    AFFINE in ``lam`` with everything else frozen -- the component declares the
    matching analytic partial for exactly that reason. So two evaluations pin the
    map exactly, and no search is needed to invert it.

    The delivered t/c does not depend on ``lam``: the spanwise stations are fixed
    for the life of the problem (span is pinned and no design variable moves a
    station), so the ``t_over_c`` spline is sampled at the same panel midpoints
    whatever the taper. That is CHECKED here rather than assumed -- if it ever stops
    holding, the inversion below silently solves the wrong problem.
    """
    def read(lam):
        prob.set_val("wing.taper_B", float(lam))
        prob.run_model()
        w = np.asarray(prob.get_val("wingbox_width", units="m")).ravel() / config.SCALE
        m = np.asarray(prob.get_val("wing.mesh", units="m")) / config.SCALE
        ym = np.abs(m[0, :, 1]); yp = 0.5 * (ym[:-1] + ym[1:])
        toc = np.asarray(prob.get_val("wing.t_over_c")).ravel()
        return w, float(np.interp(y_ail, yp, toc))

    lam0 = float(prob.get_val("wing.taper_B")[0])
    la, lb = lam0, lam0 + 0.10
    wa, toc_a = read(la)
    wb, toc_b = read(lb)
    prob.set_val("wing.taper_B", lam0)

    # Tolerance is RELATIVE and deliberately not machine-epsilon: sampling the
    # spline at panel midpoints carries ~1e-7 of round-off even when the stations
    # are identical. Depth is linear in t/c, so a relative drift of eps moves the
    # delivered depth by eps -- 1e-5 is four orders inside the tolerance the
    # requirement itself carries, while still catching a real taper dependence.
    drift = abs(toc_b - toc_a) / max(abs(toc_a), 1e-12)
    if drift > 1e-5:
        raise RuntimeError(
            f"t/c at the aileron moved {drift:.2e} (relative) when taper_B changed "
            f"{la:.3f} -> {lb:.3f}. The closed-form taper assumes it does not, so "
            f"the spanwise stations must now depend on the taper. Re-derive the "
            f"inversion.")

    slope = (wb - wa) / (lb - la)
    # The delivered t/c distribution rides along: it is what the SplineComp actually
    # produced from the requested control points, sampled at the true mid-panel
    # stations, and comparing it against the request is the only way to see that the
    # two differ (plot_toc_request.py).
    m = np.asarray(prob.get_val("wing.mesh", units="m")) / config.SCALE
    ym = np.abs(m[0, :, 1])
    return {"intercept_in": wa - slope * la, "slope_in": slope,
            "toc_ail": toc_a, "lam0": lam0,
            "y_panel_in": 0.5 * (ym[:-1] + ym[1:]),
            "y_node_in": ym,
            "toc_panel": np.asarray(prob.get_val("wing.t_over_c")).ravel(),
            "toc_cp": np.asarray(prob.get_val("wing.t_over_c_cp")).ravel()}


def solve_taper(pmap, req_in, bounds):
    """The smallest taper_B that clears every width station. Closed form.

    Each station's width is affine in the taper, and every requirement is a FLOOR,
    so each station names a minimum taper and the binding one is simply the largest.
    Drag rises monotonically with taper over the whole feasible range (measured:
    out/logs/taper_sweep_arcA.json), so that smallest feasible value IS the optimum
    -- this replaces the search, it does not approximate it.

    A station with zero slope sits in region A, where the chord is frozen and no
    design variable can move it. Such a station is either already satisfied or the
    design is infeasible; it cannot be "optimized" either way.
    """
    a, b = np.asarray(pmap["intercept_in"]), np.asarray(pmap["slope_in"])
    req = np.asarray(req_in, dtype=float)
    need, frozen_short = [], []
    for i, (ai, bi, wi) in enumerate(zip(a, b, req)):
        if abs(bi) < 1e-9:
            if ai < wi - 1e-6:
                frozen_short.append((i, ai, wi))
            continue
        need.append((wi - ai) / bi)
    if frozen_short:
        raise ValueError(
            "width stations in region A are short and the taper cannot reach them: "
            + "; ".join(f"station {i}: {got:.2f} in against {want:.2f} required"
                        for i, got, want in frozen_short))
    lam = max(need) if need else bounds[0]
    return float(np.clip(lam, *bounds)), need


def optimize(y_a_in, rule, pin_p, cp_toc, c_req, schedule, fix_pct=None,
             probe=False, taper_fixed=None):
    spar_ail = spar_at_aileron(schedule)
    stations = width_stations(spar_ail, c_req)
    w2.REAR_SCHEDULE, w2.WIDTH_STATIONS = schedule, stations
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = stations
    pct0 = config.WINGBOX_CHORD_PCT_BOUNDS
    config.WINGBOX_CHORD_PCT_BOUNDS = pct0 if pin_p is None else (pin_p, pin_p)
    saved = param.REGION_A_RULE[w2.BASELINE]
    param.REGION_A_RULE[w2.BASELINE] = rule
    try:
        mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, y_a_in)
        if pin_p is not None:
            planform0 = baseline_planform(stick, regions, rule=rule)
        # c_max_t is read once when the viscous component is set up, so the
        # section has to be injected before the problem is built.
        orig_build_surface = ro.build_surface

        def _surface(mesh_, stick_, regions_, **kw):
            sd = orig_build_surface(mesh_, stick_, regions_, **kw)
            if CMT_AT is None:
                sd["c_max_t"] = C_MAX_T
            else:
                # One c_max_t per PANEL. Span is pinned and the DVs are taper,
                # twist, t/c and alpha -- none of which move a spanwise station --
                # so the panel y positions are fixed for the life of the problem
                # and this array can be built once here.
                ym = np.abs(np.asarray(mesh_)[0, :, 1]) / config.SCALE   # inches
                yp = 0.5 * (ym[:-1] + ym[1:])
                sd["c_max_t"] = np.array([CMT_AT(v) for v in yp])
            return sd

        ro.build_surface = _surface
        try:
            prob, _ = ro.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
        finally:
            ro.build_surface = orig_build_surface
        if fix_pct is not None:
            prob.set_val("wing.wingbox_pct", fix_pct)
        cp, toc_resid = solve_toc_cp(prob, cp_toc[0], cp_toc[1])
        prob.set_val("wing.t_over_c_cp", cp)
        if taper_fixed is not None:
            prob.set_val("wing.taper_B", float(taper_fixed))
        prob.run_model()
        if probe:
            return probe_map(prob)
        s0 = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
        alpha0 = trim_alpha(prob, w2.W / (q * s0))
        # A quantity pinned by the arc is not a design variable. pin_p collapses
        # the bounds to a single value, and SLSQP carrying a zero-range variable on
        # its own bound is the documented "positive directional derivative for
        # linesearch" failure (run_opt.add_optimization). Treat pin_p exactly as
        # fix_pct: set it, and leave it out.
        ro.add_optimization(prob, "plan_l", mesh, planform0, s0, mode="fixed_lift",
                            weight=w2.W,
                            pct_dv=(fix_pct is None and pin_p is None),
                            taper_dv=(taper_fixed is None))
        prob.set_val("wing.t_over_c_cp", cp)          # setup() reset it
        if pin_p is not None:
            prob.set_val("wing.wingbox_pct", pin_p)
        if fix_pct is not None:
            prob.set_val("wing.wingbox_pct", fix_pct)   # setup() reset it
        if taper_fixed is not None:
            prob.set_val("wing.taper_B", float(taper_fixed))   # setup() reset it
        prob.set_val("alpha", alpha0, units="deg")
        prob.run_model()
        prob.run_driver()
    finally:
        param.REGION_A_RULE[w2.BASELINE] = saved
        config.WINGBOX_CHORD_PCT_BOUNDS = pct0

    r = w2.evaluate(prob, regions.y_c_start)
    m = np.asarray(prob.get_val("wing.mesh", units="m")) / config.SCALE
    ym = np.abs(m[0, :, 1]); yp = 0.5 * (ym[:-1] + ym[1:])
    toc = np.asarray(prob.get_val("wing.t_over_c")).ravel()
    r["toc_delivered_ail"] = float(np.interp(Y_AIL, yp, toc))
    r["chord_at_aileron_in"] = float(np.asarray(prob.get_val("station_chord", units="m"))[3] / config.SCALE)
    ret_ail = RET(spar_ail) if RET_AT is None else RET_AT(Y_AIL, spar_ail)
    r["depth_delivered_in"] = ret_ail * r["toc_delivered_ail"] * r["chord_at_aileron_in"]
    r["spar_at_aileron"] = spar_ail
    r["retention_at_spar"] = ret_ail
    # Downstream draws depth, box width and the spar ratios from this, so it has
    # to travel with the design: Arc A's schedule is not B's and not C's.
    r["rear_schedule"] = [[float(a_), float(b_)] for a_, b_ in schedule]
    r["section_blend"] = None      # filled in by solve() when the arc blends
    r["toc_root"], r["toc_tip"] = float(toc[0]), float(toc[-1])
    r["t_over_c_cp"] = cp.tolist()
    r["alpha"] = float(prob.get_val("alpha", units="deg")[0])
    r["twist_cp"] = prob.get_val("wing.twist_cp", units="deg").tolist()
    r["success"] = bool(prob.driver.result.success)
    r["_prob"] = prob
    r["_toc_full"] = toc
    return r


def solve(arc, profile):
    """Converge the depth requirement, then converge the weight."""
    global RET_AT, CMT_AT
    y_a, rule, pin_p, schedule = ARCHS[arc]
    spar_ail = spar_at_aileron(schedule)
    cp_toc = PROFILES[profile]
    label = f"arc {arc} / {profile}"

    blend = SECTION_BLEND.get(arc)
    if blend is None:
        RET_AT = CMT_AT = None
        ret_ail = RET(spar_ail)
        sec_note = "single section"
    else:
        n_in, n_out, f0, f1 = blend
        RET_AT, CMT_AT, _w = blended_section(n_in, n_out, f0, f1)
        ret_ail = RET_AT(Y_AIL, spar_ail)
        sec_note = (f"{n_in} inboard -> {n_out} outboard, transition "
                    f"{f0:.0%}-{f1:.0%} semi ({f0*SEMI_IN:.0f}-{f1*SEMI_IN:.0f} in)")
    fix_pct = STRAIGHT_AFT_PCT.get(arc)
    straight_aft = fix_pct is not None
    if straight_aft:
        schedule = ((356.0, fix_pct), (674.9, fix_pct))
        spar_ail = fix_pct
        print(f"  {label}: STRAIGHT aft spar -- rear spar and wingbox_pct both FIXED "
              f"at {fix_pct:.3f}c, and wingbox_pct is NOT a design variable", flush=True)
    else:
        print(f"  {label}: rear spar {schedule[0][1]:.3f}c -> {schedule[-1][1]:.3f}c, "
              f"aileron spar {spar_ail:.4f}c", flush=True)
    print(f"  {label}: {sec_note}", flush=True)

    def _ret(spar):
        return RET(spar) if RET_AT is None else RET_AT(Y_AIL, spar)

    # --- depth: c_req from the DELIVERED t/c, and for a straight aft spar also
    #     from the DELIVERED p, since the spar station is then a design outcome.
    p_pinned = fix_pct if fix_pct is not None else pin_p
    if p_pinned is not None:
        # The taper is SOLVED, not searched. Two evaluations measure the affine
        # width(taper) map and the delivered t/c at the aileron; the requirement
        # then inverts exactly. The old loop below re-ran a full SLSQP up to five
        # times to chase the same number and, on arc A, returned the identical
        # infeasible point every pass.
        seed = optimize(y_a, rule, pin_p, cp_toc, 1.0, schedule,
                        fix_pct=fix_pct, probe=True)
        ret_ail = _ret(spar_ail)
        c_req = DEPTH_REQ / (ret_ail * seed["toc_ail"])
        req_in = [w for _, w in width_stations(spar_ail, c_req)]
        lam, per_station = solve_taper(seed, req_in, config.TAPER_B_BOUNDS)
        binding = int(np.argmax([-1e30 if not np.isfinite(v) else v
                                 for v in per_station])) if per_station else -1
        print(f"  {label}: t/c at the aileron {seed['toc_ail']:.5f}, retention "
              f"{ret_ail:.4f} -> c_req {c_req:.2f} in", flush=True)
        print(f"  {label}: taper_B solved = {lam:.6f} (per-station minima "
              f"{[round(v, 4) for v in per_station]}, binding index {binding})",
              flush=True)
        r = optimize(y_a, rule, pin_p, cp_toc, c_req, schedule,
                     fix_pct=fix_pct, taper_fixed=lam)
        err = r["depth_delivered_in"] - DEPTH_REQ
        print(f"  {label}: depth delivered {r['depth_delivered_in']:.2f} in "
              f"({err:+.3f} vs the {DEPTH_REQ:.1f} in requirement, "
              f"{100 * err / DEPTH_REQ:+.1f}%), drag {r['drag_N']:.1f} N", flush=True)
    else:
        r = _solve_depth_loop(y_a, rule, pin_p, cp_toc, schedule, fix_pct,
                              spar_ail, straight_aft, _ret, label)

    return _finish(r, arc, profile, label, blend, cp_toc)


def _solve_depth_loop(y_a, rule, pin_p, cp_toc, schedule, fix_pct, spar_ail,
                      straight_aft, _ret, label):
    """The original search. Still used when the spar fraction is a design variable.

    With ``wingbox_pct`` free under ``root_le_fixed`` the chord carries a ``(p0/p)``
    factor, so width is no longer affine in the taper alone and the one-line
    inversion does not apply. Arc C is the case.
    """
    toc_use = None
    for p in range(1, MAX_PASS + 1):
        if toc_use is None:                     # seed from the baseline loft
            w2.apply_wing2_box()
            _, stick0, _, _ = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
            ys = np.abs(np.asarray(stick0.le[:, 1], dtype=float))
            toc_use = float(np.interp(Y_AIL, ys, stick0.toc))
        ret_ail = _ret(spar_ail)
        c_req = DEPTH_REQ / (ret_ail * toc_use)
        r = optimize(y_a, rule, pin_p, cp_toc, c_req, schedule, fix_pct=fix_pct)
        err = r["depth_delivered_in"] - DEPTH_REQ
        extra = (f", spar {spar_ail:.3f}c fixed (ret {ret_ail:.4f})"
                 if straight_aft else "")
        print(f"  {label} depth pass {p}: c_req {c_req:6.2f} -> depth "
              f"{r['depth_delivered_in']:5.2f} in ({err:+.3f}), drag {r['drag_N']:9.1f} N"
              f"{extra}", flush=True)
        # The requirement is a FLOOR. Over-delivering is free -- it happens when
        # the aileron width constraint stops binding and the chord is set
        # elsewhere -- so anything at or above the floor is done. Only a shortfall
        # needs another pass, and only a shortfall can be fixed by more chord.
        if err >= -TOL_IN:
            break
        toc_use = r["toc_delivered_ail"]
    return r


def _finish(r, arc, profile, label, blend, cp_toc):
    """Weight, range and bookkeeping -- identical for both taper paths."""
    # --- weight: bi-level fixed point, damped
    prob = r.pop("_prob"); toc_full = r.pop("_toc_full")
    comp = list(lifting_surfaces(read_degen_csv(config.BASELINES[w2.BASELINE])).values())[0][0]

    # THE SECTION THE DESIGN IS ON, not the baseline loft. Arc A picks goe16k
    # outboard precisely because it keeps 0.811 of its thickness at the far-aft
    # spar against e694's 0.640; exporting the baseline instead threw that away and
    # sized a box ~30% shallower than every reported depth claimed.
    if blend is not None:
        airfoil = {"inboard": blend[0], "outboard": blend[1],
                   "f_start": blend[2], "f_end": blend[3]}
    else:
        airfoil = AIRFOIL_NAME          # None means the as-built loft, correctly

    # The 1 g spanload at MTOW, off the SAME trimmed state the drag came from.
    # Without it WingCalc silently substitutes an elliptical distribution.
    alpha = float(prob.get_val("alpha", units="deg")[0])
    ca, sa = np.cos(np.radians(alpha)), np.sin(np.radians(alpha))
    sec_f = np.asarray(prob.get_val(f"{POINT}.aero_states.wing_sec_forces"))
    strip = sec_f.sum(axis=0)                                   # (ny-1, 3), newtons
    lift_n = strip[:, 2] * ca - strip[:, 0] * sa
    m_in = np.asarray(prob.get_val("wing.mesh", units="m")) / config.SCALE
    y_node = np.abs(m_in[0, :, 1])
    y_mid = 0.5 * (y_node[:-1] + y_node[1:])
    width_in = np.asarray(prob.get_val(f"{POINT}.wing.widths", units="m")) / config.SCALE
    lift_lb_in = (lift_n / width_in) / 4.4482216                # N/in -> lbf/in
    order = np.argsort(y_mid)                                   # root -> tip
    spanload = (y_mid[order], width_in[order], lift_lb_in[order])
    print(f"  {label}: spanload {2 * float((lift_lb_in * width_in).sum()):.0f} lb "
          f"over both wings at alpha {alpha:.3f} deg", flush=True)

    oas = {"mesh": np.asarray(prob.get_val("wing.mesh", units="m")), "toc": toc_full,
           "plate": comp.plate, "stick": comp.stick, "y_junction": 674.9,
           "airfoil": airfoil, "spanload": spanload}
    # The deck directory is wiped and rebuilt on every sizing, so two runs sharing a
    # tag delete each other's files mid-run. Include the section: arc/profile alone
    # collided the moment two airfoils were compared side by side.
    tag = f"{arc}_{profile}" + ("" if AIRFOIL_NAME is None else f"_{AIRFOIL_NAME}")
    # The sizer places the access cut-out from ONE wing-wide stringer pair
    # (default / alternative) in planformIn.csv, but each bay carries a different
    # stringer range -- bay 16 carries Stg 1..6, bay 19 carries Stg 6..9 -- so no
    # single pair suits every bay on every planform, and the run stops with
    # "bay N has nowhere to put the access cut-out". That is a deck-configuration
    # limit, not a property of the design, so it must not cost the aero result:
    # the weight is recorded as unavailable and the t/c, depth and drag stand.
    # ONE sizing. The deck's wing-weight estimate is HELD, not iterated: see
    # deck.write_deck for why the fixed point converged to the wrong quantity
    # (structure only, no systems) and below its own quantisation noise.
    # A sizing failure must not cost the aero result, so the weight is recorded
    # as unavailable and the t/c, depth and drag still stand.
    hist, sizing_error = [], None
    w_assumed = wcdeck._wing_loading_value(wcdeck.WC_DECK, "Wing Estimated weight")
    try:
        wcdeck.write_deck(wcdeck.WC_DECK, Path(LOGS) / f"deck_arc{tag}", mission.MTOW_LB, oas=oas)
        w_new = wcdeck.run_wingcalc(Path(LOGS) / f"deck_arc{tag}", Path(LOGS) / f"wc_arc{tag}")
    except Exception as exc:
        sizing_error = f"{type(exc).__name__}: {exc}"
        print(f"  !!! {label} sizing FAILED: {sizing_error}", flush=True)
    else:
        hist.append({"pass": 1, "w_in_lb": w_assumed, "w_wing_lb": w_new,
                      "residual_lb": w_new - w_assumed})
        gap = w_new - w_assumed
        print(f"  >>> {label} W_wing {w_new:.1f} lb, sized against a HELD estimate of "
              f"{w_assumed:.1f} lb ({gap:+.1f}, {100 * gap / w_assumed:+.1f}%)", flush=True)
        # The estimate is inertia relief, so it SHOULD sit above the structural
        # result -- the difference is the wing systems the sizer does not model.
        # A gap the other way means the relief is under-counted with no allowance
        # left at all, which is the case actually worth flagging.
        if gap > 0:
            print(f"  !!! the held estimate {w_assumed:.0f} lb is BELOW the sized "
                  f"structure {w_new:.0f} lb, so it carries no systems allowance and "
                  f"under-counts the relief. Raise it.", flush=True)
        else:
            print(f"      implied wing systems allowance {-gap:.0f} lb "
                  f"({-100 * gap / w_new:.1f}% of structure)", flush=True)

    r["sizing_error"] = sizing_error
    r["w_wing_lb"] = hist[-1]["w_wing_lb"] if hist else None
    r["batt_lb"] = mission.battery_lb(r["w_wing_lb"]) if hist else None
    r["R_nmi"] = mission.electric_range_nmi(r["w_wing_lb"], r["drag_N"]) if hist else None
    r["weight_history"] = hist
    r["w_wing_assumed_lb"] = w_assumed
    r["weight_method"] = "one sizing against the deck's held estimate; not iterated"
    r["converged"] = bool(hist)      # a single sizing either produced a weight or did not
    r["arc"], r["profile"] = arc, profile
    if blend is not None:
        r["section_blend"] = {"inboard": blend[0], "outboard": blend[1],
                              "f_start": blend[2], "f_end": blend[3]}
    r["root_toc_req"], r["ratio_req"] = cp_toc
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc", required=True, choices=sorted(ARCHS))
    ap.add_argument("--profile", required=True, choices=sorted(PROFILES))
    ap.add_argument("--airfoil", default="as-built",
                    help="database section name, or 'as-built' (default)")
    ap.add_argument("--region-a-rule", choices=("preserved", "root_le_fixed"),
                    default=None,
                    help="override the arc's region-A chord rule; see ARCHS for why "
                         "arc A is 'preserved'")
    a = ap.parse_args()

    if a.region_a_rule is not None:
        y_a, _rule, pin_p, sched = ARCHS[a.arc]
        ARCHS[a.arc] = (y_a, a.region_a_rule, pin_p, sched)
        print(f"region A rule: {_rule} -> {a.region_a_rule} (override)", flush=True)
    print(f"arc {a.arc}: region A rule '{ARCHS[a.arc][1]}'", flush=True)

    RET, C_MAX_T = section(a.airfoil)
    print(f"section {a.airfoil}: c_max_t {C_MAX_T:.4f}", flush=True)
    res = solve(a.arc, a.profile)
    res["airfoil"] = a.airfoil
    res["c_max_t"] = C_MAX_T
    # retention_at_spar and spar_at_aileron come from solve(), per arc
    suffix = "" if a.airfoil == "as-built" else f"_{a.airfoil}"
    out = os.path.join(LOGS, f"arc_optimal_toc_{a.arc}_{a.profile}{suffix}.json")
    with open(out, "w") as f:
        json.dump({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res.items()}, f, indent=2)
    wtxt = (f"W_wing {res['w_wing_lb']:.1f} lb, R {res['R_nmi']:.1f} nmi" if res["w_wing_lb"]
            else f"W_wing UNSIZED ({str(res['sizing_error'])[:70]})")
    print(f"\n  arc {a.arc} / {a.profile}: t/c {res['toc_root']:.4f} -> {res['toc_tip']:.4f}, "
          f"depth {res['depth_delivered_in']:.2f} in, S_ref {res['S_ref']*10.7639104:.1f} ft2, "
          f"drag {res['drag_N']:.1f} N, {wtxt}")
    print(f"  wrote {out}")
