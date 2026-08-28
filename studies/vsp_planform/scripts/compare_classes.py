"""The planform trade reduced to three constraint classes, best drag in each.

The full comparison sheet ranks seven wings that differ in several ways at once.
This one asks a narrower question: given a class of constraint on the planform,
what is the least drag available inside it?

  free                 no constraint on chord shape beyond the parameterization:
                       region A|B re-lofted to 176 in, taper_B and the straight
                       line's chord fraction both free. The aft spar kinks
                       0.750 -> 0.550. This is wing 3.
  straight front spar   the straight line pinned at the front spar's 0.12c, so
                       the leading edge is x_spar_fwd - 0.12c and the aft spar is
                       free to curve. Region A runs under the `preserved` rule,
                       since under `root_le_fixed` the LE is frozen by
                       construction and the question cannot be posed. Wing 7.
  constant chord        the ConstChord loft's own A|B breakpoint kept at 361.7 in,
                       so the constant bay survives 51% of the semi-span instead
                       of 25%. Wing 8, as-built t/c.

All three are at MTOW 382,547 N, span pinned at 118 ft, trimmed to the same lift,
under the same box-width stations, 6 in aileron depth at 90% semi-span and rear
spar schedule. Each is replayed from its stored design vector rather than
re-optimized, and the replayed drag is checked against the logged value.

DRAG IS NOT THE MERIT FUNCTION. The study ranks designs on electric range at
fixed MTOW, m_batt/D, where wing weight trades against battery. Break-even is
1.486 lb of wing weight per newton of drag, so each drag bar is annotated with
the weight that class must save to be worth having, rather than leaving a drag
ranking to be read as a verdict.

All three ARE now sized, so the merit function itself is plotted: wing weight,
electric range, and the range-vs-weight trade each architecture sits on. Arc B
needed a fix to get there -- its straight forward spar means the stringer ladder
sheds rungs from the BACK, so the deck's access cut-out pair (Stg 6 / Stg 8) was
on the wrong side of the box and bays 16-20 had nowhere to put the cut-out; see
coupling/deck.py:resolve_cutout. Plan L was never sized in this batch and is drawn
as 'not sized' rather than given a borrowed weight.

Writes out/figures/class_comparison.png.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

from studies.vsp_planform import config, param                 # noqa: E402
import studies.vsp_planform.run_opt as ro                       # noqa: E402
from studies.vsp_planform.run_opt import POINT, trim_alpha      # noqa: E402
import wing2_oas as w2                                         # noqa: E402
from wing5_mtow import stations_and_schedule                   # noqa: E402
from wing8_constchord_toc import REGION_A_AS_BUILT_IN          # noqa: E402
from doe_v3 import asbuilt                                     # noqa: E402
from studies.vsp_planform.param import rear_spar_fraction      # noqa: E402
from studies.vsp_planform.coupling import mission              # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")
FIGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "figures")
q = 0.5 * config.RHO * config.V_MS**2

# Electric range at fixed MTOW: break-even wing weight per newton of drag.
# m_batt at wing 3's 8613.9 lb, over wing 3's drag. See coupling/mission.py.
BREAK_EVEN_LB_PER_N = 15551.7 / 10463.9341

# The box-width requirement is set by what has to fit through the wing at each
# station. 176 in is the INBOARD nacelle and 356 in the OUTBOARD one; 674.9 in is
# the winglet junction, whose 20 in is what derives the junction chord.
NACELLES = {176.0: "inboard nacelle", 356.0: "outboard nacelle"}


def retention_fn(airfoil=None):
    """Fraction of max thickness a section still has at x/c.

    Retention belongs to the SECTION, and it is what turns t/c into depth:
    depth = retention(spar) * t/c * chord. Using the as-built curve on a wing
    built with another section understates its depth -- measured, 6.65 in against
    the 7.65 in the design actually delivers, because e694 keeps 0.935 of its
    thickness at the 0.574c spar where the as-built keeps 0.814.
    """
    if airfoil in (None, "", "as-built"):
        af = asbuilt()
    else:
        import aerosandbox as asb
        af = asb.Airfoil(airfoil)
    xs = np.linspace(0.05, 0.95, 300)
    t = np.array([float(af.local_thickness(x_over_c=x)) for x in xs])
    return lambda x: float(np.interp(x, xs, t / t.max()))


# Where the box is reported. RegionPlanform emits `station_chord` at whatever
# stations it is given, computed analytically from the baseline chord and the
# scale factor -- so asking it for a dense vector gives the exact chord
# distribution. Interpolating the 35-node resampled mesh instead runs ~1% low
# through region B and ~5% low near the aileron, where the chord changes fastest
# and the spanwise stations are clustering into the winglet; that error reads as
# a violated constraint on a design the optimizer has satisfied.
REPORT_STATIONS_IN = np.linspace(0.0, 674.9, 80)


def spanwise(r, schedule, ret):
    """Chord, aft-spar depth and box width per station, all in inches.

    Everything here comes from the model's own `station_chord`, not from the
    mesh -- see REPORT_STATIONS_IN.
    """
    y = REPORT_STATIONS_IN
    chord = r["station_chord_in"]
    if r.get("aft_pct") is not None:      # as-built: one fitted, unscheduled spar
        sp = np.full(y.shape, float(r["aft_pct"]))
        m0 = r["mesh"] / config.SCALE
        y0 = np.abs(m0[0, :, 1]); yp0 = 0.5 * (y0[:-1] + y0[1:])
        toc0 = np.interp(y, yp0, r["toc"])
        return y, chord, np.array([ret(v) for v in sp]) * toc0 * chord, (sp - w2.FRONT_PCT) * chord
    m = r["mesh"] / config.SCALE
    y_mesh = np.abs(m[0, :, 1])
    yp = 0.5 * (y_mesh[:-1] + y_mesh[1:])
    toc = np.interp(y, yp, r["toc"])
    sp = np.array([float(rear_spar_fraction(v, schedule)) for v in y])
    depth = np.array([ret(v) for v in sp]) * toc * chord
    width = (sp - w2.FRONT_PCT) * chord
    return y, chord, depth, width


PLAN_L_COLOR = "#8C7B6B"
# Plan L's straight spar, least-squares fitted over regions A+B (README): the
# loft has no scheduled box, so this is what its aft spar actually is. The front
# spar is not measurable from the loft; the study's 0.12c is applied so the box
# width is comparable, and the panels say so.
PLAN_L_AFT_PCT = 0.60065


def baseline_case(name="plan_l"):
    """As-built baseline at MTOW: run, trimmed, never optimized."""
    from studies.vsp_planform.run_opt import load_baseline
    schedule, stations, _ = stations_and_schedule()
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    report = tuple((float(v), 0.0) for v in REPORT_STATIONS_IN)
    config.WINGBOX_WIDTH_STATIONS = tuple(stations) + report

    mesh, stick, regions, planform0, _, _ = load_baseline(name)
    prob, _ = ro.build_problem(name, mesh, stick, regions, planform0)
    prob.run_model()
    s_ref = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    trim_alpha(prob, w2.W / (q * s_ref))
    s_ref = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    cd = lambda k: float(prob.get_val(f"{POINT}.wing_perf.{k}")[0])
    n_st = len(stations)
    return {"S_ref": s_ref,
            "drag_N": q * s_ref * (cd("CDi") + cd("CDv") + cd("CDw")),
            "induced_N": q * s_ref * cd("CDi"), "viscous_N": q * s_ref * cd("CDv"),
            "mesh": np.asarray(prob.get_val("wing.mesh", units="m")),
            "toc": np.asarray(prob.get_val("wing.t_over_c")).ravel().copy(),
            "twist": np.asarray(prob.get_val("twist_abs", units="deg")).ravel().copy(),
            "station_chord_in": (np.asarray(prob.get_val("station_chord", units="m"))
                                 [n_st:] / config.SCALE),
            "constraint_width_in": None, "constraint_stations": stations,
            "aft_pct": PLAN_L_AFT_PCT,
            STRAIGHT_LINE_KEY: float(prob.get_val("wing.wingbox_pct")[0])}


# Arc A / B / C are the architectures; the wing numbers are the runs that
# produced them, kept in the labels so the design points stay traceable.
# Labels are the names alone; what each architecture IS goes in the caption, so
# no legend has to carry a description.
CLASSES = [
    ("Arc A", "#DD8452", "wing 8"),
    ("Arc B", "#4C72B0", "wing 7"),
    ("Arc C", "#C44E52", "wing 3"),
    ("Plan L", PLAN_L_COLOR, "reference"),
]
DEFINITIONS = ("Arc A: constant chord.   Arc B: straight forward spar.   "
               "Arc C: free (kinking aft spar).   Plan L: as-built reference.")
# The straight line each architecture is built around -- the chord fraction held
# straight, `wingbox_pct`. This is what actually separates them: Arc B pins it at
# the front spar, Arc A and Arc C let it optimize near the aft spar, and Plan L's
# is the least-squares fit to its as-built loft.
STRAIGHT_LINE_KEY = "wingbox_pct"


def replay(case, y_a_in, rule):
    """Rebuild a stored case and confirm it reproduces its logged drag."""
    schedule, stations, _ = stations_and_schedule()
    w2.REAR_SCHEDULE, w2.WIDTH_STATIONS = schedule, stations
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    # The design's own constraint stations, plus the reporting grid. The stations
    # are an OUTPUT location only -- nothing is constrained here, the case is
    # replayed rather than re-optimized -- so adding them cannot move the design.
    report = tuple((float(v), 0.0) for v in REPORT_STATIONS_IN)
    config.WINGBOX_WIDTH_STATIONS = tuple(stations) + report

    saved = param.REGION_A_RULE[w2.BASELINE]
    param.REGION_A_RULE[w2.BASELINE] = rule
    # A design point built on a chosen section carries its c_max_t, and c_max_t
    # is read once when the viscous component is set up. Replaying without it
    # rebuilds the wing with the as-built section's form factor: measured, a
    # 252 N discrepancy on Arc C, which the drag cross-check below catches.
    orig_build_surface = ro.build_surface
    c_max_t = case.get("c_max_t")
    if c_max_t is not None:
        def _surface(mesh_, stick_, regions_, **kw):
            sd = orig_build_surface(mesh_, stick_, regions_, **kw)
            sd["c_max_t"] = float(c_max_t)
            return sd
        ro.build_surface = _surface
    try:
        prob, _, _, _, _ = w2.build(w2.BASELINE, y_a_in)
        prob.set_val("wing.taper_B", case["taper_B"])
        prob.set_val("wing.wingbox_pct", case["wingbox_pct"])
        prob.set_val("wing.twist_cp", np.array(case["twist_cp"]), units="deg")
        prob.set_val("alpha", case["alpha"], units="deg")
        if case.get("t_over_c_cp") is not None:
            prob.set_val("wing.t_over_c_cp", np.array(case["t_over_c_cp"]))
        prob.run_model()
    finally:
        param.REGION_A_RULE[w2.BASELINE] = saved
        ro.build_surface = orig_build_surface

    s_ref = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    cd = lambda k: float(prob.get_val(f"{POINT}.wing_perf.{k}")[0])
    drag = q * s_ref * (cd("CDi") + cd("CDv") + cd("CDw"))
    if abs(drag - case["drag_N"]) > 2.0:
        raise RuntimeError(f"replayed {drag:.1f} N != logged {case['drag_N']:.1f} N")
    m = np.asarray(prob.get_val("wing.mesh", units="m"))
    return {"S_ref": s_ref, "drag_N": drag,
            "induced_N": q * s_ref * cd("CDi"), "viscous_N": q * s_ref * cd("CDv"),
            "mesh": m,
            "toc": np.asarray(prob.get_val("wing.t_over_c")).ravel().copy(),
            "twist": np.asarray(prob.get_val("twist_abs", units="deg")).ravel().copy(),
            # the model's analytic chord at the reporting stations, inches
            "station_chord_in": (np.asarray(prob.get_val("station_chord", units="m"))
                                 [len(stations):] / config.SCALE),
            "constraint_chord_in": (np.asarray(prob.get_val("station_chord", units="m"))
                                    [:len(stations)] / config.SCALE),
            "constraint_width_in": (np.asarray(prob.get_val("wingbox_width", units="m"))
                                    [:len(stations)] / config.SCALE),
            "constraint_stations": stations,
            # retention belongs to the section, so the depth panel must use this
            "airfoil": case.get("airfoil"),
            STRAIGHT_LINE_KEY: float(prob.get_val("wing.wingbox_pct")[0])}


if __name__ == "__main__":
    # Prefer the t/c-optimised design points when they exist. arc_optimal_toc.py
    # writes one per architecture per profile; `optimal` is the sweep's peak
    # (root 0.250 -> tip 0.145) and `capped` stays inside conventional thickness
    # (0.220 -> 0.165). Fall back to the as-built-t/c design points otherwise, so
    # the figure is always producible.
    TOC_PROFILE = os.environ.get("ARC_TOC_PROFILE", "optimal")
    # The section is part of the design point's identity: arc_optimal_toc writes
    # <arc>_<profile>_<airfoil>.json for a chosen section and <arc>_<profile>.json
    # for the as-built one. Naming it here stops a stale file from a different
    # section being picked up silently -- which it was, for one architecture,
    # leaving two arcs on their old design points and one on a new one.
    ARC_AIRFOIL = os.environ.get("ARC_AIRFOIL", "e694")
    suffix = "" if ARC_AIRFOIL in ("", "as-built") else f"_{ARC_AIRFOIL}"

    def arc_case(arc, fallback_file, fallback_key):
        p_arc = os.path.join(LOGS, f"arc_optimal_toc_{arc}_{TOC_PROFILE}{suffix}.json")
        if os.path.exists(p_arc):
            c = json.load(open(p_arc))
            return c, (f"{ARC_AIRFOIL}, {TOC_PROFILE} t/c "
                       f"({c['toc_root']:.3f}→{c['toc_tip']:.3f})")
        print(f"  NOTE: {os.path.basename(p_arc)} absent -- Arc {arc} falls back "
              f"to the as-built section at as-built t/c")
        return json.load(open(os.path.join(LOGS, fallback_file)))[fallback_key], "as-built section, as-built t/c"

    w7log = json.load(open(os.path.join(LOGS, "wing7_design_point.json")))
    w8log = json.load(open(os.path.join(LOGS, "wing8_design_point.json")))

    cC, provC = arc_case("C", "wing7_design_point.json", "wing3_mtow")
    cB, provB = arc_case("B", "wing7_design_point.json", "wing7_mtow")
    cA, provA = arc_case("A", "wing8_design_point.json", "constchord_asbuilt")
    print(f"  Arc C: {provC}\n  Arc B: {provB}\n  Arc A: {provA}")
    print("  replaying Arc C (free) ...")
    r_free = replay(cC, w2.REGION_A_END_IN, "root_le_fixed")
    print("  replaying Arc B (straight front spar) ...")
    r_fwd = replay(cB, w2.REGION_A_END_IN, "preserved")
    print("  replaying Arc A (constant chord) ...")
    r_cc = replay(cA, REGION_A_AS_BUILT_IN, "root_le_fixed")
    # The sized weight belongs to the DESIGN POINT, not the replay: replay
    # reproduces the aero, WingCalc produced the weight. Carrying it across is what
    # lets the merit function be plotted instead of proxied by a break-even
    # allowance. `converged` travels with it -- these loops stop at their pass
    # limit, so the number is close but flagged, not certified.
    for _r, _c in ((r_cc, cA), (r_fwd, cB), (r_free, cC)):
        _r["w_wing_lb"] = _c.get("w_wing_lb")
        _r["R_nmi"] = _c.get("R_nmi")
        _r["w_converged"] = bool(_c.get("converged"))
        _r["sizing_error"] = _c.get("sizing_error")

    TOC_NOTE = provC
    print("  evaluating Plan L as-built (the reference) ...")
    r_pl = baseline_case("plan_l")
    # Plan L LAST in the list so the three optimized classes keep their order and
    # colours, and the reference reads as a reference.
    res = [r_cc, r_fwd, r_free, r_pl]          # Arc A, Arc B, Arc C, reference
    ref = r_pl["drag_N"]                        # Plan L as-built: every % is against it

    schedule, stations, _ = stations_and_schedule()
    # one retention curve per case, from that design's own section
    rets = [retention_fn(r.get("airfoil")) for r in res]
    span = [spanwise(r, schedule, rt) for r, rt in zip(res, rets)]

    fig = plt.figure(figsize=(16.5, 19.5))
    gs = fig.add_gridspec(5, 3, height_ratios=[1.0, 1.2, 1.0, 1.0, 1.0],
                          hspace=0.42, wspace=0.30, bottom=0.075)
    names = [c[0] for c in CLASSES]
    # Bar ticks get the short name only -- "Arc A constant chord" and its
    # neighbours overlap. The line panels carry the full descriptor in the legend.
    short = [c[0] for c in CLASSES]
    cols = [c[1] for c in CLASSES]
    xs = np.arange(len(CLASSES))

    # --- total drag, with the break-even weight each class must buy back
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(xs, [r["drag_N"] for r in res], color=cols, width=0.62)
    for i, r in enumerate(res):
        d = r["drag_N"] - ref
        lbl = f"{r['drag_N']:.0f}\n{100*(r['drag_N']/ref-1):+.2f}%"
        if abs(d) > 1e-9:
            # drag saved against Plan L is a weight ALLOWANCE at break-even: this
            # design may be that much heavier and still match Plan L on range.
            lbl += f"\nmay weigh\n{abs(BREAK_EVEN_LB_PER_N*d):+.0f} lb"
        ax.text(i, r["drag_N"] + 6, lbl, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(short, fontsize=9.5)
    ax.set_ylabel("drag, N"); ax.set_title("Total drag at MTOW")
    ax.set_ylim(min(r["drag_N"] for r in res) - 60, max(r["drag_N"] for r in res) + 130)
    ax.grid(alpha=0.25, axis="y")

    # --- where the penalty is spent
    ax = fig.add_subplot(gs[0, 1])
    ind = [r["induced_N"] for r in res]; vis = [r["viscous_N"] for r in res]
    ax.bar(xs, ind, color="#4C72B0", width=0.62, label="induced")
    ax.bar(xs, vis, bottom=ind, color="#DD8452", width=0.62, label="viscous")
    for i in range(len(CLASSES)):
        ax.text(i, ind[i] / 2, f"{ind[i]:.0f}", ha="center", va="center", color="w", fontsize=9)
        ax.text(i, ind[i] + vis[i] / 2, f"{vis[i]:.0f}", ha="center", va="center", color="w", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(short, fontsize=9.5)
    ax.set_ylabel("drag, N"); ax.set_title("Induced vs viscous (wave = 0)")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.25, axis="y")

    # Area in ft2 only -- the model works in m2, so it is converted here and
    # nowhere else, and m2 never reaches the figure.
    ax = fig.add_subplot(gs[0, 2])
    M2_FT2 = 10.7639104
    area_ft2 = [r["S_ref"] * M2_FT2 for r in res]
    ax.bar(xs, area_ft2, color=cols, width=0.62)
    for i, a in enumerate(area_ft2):
        ax.text(i, a + 1.2, f"{a:.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(short, fontsize=9.5)
    ax.set_ylabel("S_ref, ft²"); ax.set_title("Wing area")
    ax.set_ylim(min(area_ft2) - 16, max(area_ft2) + 26)
    ax.grid(alpha=0.25, axis="y")

    # --- planforms. x down, as in the study's other planform panels.
    ax = fig.add_subplot(gs[1, :])
    for (nm, c, tag), r in zip(CLASSES, res):
        m = r["mesh"] / config.SCALE
        y = np.abs(m[0, :, 1])
        ls = "--" if tag == "reference" else "-"
        ax.plot(y, m[0, :, 0], color=c, lw=1.6, ls=ls, label=nm)
        ax.plot(y, m[-1, :, 0], color=c, lw=1.6, ls=ls)
    for ys_, lab in NACELLES.items():
        ax.axvline(ys_, color="0.45", ls=":", lw=1.0)
        ax.annotate(lab, xy=(ys_, 0.985), xycoords=("data", "axes fraction"),
                    xytext=(3, 0), textcoords="offset points",
                    color="0.35", fontsize=7.5, va="top", ha="left")
    ax.axvline(0.90 * 708.0, color="#8172B2", ls="--", lw=1.1)
    ax.text(0.90 * 708.0, ax.get_ylim()[0], " aileron", color="#8172B2", fontsize=8, va="bottom")
    ax.invert_yaxis(); ax.set_xlabel("y, in"); ax.set_ylabel("x, in")
    ax.set_title("Planforms — leading and trailing edges")
    ax.legend(fontsize=8.5, loc="center left"); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[3, 2])
    for (nm, c, tag), (y, chord, _, _) in zip(CLASSES, span):
        ax.plot(y, chord, color=c, lw=1.6, ls="--" if tag == "reference" else "-")
    for ys_ in NACELLES:
        ax.axvline(ys_, color="0.45", ls=":", lw=1.0)
    ax.set_xlabel("y, in"); ax.set_ylabel("chord, in")
    ax.set_title("Chord distribution"); ax.grid(alpha=0.25)

    # --- twist
    ax = fig.add_subplot(gs[2, 0])
    for (nm, c, tag), r in zip(CLASSES, res):
        m = r["mesh"] / config.SCALE
        ax.plot(np.abs(m[0, :, 1]), r["twist"], color=c, lw=1.6,
                ls="--" if tag == "reference" else "-", label=nm)
    ax.axhline(config.TWIST_BOUNDS[1], color="0.6", ls=":", lw=1.0)
    ax.text(5, config.TWIST_BOUNDS[1], f" +{config.TWIST_BOUNDS[1]:.0f}° bound", color="0.45",
            fontsize=7.5, va="bottom")
    ax.set_xlabel("y, in"); ax.set_ylabel("twist, deg")
    ax.set_title("Twist distribution (absolute)")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)

    # --- t/c. The question the thickness work turns on: does the section get
    # thinner outboard, and by how much. Both baselines are lofted that way
    # (ConstChord 0.178 -> 0.100, Plan L 0.1774 constant), and wing 5 raises the
    # inboard end of that without touching the outboard.
    ax = fig.add_subplot(gs[2, 1])
    for (nm, c, tag), r in zip(CLASSES, res):
        m = r["mesh"] / config.SCALE
        y = np.abs(m[0, :, 1]); yp = 0.5 * (y[:-1] + y[1:])
        ax.plot(yp, r["toc"], color=c, lw=1.6, ls="--" if tag == "reference" else "-",
                label=nm)
    for ys_ in NACELLES:
        ax.axvline(ys_, color="0.45", ls=":", lw=1.0)
    ax.set_xlabel("y, in"); ax.set_ylabel("t/c")
    ax.set_title("Thickness ratio t/c", fontsize=10.5)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)

    # --- spar chord ratios. The front spar is 0.12c on every optimized design;
    # the aft spar is the piecewise-linear schedule. Plan L has neither -- it is
    # an as-built loft, so its ONE fitted straight spar is drawn instead and its
    # front spar is the study's 0.12c applied for comparability, not a measurement.
    ax = fig.add_subplot(gs[2, 2])
    yy = REPORT_STATIONS_IN
    sched = np.array([float(rear_spar_fraction(v, schedule)) for v in yy])
    # A, B and C are built to the SAME box -- front 0.12c, aft 0.750c held to
    # 356 in then kinking to 0.550c at the junction -- so their spar lines
    # coincide exactly. Staggered dashes draw all four rather than hiding three
    # under one curve. Plan L has no scheduled box at all: one fitted straight
    # spar, and a front spar that is the study's 0.12c applied for comparability.
    dashes = [(1, 0), (6, 3), (2, 2.5), (5, 2)]
    for (nm, c, tag), dash, r in zip(CLASSES, dashes, res):
        lbl = nm
        if tag == "reference":
            ax.plot(yy, np.full_like(yy, PLAN_L_AFT_PCT), color=c, lw=2.0, dashes=dash,
                    label=lbl)
        else:
            ax.plot(yy, sched, color=c, lw=1.9, dashes=dash, label=lbl)
        ax.plot(yy, np.full_like(yy, w2.FRONT_PCT), color=c, lw=1.4, dashes=dash, alpha=0.85)
    # the straight line each architecture is actually built around
    for (nm, c, tag), r in zip(CLASSES, res):
        p_str = r.get(STRAIGHT_LINE_KEY)
        if p_str is None:
            continue
        ax.plot([692], [p_str], marker="<", color=c, ms=9, mec="k", mew=0.7, clip_on=False, zorder=6)
        ax.annotate(f"{p_str:.3f}", xy=(702, p_str), fontsize=7.2, color=c, va="center")
    ax.text(0.02, 0.05, "lower band: front spar 0.12c (all four)\nmarkers: the STRAIGHT line each is built around",
            transform=ax.transAxes, fontsize=7.0, color="0.3")
    for ys_ in NACELLES:
        ax.axvline(ys_, color="0.45", ls=":", lw=1.0)
    ax.axvline(0.90 * 708.0, color="#8172B2", ls="--", lw=1.1)
    ax.annotate("breakpoints\n356 / 674.9 in", xy=(362, 0.28), fontsize=7.0, color="0.35")
    ax.set_ylim(0.0, 0.88); ax.set_xlim(0, 760)
    ax.set_xlabel("y, in"); ax.set_ylabel("spar station, x/c")
    ax.set_title("Front and aft spar chord ratios", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="upper left"); ax.grid(alpha=0.25)

    # --- aft-spar DEPTH. The requirement the whole wing 2/3 exercise turns on:
    # depth = retention(spar x/c) * t/c * chord, so chord taken for drag is depth
    # taken from the structure.
    ax = fig.add_subplot(gs[3, 0])
    for (nm, c, tag), (y, _, depth, _) in zip(CLASSES, span):
        ax.plot(y, depth, color=c, lw=1.6, ls="--" if tag == "reference" else "-")
    ax.axhline(6.0, color="#C44E52", ls="--", lw=1.2)
    ax.text(5, 6.0, " 6 in required at the aileron", color="#C44E52", fontsize=7.5, va="bottom")
    ax.axvline(0.90 * 708.0, color="#8172B2", ls="--", lw=1.1)
    for ys_ in NACELLES:
        ax.axvline(ys_, color="0.45", ls=":", lw=1.0)
    ax.set_xlabel("y, in"); ax.set_ylabel("aft-spar depth, in")
    for (nm, c, tag), (y, _, depth, _) in zip(CLASSES, span):
        d_ail = float(np.interp(0.90 * 708.0, y, depth))
        ax.plot([0.90 * 708.0], [d_ail], "o", color=c, ms=6, mec="k", mew=0.7, zorder=5)
    ax.set_title("Aft-spar depth — DELIVERED, not requested\n(retention × t/c × chord, spar 0.750c → 0.550c)",
                 fontsize=10.5)
    ax.grid(alpha=0.25)

    # --- wingbox WIDTH against the requirement at each station, nacelles named.
    ax = fig.add_subplot(gs[3, 1])
    for (nm, c, tag), (y, _, _, width) in zip(CLASSES, span):
        ax.plot(y, width, color=c, lw=1.6, ls="--" if tag == "reference" else "-")
    ys_req = [s_[0] for s_ in stations]; w_req = [s_[1] for s_ in stations]
    for (nm, c, tag), r in zip(CLASSES, res):
        if r.get("constraint_width_in") is not None:
            ax.plot(ys_req, r["constraint_width_in"], "o", color=c, ms=5, mec="k", mew=0.6, zorder=5)
    ax.plot(ys_req, w_req, "k_", ms=16, mew=2.0, label="required")
    for ys_, wr in zip(ys_req, w_req):
        lab = NACELLES.get(ys_, "")
        ax.annotate(f"{wr:.0f} in" + (f"\n{lab}" if lab else ""), xy=(ys_, wr),
                    xytext=(-6 if ys_ > 600 else 5, -15), textcoords="offset points",
                    fontsize=7.2, color="0.25", ha="right" if ys_ > 600 else "left")
    ax.set_xlabel("y, in"); ax.set_ylabel("wingbox width, in")
    ax.set_title("Wingbox width vs requirement\n(0.12c to the scheduled aft spar)", fontsize=10.5)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)

    # ================= THE MERIT FUNCTION ITSELF =================
    # Everything above is drag, area and geometry. These three are what the study
    # actually ranks on, and they exist only because all three arcs are sized.
    sized = [(nm, c, r) for (nm, c, tag), r in zip(CLASSES, res)
             if r.get("w_wing_lb") is not None]

    # --- wing weight, as sized by WingCalc at MTOW
    ax = fig.add_subplot(gs[4, 0])
    for i, ((nm, c, tag), r) in enumerate(zip(CLASSES, res)):
        w = r.get("w_wing_lb")
        if w is None:
            ax.text(i, 0.02, "not sized\n(as-built loft)", ha="center", va="bottom",
                    fontsize=8, color="0.45", transform=ax.get_xaxis_transform())
            continue
        ax.bar(i, w, color=c, edgecolor="k", lw=0.6)
        # A trailing * is the honest mark: the weight loop hit its pass limit
        # rather than the 25 lb tolerance.
        ax.text(i, w + 30, f"{w:,.0f}" + ("" if r["w_converged"] else "*"),
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    wl = [r["w_wing_lb"] for _n, _c, r in sized]
    if wl:
        ax.set_ylim(0.95 * min(wl), 1.03 * max(wl))
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("W_wing, lb")
    ax.set_title("Wing weight — WingCalc, sized at MTOW\n(* = weight loop hit its pass limit)",
                 fontsize=10.5)
    ax.grid(alpha=0.25, axis="y")

    # --- electric range: the actual ranking
    ax = fig.add_subplot(gs[4, 1])
    rr = [(nm, c, r["R_nmi"]) for nm, c, r in sized if r.get("R_nmi") is not None]
    for i, ((nm, c, tag), r) in enumerate(zip(CLASSES, res)):
        R = r.get("R_nmi")
        if R is None:
            ax.text(i, 0.02, "not sized", ha="center", va="bottom", fontsize=8,
                    color="0.45", transform=ax.get_xaxis_transform())
            continue
        ax.bar(i, R, color=c, edgecolor="k", lw=0.6)
        ax.text(i, R + 0.25, f"{R:.1f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")
    if rr:
        best = max(rr, key=lambda t: t[2])
        ax.axhline(best[2], color=best[1], ls="--", lw=1.1, alpha=0.7)
        ax.text(0.99, best[2], f" best: {best[0]} ", ha="right", va="bottom",
                fontsize=8, color=best[1], fontweight="bold",
                transform=ax.get_yaxis_transform())
        vals = [t[2] for t in rr]
        ax.set_ylim(0.97 * min(vals), 1.012 * max(vals))
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("electric range, nmi")
    ax.set_title("Electric range at fixed MTOW — THE MERIT FUNCTION\n(m_batt/D; wing weight trades against battery)",
                 fontsize=10.5)
    ax.grid(alpha=0.25, axis="y")

    # --- the trade each architecture sits on. One curve per design at ITS OWN
    # drag, so the vertical gap between curves is the drag difference and
    # movement along a curve is the weight difference. Plan L has a curve but no
    # point: it shows the weight Plan L would have to reach to match.
    ax = fig.add_subplot(gs[4, 2])
    wgrid = np.linspace(6400.0, 9200.0, 120)
    for (nm, c, tag), r in zip(CLASSES, res):
        ax.plot(wgrid, [mission.electric_range_nmi(w, r["drag_N"]) for w in wgrid],
                color=c, lw=1.6, ls="--" if tag == "reference" else "-",
                label=f"{nm} ({r['drag_N']:.0f} N)")
        w = r.get("w_wing_lb")
        if w is not None:
            ax.plot([w], [mission.electric_range_nmi(w, r["drag_N"])], "o",
                    color=c, ms=8, mec="k", mew=0.8, zorder=6)
    ax.set_xlabel("wing weight, lb"); ax.set_ylabel("electric range, nmi")
    ax.set_title("Range vs wing weight — each at its own drag\n(marker = where the design actually sits)",
                 fontsize=10.5)
    ax.legend(fontsize=7.2); ax.grid(alpha=0.25)

    fig.suptitle("Best drag available inside each planform constraint class — full OAS at MTOW 382 547 N, "
                 "span pinned at 118 ft, all trimmed to the same lift\n"
                 f"Arc A / B / C carry the {TOC_NOTE} profile; Plan L is the as-built loft",
                 fontsize=12)
    fig.text(0.5, 0.048, DEFINITIONS, ha="center", fontsize=10, fontweight="bold")
    fig.text(0.5, 0.020,
             "All percentages are against PLAN L AS-BUILT. Drag is NOT the merit function: the study ranks on electric range at fixed MTOW (m_batt/D), break-even "
             f"{BREAK_EVEN_LB_PER_N:.3f} lb of wing per newton,\nso 'may weigh' is how much heavier each architecture can be and still match Plan L on range -- and the bottom row now plots that range directly. "
             "Wing-only drag throughout. Depth and width use EACH design's own section retention.\nDesign points: Arc A = wing 8, Arc B = wing 7, Arc C = wing 3.",
             ha="center", fontsize=8.5, style="italic")

    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, "class_comparison.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\n  wrote {path}")

    for (nm, _, tag), (y, chord, depth, width) in zip(CLASSES, span):
        d_ail = float(np.interp(0.90 * 708.0, y, depth))
        d_jun = float(np.interp(674.9, y, depth))
        w_in = float(np.interp(176.0, y, width)); w_out = float(np.interp(356.0, y, width))
        print(f"  {nm.replace(chr(10), ' '):26} depth {d_ail:5.2f} in @ aileron, {d_jun:5.2f} in @ junction | "
              f"box {w_in:6.2f} in @ inboard nacelle (need 65), {w_out:6.2f} in @ outboard (need 55)")

    print("\n" + "=" * 104)
    print(f"{'architecture':26} {'drag N':>10} {'vs Plan L':>9} {'S_ref ft2':>10} "
          f"{'W_wing lb':>11} {'R nmi':>8} {'vs Plan L':>9} {'may weigh lb':>13}")
    for (nm, _, tag), r in zip(CLASSES, res):
        d = r["drag_N"] - ref
        w, R = r.get("w_wing_lb"), r.get("R_nmi")
        wtxt = f"{w:>10,.0f}" + ("" if r.get("w_converged") else "*") if w else f"{'UNSIZED':>11}"
        rtxt = f"{R:>8.1f}" if R else f"{'-':>8}"
        print(f"{nm + ' (' + tag + ')':26} {r['drag_N']:>10.1f} "
              f"{100*(r['drag_N']/ref-1):>+8.2f}% {r['S_ref']*10.7639104:>10.1f} "
              f"{wtxt:>11} {rtxt} {'':>9} {-BREAK_EVEN_LB_PER_N*d:>+13.1f}")
    print("=" * 104)
    # Range against the best, which is what a reader wants from a ranking.
    rk = sorted(((r.get("R_nmi"), nm) for (nm, _, _), r in zip(CLASSES, res)
                 if r.get("R_nmi")), reverse=True)
    if rk:
        top = rk[0][0]
        print("  ranking on electric range: " + ",  ".join(
            f"{nm} {R:.1f} nmi ({100*(R/top-1):+.2f}%)" for R, nm in rk))
        print("  * weight loop stopped at its pass limit, not the 25 lb tolerance.")
