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
# The wingbox ends at the winglet junction. Outboard of it there is wing but no
# box, so anything drawn or reported as structure has to stop here.
Y_JUNCTION_IN = 674.9
# Aileron actuator depth floor, matching arc_optimal_toc.DEPTH_REQ.
DEPTH_REQ_IN = 7.0
# The aileron station the depth floor is written at: 90 percent of the semi-span,
# matching arc_optimal_toc.Y_AIL. The spanwise-load panels mark it because it is
# where the stall margin matters most.
Y_AIL_IN = 0.90 * 708.0


def retention_fn(airfoil=None, blend=None):
    """Fraction of max thickness a section still has at x/c.

    Retention belongs to the SECTION, and it is what turns t/c into depth:
    depth = retention(spar) * t/c * chord. Using the as-built curve on a wing
    built with another section understates its depth -- measured, 6.65 in against
    the 7.65 in the design actually delivers, because e694 keeps 0.935 of its
    thickness at the 0.574c spar where the as-built keeps 0.814.
    """
    def _thk(name):
        if name in (None, "", "as-built"):
            af = asbuilt()
        else:
            import aerosandbox as asb
            af = asb.Airfoil(name)
        return np.array([float(af.local_thickness(x_over_c=float(x))) for x in xs])

    xs = np.linspace(0.05, 0.95, 300)
    # A design may carry a SPANWISE section, in which case retention is a function
    # of the station as well as the chord fraction. Returned with the same (y, spar)
    # signature either way so callers need no special case.
    if blend:
        t_in, t_out = _thk(blend["inboard"]), _thk(blend["outboard"])
        f0, f1 = float(blend["f_start"]), float(blend["f_end"])
        semi = 708.0

        def _ret(y_in, spar):
            w = float(np.clip((abs(y_in) / semi - f0) / (f1 - f0), 0.0, 1.0))
            t = (1.0 - w) * t_in + w * t_out
            return float(np.interp(spar, xs, t / t.max()))

        return _ret

    t = _thk(airfoil)
    return lambda y_in, spar: float(np.interp(spar, xs, t / t.max()))


# Where the box is reported. RegionPlanform emits `station_chord` at whatever
# stations it is given, computed analytically from the baseline chord and the
# scale factor -- so asking it for a dense vector gives the exact chord
# distribution. Interpolating the 35-node resampled mesh instead runs ~1% low
# through region B and ~5% low near the aileron, where the chord changes fastest
# and the spanwise stations are clustering into the winglet; that error reads as
# a violated constraint on a design the optimizer has satisfied.
REPORT_STATIONS_IN = np.linspace(0.0, 674.9, 80)


def rear_fraction(r, y_in, chord_in, schedule):
    """This design's aft-spar chord fraction at these stations.

    A straight aft spar is rear(y) = p + K/c(y), which is NOT linear in y, so
    interpolating a handful of stored knots bows it between them. When the design
    carries K the rule is used directly and the spar is straight everywhere, not
    just at the stations the knots were placed on.
    """
    K = r.get("K_in")
    if K is not None:
        p_ = float(r["wingbox_pct"])
        return np.array([p_ + float(K) / float(c) for c in chord_in])
    return np.array([float(rear_spar_fraction(v, schedule)) for v in y_in])


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
        return (y, chord, np.array([ret(yy, v) for yy, v in zip(y, sp)]) * toc0 * chord,
                (sp - w2.FRONT_PCT) * chord)
    m = r["mesh"] / config.SCALE
    y_mesh = np.abs(m[0, :, 1])
    yp = 0.5 * (y_mesh[:-1] + y_mesh[1:])
    toc = np.interp(y, yp, r["toc"])
    sp = rear_fraction(r, y, chord, schedule)
    depth = np.array([ret(yy, v) for yy, v in zip(y, sp)]) * toc * chord
    width = (sp - w2.FRONT_PCT) * chord
    return y, chord, depth, width


PLAN_L_COLOR = "#8C7B6B"
# Plan L's straight spar, least-squares fitted over regions A+B (README): the
# loft has no scheduled box, so this is what its aft spar actually is. The front
# spar is not measurable from the loft; the study's 0.12c is applied so the box
# width is comparable, and the panels say so.
PLAN_L_AFT_PCT = 0.60065


def baseline_case(name="plan_l", want_prob=False):
    """As-built baseline at MTOW: run, trimmed, never optimized.

    ``want_prob`` returns the built problem, so the reference can carry a spanwise
    load in the comparison alongside the three arcs.
    """
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
            STRAIGHT_LINE_KEY: float(prob.get_val("wing.wingbox_pct")[0]),
            **({"prob": prob} if want_prob else {})}


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
# Constructed design points per arc, in PREFERENCE ORDER. An arc letter names a
# CONSTRAINT CLASS, so two constructions of the same class share a letter and are
# told apart by how they were produced. Kept in step with export_dat.CONSTRUCTED.
CONSTRUCTED = {
    "A": ("arc_a_constfrac", "arc_a_constructed"),
}


def replay(case, y_a_in, rule, want_prob=False):
    """Rebuild a stored case and confirm it reproduces its logged drag.

    ``want_prob`` returns the built OpenMDAO problem alongside the results, for a
    caller that has to interrogate the model further -- a spanwise load at a
    different weight, for instance. It is an addition, not a change: the drag
    cross-check below still runs, so anything built this way is a design point that
    reproduced itself rather than a fresh guess at one.
    """
    schedule, stations, _ = stations_and_schedule()
    # A design may carry its own rear-spar schedule -- Arc A's straight aft spar is
    # rear(y) = p + K/c(y), which is neither the shared kink nor a constant -- and its
    # own SPANWISE section. Replaying either on the study defaults reproduces a
    # different wing, which is what the drag cross-check at the end of this function
    # exists to catch.
    if case.get("rear_schedule"):
        schedule = tuple((float(a_), float(b_)) for a_, b_ in case["rear_schedule"])
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
    blend = case.get("section_blend")
    if blend:
        # A spanwise section reaches OAS as a per-PANEL c_max_t. OAS takes an array
        # here with no change: a constant array reproduces the scalar exactly and the
        # analytic partials still match complex-step to 1e-19.
        import aerosandbox as asb
        _xs = np.linspace(0.05, 0.95, 300)

        def _t(nm):
            af = asbuilt() if nm in (None, "", "as-built") else asb.Airfoil(nm)
            return np.array([float(af.local_thickness(x_over_c=float(x))) for x in _xs])

        _ti, _to = _t(blend["inboard"]), _t(blend["outboard"])
        _f0, _f1 = float(blend["f_start"]), float(blend["f_end"])

        def _cmt(y_in):
            w = float(np.clip((abs(y_in) / 708.0 - _f0) / (_f1 - _f0), 0.0, 1.0))
            tt = (1.0 - w) * _ti + w * _to
            return float(_xs[int(np.argmax(tt))])

        def _surface(mesh_, stick_, regions_, **kw):
            sd = orig_build_surface(mesh_, stick_, regions_, **kw)
            ym = np.abs(np.asarray(mesh_)[0, :, 1]) / config.SCALE
            yp = 0.5 * (ym[:-1] + ym[1:])
            sd["c_max_t"] = np.array([_cmt(v) for v in yp])
            return sd
        ro.build_surface = _surface
    elif c_max_t is not None:
        def _surface(mesh_, stick_, regions_, **kw):
            sd = orig_build_surface(mesh_, stick_, regions_, **kw)
            sd["c_max_t"] = float(c_max_t)
            return sd
        ro.build_surface = _surface
    try:
        prob, _, _, regions, planform0 = w2.build(w2.BASELINE, y_a_in)
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
            # The box this wing was replayed with, so a caller that exports the
            # geometry can export the SPARS too rather than guess at them. A
            # design point written before these were recorded falls back to the
            # study schedule above, and that fallback is resolved here -- once --
            # instead of once per caller.
            "rear_schedule": tuple((float(a_), float(b_)) for a_, b_ in schedule),
            "front_pct": float(w2.FRONT_PCT),
            # How straight the BASELINE's own spar line is, inches. It is the floor
            # on the straightness of anything lofted from it: the parameterization
            # scales the baseline, so it preserves this departure rather than
            # removing it.
            "spar_max_dev_in": float(planform0["spar_max_dev"]),
            "y_c_start_in": float(regions.y_c_start),
            # retention belongs to the section, so the depth panel must use this
            "airfoil": case.get("airfoil"),
            STRAIGHT_LINE_KEY: float(prob.get_val("wing.wingbox_pct")[0]),
            **({"prob": prob, "regions": regions} if want_prob else {})}


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
        # A CONSTRUCTED design point is not optimized: its straight aft spar is a
        # geometric requirement and the design is built to satisfy it rather than
        # searched for. Preferred when present and feasible; a design that failed its
        # own constraints must never silently become the figure.
        #
        # Arc A has two constructions, in the same preference order export_dat uses --
        # constant fraction first. Duplicating the order rather than importing it is
        # deliberate: this script must not import export_dat, which imports this one.
        for stem in CONSTRUCTED.get(arc, ()):
            p_con = os.path.join(LOGS, f"{stem}_{TOC_PROFILE}{suffix}.json")
            if not os.path.exists(p_con):
                continue
            c = json.load(open(p_con))
            if c.get("feasible"):
                what = ("CONSTRUCTED straight aft spar at a constant fraction"
                        if c.get("constant_aft_fraction")
                        else "CONSTRUCTED straight aft spar")
                return c, (f"{ARC_AIRFOIL}->{c['section_blend']['outboard']}, "
                           f"{TOC_PROFILE} t/c, {what}")
            print(f"  NOTE: {os.path.basename(p_con)} is not feasible -- not used")
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
    # The region A rule belongs to the DESIGN POINT, not to the arc: Arc A has two
    # constructions solved on different rules, and replaying one on the other's rule
    # builds a different wing. It was hardcoded here, which the drag cross-check
    # inside replay() would have caught -- loudly, but only at run time.
    print("  replaying Arc C (free) ...")
    r_free = replay(cC, w2.REGION_A_END_IN,
                    cC.get("region_a_rule") or "root_le_fixed", want_prob=True)
    print("  replaying Arc B (straight front spar) ...")
    r_fwd = replay(cB, w2.REGION_A_END_IN,
                   cB.get("region_a_rule") or "preserved", want_prob=True)
    print("  replaying Arc A (constant chord) ...")
    r_cc = replay(cA, REGION_A_AS_BUILT_IN,
                  cA.get("region_a_rule") or "root_le_fixed", want_prob=True)
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
        # The design's OWN spar schedule and section have to travel with it, not
        # just be used inside replay(). Without these the panels fall back to the
        # SHARED 0.750 -> 0.550 kink and the single-section retention, which drew
        # Arc A's straight aft spar as a kinked one -- the model was right and the
        # picture was wrong, which is the worse way round.
        _r["rear_schedule"] = _c.get("rear_schedule")
        _r["section_blend"] = _c.get("section_blend")
        _r["constructed"] = bool(_c.get("constructed"))
        _r["constant_aft_fraction"] = bool(_c.get("constant_aft_fraction"))

    TOC_NOTE = provC
    print("  evaluating Plan L as-built (the reference) ...")
    r_pl = baseline_case("plan_l", want_prob=True)
    # Plan L LAST in the list so the three optimized classes keep their order and
    # colours, and the reference reads as a reference.
    res = [r_cc, r_fwd, r_free, r_pl]          # Arc A, Arc B, Arc C, reference
    ref = r_pl["drag_N"]                        # Plan L as-built: every % is against it

    # The spanwise load at the weight the aircraft is FLOWN at. Deferred import:
    # lift_distribution imports this module, so taking it at the top would be a cycle.
    # Every case is re-trimmed to the same mid-cruise weight, so the four curves are
    # compared at equal lift and differ only by planform.
    import lift_distribution as LD
    W_CRUISE_LB = LD.mission.cruise_weight_lb()
    print(f"  spanwise load at mid-cruise {W_CRUISE_LB:.0f} lb ...")
    for _r, _nm in zip(res, [c[0].replace(chr(10), " ") for c in CLASSES]):
        _d = LD.distribution(_r["prob"], W_CRUISE_LB)
        _ck = _d["checks"]
        _bad = [k for k in ("rel_vs_CL", "rel_vs_weight", "rel_cl_sectional")
                if _ck[k] > LD.TOL_REL]
        if _bad:
            raise RuntimeError(f"{_nm}: spanwise load fails {_bad}")
        _r["load"] = _d
        print(f"    {_nm:22s} alpha {_d['alpha_deg']:6.3f} deg  max cl "
              f"{_d['cl'][_d['y_in'] <= LD.Y_JUNCTION_IN].max():.4f}  centre of lift "
              f"{_d['y_cp_frac']:.4f} semi-span")

    schedule, stations, _ = stations_and_schedule()

    def sched_of(r):
        """This case's OWN rear-spar schedule.

        Arc A holds a constant 0.750c aft spar so that its aft spar is straight;
        B and C carry the shared 0.750 -> 0.550 kink. Drawing all three on one
        schedule would put Arc A's depth, box width and spar ratio on a spar it
        does not have.
        """
        sc = r.get("rear_schedule")
        return tuple((float(a_), float(b_)) for a_, b_ in sc) if sc else schedule

    # one retention curve per case, from that design's own section
    rets = [retention_fn(r.get("airfoil"), r.get("section_blend")) for r in res]
    span = [spanwise(r, sched_of(r), rt) for r, rt in zip(res, rets)]

    fig = plt.figure(figsize=(16.5, 26.6))
    gs = fig.add_gridspec(7, 3, height_ratios=[1.0, 1.2, 1.0, 1.0, 1.0, 1.15, 1.0],
                          hspace=0.42, wspace=0.30, bottom=0.088)
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
    ax.axvline(Y_JUNCTION_IN, color="#0B7A75", ls="--", lw=1.1)
    ax.text(Y_JUNCTION_IN, ax.get_ylim()[0], " winglet junction", color="#0B7A75",
            fontsize=8, va="bottom", ha="right")
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
    # EACH design's own aft spar. B and C share the 0.750c-to-0.550c kink and so
    # coincide; Arc A's straight aft spar goes the OTHER WAY -- constant x means a
    # RISING chord fraction as the chord shrinks, 0.750c to 0.804c -- and drawing it
    # on the shared schedule showed a spar it does not have.
    dashes = [(1, 0), (6, 3), (2, 2.5), (5, 2)]
    for (nm, c, tag), dash, (r, sp) in zip(CLASSES, dashes, zip(res, span)):
        if tag == "reference":
            ax.plot(yy, np.full_like(yy, PLAN_L_AFT_PCT), color=c, lw=2.0, dashes=dash,
                    label=nm)
        else:
            rear = rear_fraction(r, sp[0], sp[1], sched_of(r))
            ax.plot(sp[0], rear, color=c, lw=1.9, dashes=dash,
                    label=nm + (" (straight)" if r.get("K_in") is not None else ""))
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
    ax.axvline(Y_JUNCTION_IN, color="#0B7A75", ls="--", lw=1.1)
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
    ax.axhline(DEPTH_REQ_IN, color="#C44E52", ls="--", lw=1.2)
    ax.text(5, DEPTH_REQ_IN, f" {DEPTH_REQ_IN:.0f} in required at the aileron",
            color="#C44E52", fontsize=7.5, va="bottom")
    ax.set_ylim(bottom=min(DEPTH_REQ_IN - 0.6, ax.get_ylim()[0]))
    # Only the winglet junction is marked, here as everywhere: it is where the box
    # ends, which is a boundary every spanwise panel shares. The aileron station is
    # still identified -- by the markers below, which sit on it -- so it does not
    # need a rule of its own.
    ax.axvline(Y_JUNCTION_IN, color="#0B7A75", ls="--", lw=1.1)
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

    # ================= THE WINGBOX IN PLAN =================
    # The panel that makes the architectures legible. Every arc carries the SAME
    # rear-spar RATIO schedule (0.750 at y=356 -> 0.550 at y=674.9) and the same
    # 0.12c front spar -- those are study-wide inputs, not outputs of the
    # architecture -- so a sloping aft-spar RATIO is not something an arc chose.
    # What separates them is which chord fraction is held STRAIGHT, `wingbox_pct`,
    # drawn dashed. Arc B pins it at the front spar, so its front spar is dead
    # straight and the box narrows from the back; Arc A and Arc C let it optimize
    # near the aft spar, so their aft spar is the straight one and the box loses
    # its front. Plan L is left out: its spar is a least-squares fit to the
    # as-built loft, not a scheduled box, so it has no comparable straight line.
    for col, ((nm, c, tag), r) in enumerate(zip(CLASSES, res)):
        if tag == "reference":
            continue
        ax = fig.add_subplot(gs[5, col])
        m = r["mesh"] / config.SCALE
        y_full = np.abs(m[0, :, 1])
        # The box ENDS at the winglet junction -- there is no wingbox outboard of
        # it, so drawing spars across the winglet would invent structure. The LE
        # and TE are still drawn to the tip, because the wing is there.
        keep = y_full <= Y_JUNCTION_IN + 1e-6
        y = y_full[keep]
        x_le, x_te = m[0, keep, 0], m[-1, keep, 0]
        chord = x_te - x_le
        pct = float(r[STRAIGHT_LINE_KEY])
        sp_aft = rear_fraction(r, y, chord, sched_of(r))
        x_fwd = x_le + w2.FRONT_PCT * chord
        x_aft = x_le + sp_aft * chord

        # LE/TE stop at the junction too: outboard of it is winglet, and drawing it
        # here only adds a steep curve that reads as a boundary of the box.
        ax.plot(y, x_le, color="0.55", lw=1.2)
        ax.plot(y, x_te, color="0.55", lw=1.2, label="LE / TE")
        # Grey fill on every panel: the box is the same object in all three, and a
        # per-arc colour here would imply the fill means something it does not.
        # No straight `wingbox_pct` line either -- it is a construction fraction,
        # not a member, and a phantom line inside a structural plot reads as one.
        # Where an arc holds a spar straight, that spar is visibly straight.
        ax.fill_between(y, x_fwd, x_aft, color="0.55", alpha=0.25, lw=0)
        ax.plot(y, x_fwd, color="#2B6CB0", lw=2.0, label=f"fwd spar {w2.FRONT_PCT:.2f}c")
        _sc = sched_of(r)
        _lab = (f"aft spar {_sc[0][1]:.3f}c" if abs(_sc[0][1] - _sc[-1][1]) < 1e-9
                else f"aft spar {_sc[0][1]:.3f}c→{_sc[-1][1]:.3f}c")
        ax.plot(y, x_aft, color="#C44E52", lw=2.0, label=_lab)
        for ys_, lab in NACELLES.items():
            ax.axvline(ys_, color="0.45", ls=":", lw=0.9)
            if col == 0:
                ax.annotate(lab, xy=(ys_, 0.02), xycoords=("data", "axes fraction"),
                            rotation=90, fontsize=6.2, color="0.35",
                            ha="right", va="bottom")
        ax.axvline(Y_JUNCTION_IN, color="#0B7A75", lw=1.4)
        ax.annotate("winglet junction", xy=(Y_JUNCTION_IN, x_aft[-1]),
                    xytext=(-8, 22), textcoords="offset points", fontsize=6.8,
                    color="#0B7A75", ha="right")
        ax.set_xlim(-15, Y_JUNCTION_IN + 15)
        ax.invert_yaxis()                       # x down, as the planform panel does
        ax.set_xlabel("y, in")
        if col == 0:
            ax.set_ylabel("x, in")
        if r.get("K_in") is not None:
            held = f"STRAIGHT aft spar (x varies {np.ptp(x_aft):.2f} in)"
        elif abs(pct - w2.FRONT_PCT) < 1e-6:
            held = "front spar held straight"
        else:
            held = f"straight line at {pct:.3f}c, aft side"
        ax.set_title(f"{nm} wingbox in plan — {held}", fontsize=10)
        ax.legend(fontsize=6.8, loc="upper left"); ax.grid(alpha=0.22)

    # ================= THE SPANWISE LOAD =================
    # Every other panel is an integral -- drag, area, weight. A spar is sized by the
    # load ALONG the span, so the row that a structures reviewer reads first is this
    # one. All four are re-trimmed to the SAME mid-cruise weight, so the curves are
    # compared at equal lift and differ only by planform.
    #
    # Wing rows only. The model carries the winglet IN THE PLANE OF THE WING, so the
    # rows outboard of the junction are a flattened winglet, not wing, and plotting
    # them would put a curve on the chart that no wing has.
    def _wing(r):
        d = r["load"]
        k = d["y_in"] <= Y_JUNCTION_IN
        return d, k

    ax = fig.add_subplot(gs[6, 0])
    for (nm, c, tag), dash, r in zip(CLASSES, dashes, res):
        d, k = _wing(r)
        ax.plot(d["y_in"][k], d["cl"][k], color=c, lw=1.9, dashes=dash, label=nm)
    for ys_ in (REGION_A_AS_BUILT_IN, Y_AIL_IN):
        ax.axvline(ys_, color="0.45", ls=":", lw=1.0)
    ax.set_xlabel("y, in"); ax.set_ylabel("sectional $c_l$")
    ax.set_title(f"Sectional lift coefficient at {W_CRUISE_LB:,.0f} lb\n"
                 f"(the local chord normalizes it, not the wing area)", fontsize=10.5)
    ax.legend(fontsize=7.2); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[6, 1])
    for (nm, c, tag), dash, r in zip(CLASSES, dashes, res):
        d, k = _wing(r)
        ax.plot(d["y_in"][k], d["lift_lb_per_in"][k], color=c, lw=1.9, dashes=dash,
                label=nm)
    for ys_ in (REGION_A_AS_BUILT_IN, Y_AIL_IN):
        ax.axvline(ys_, color="0.45", ls=":", lw=1.0)
    ax.set_xlabel("y, in"); ax.set_ylabel("running load, lb/in")
    ax.set_title(f"Running load at {W_CRUISE_LB:,.0f} lb, 1 g\n"
                 f"(no gust and no manoeuvre factor)", fontsize=10.5)
    ax.legend(fontsize=7.2); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[6, 2])
    for (nm, c, tag), dash, r in zip(CLASSES, dashes, res):
        d, k = _wing(r)
        ax.plot(d["y_in"][k], d["lift_N_per_in"][k] / d["elliptical_N_per_in"][k],
                color=c, lw=1.9, dashes=dash, label=nm)
    ax.axhline(1.0, color="0.35", lw=1.0)
    ax.set_xlabel("y, in"); ax.set_ylabel("load / elliptical")
    ax.set_title("Load against the elliptical reference\n"
                 "(same total lift on the same semi-span)", fontsize=10.5)
    ax.legend(fontsize=7.2); ax.grid(alpha=0.25)
    ax.text(0.02, 0.03,
            "wing rows only: this model lays the winglet\nin the wing plane",
            transform=ax.transAxes, fontsize=7.0, color="0.35", va="bottom")

    fig.suptitle("Best drag available inside each planform constraint class — full OAS at MTOW 382 547 N, "
                 "span pinned at 118 ft, all trimmed to the same lift\n"
                 f"Arc A / B / C carry the {TOC_NOTE} profile; Plan L is the as-built loft",
                 fontsize=12)
    _con = [(nm, r) for (nm, _c, _t), r in zip(CLASSES, res) if r.get("constructed")]
    _what = ", ".join(
        f"{nm} CONSTRUCTED to a straight aft spar"
        + (" at a constant chord fraction" if r.get("constant_aft_fraction") else "")
        for nm, r in _con)
    _note = DEFINITIONS + (
        f"    |    {_what}, not drag-optimized -- its drag is feasible, "
        f"not best-in-class." if _con else "")
    fig.text(0.5, 0.058, _note, ha="center", fontsize=10, fontweight="bold")
    fig.text(0.5, 0.028,
             "All percentages are against PLAN L AS-BUILT. Drag is NOT the merit function: the study ranks on electric range at fixed MTOW (m_batt/D), break-even "
             f"{BREAK_EVEN_LB_PER_N:.3f} lb of wing per newton,\nso 'may weigh' is how much heavier each architecture can be and still match Plan L on range -- and the range-against-weight panel plots that directly. "
             "Wing-only drag throughout. Depth and width use EACH design's own section retention.\n"
             f"The bottom row is the spanwise load, re-trimmed to mid-cruise {W_CRUISE_LB:,.0f} lb at 1 g -- unfactored, so it is not a limit load.\n"
             "Design points: Arc A = wing 8, Arc B = wing 7, Arc C = wing 3.",
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
