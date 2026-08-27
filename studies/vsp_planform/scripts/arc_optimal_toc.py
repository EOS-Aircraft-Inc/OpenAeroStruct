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
DEPTH_REQ = 6.0
SCHEDULE = ((356.0, 0.750), (674.9, 0.550))
TOL_IN, MAX_PASS = 0.005, 5
W_SEED_LB, W_PASSES, W_TOL_LB = 8000.0, 4, 25.0

PROFILES = {                    # (root t/c, tip/root) -> the 5 spline control points
    "optimal": (0.250, 0.58),   # the sweep's peak, 344.0 nmi
    "capped":  (0.220, 0.75),   # inside conventional thickness, 343.0 nmi
}
ARCHS = {                       # region A end, region A rule, pinned straight line
    "A": (REGION_A_AS_BUILT_IN, "root_le_fixed", None),
    "B": (w2.REGION_A_END_IN, "preserved", w2.FRONT_PCT),
    "C": (w2.REGION_A_END_IN, "root_le_fixed", None),
}


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


SPAR_AIL = float(rear_spar_fraction(Y_AIL, SCHEDULE))
RET, C_MAX_T = section()          # replaced at run time by --airfoil


def optimize(y_a_in, rule, pin_p, cp_toc, c_req):
    stations = ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0),
                (Y_AIL, (SPAR_AIL - w2.FRONT_PCT) * c_req),
                (674.9, w2.JUNCTION_BOX_IN))
    w2.REAR_SCHEDULE, w2.WIDTH_STATIONS = SCHEDULE, stations
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = SCHEDULE
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
            sd["c_max_t"] = C_MAX_T
            return sd

        ro.build_surface = _surface
        try:
            prob, _ = ro.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
        finally:
            ro.build_surface = orig_build_surface
        n_cp = int(np.asarray(prob.get_val("wing.t_over_c_cp")).size)
        cp = np.linspace(cp_toc[0], cp_toc[0] * cp_toc[1], n_cp)
        prob.set_val("wing.t_over_c_cp", cp)
        prob.run_model()
        s0 = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
        alpha0 = trim_alpha(prob, w2.W / (q * s0))
        ro.add_optimization(prob, "plan_l", mesh, planform0, s0, mode="fixed_lift", weight=w2.W)
        prob.set_val("wing.t_over_c_cp", cp)          # setup() reset it
        if pin_p is not None:
            prob.set_val("wing.wingbox_pct", pin_p)
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
    r["depth_delivered_in"] = RET(SPAR_AIL) * r["toc_delivered_ail"] * r["chord_at_aileron_in"]
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
    y_a, rule, pin_p = ARCHS[arc]
    cp_toc = PROFILES[profile]
    label = f"arc {arc} / {profile}"

    # --- depth: c_req from the DELIVERED t/c, iterated
    toc_use = None
    for p in range(1, MAX_PASS + 1):
        if toc_use is None:                     # seed from the baseline loft
            w2.apply_wing2_box()
            _, stick0, _, _ = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
            ys = np.abs(np.asarray(stick0.le[:, 1], dtype=float))
            toc_use = float(np.interp(Y_AIL, ys, stick0.toc))
        c_req = DEPTH_REQ / (RET(SPAR_AIL) * toc_use)
        r = optimize(y_a, rule, pin_p, cp_toc, c_req)
        err = r["depth_delivered_in"] - DEPTH_REQ
        print(f"  {label} depth pass {p}: c_req {c_req:6.2f} -> depth "
              f"{r['depth_delivered_in']:5.2f} in ({err:+.3f}), drag {r['drag_N']:9.1f} N", flush=True)
        # The requirement is a FLOOR. Over-delivering is free -- it happens when
        # the aileron width constraint stops binding and the chord is set
        # elsewhere -- so anything at or above 6 in is done. Only a shortfall
        # needs another pass, and only a shortfall can be fixed by more chord.
        if err >= -TOL_IN:
            break
        toc_use = r["toc_delivered_ail"]

    # --- weight: bi-level fixed point, damped
    prob = r.pop("_prob"); toc_full = r.pop("_toc_full")
    comp = list(lifting_surfaces(read_degen_csv(config.BASELINES[w2.BASELINE])).values())[0][0]
    oas = {"mesh": np.asarray(prob.get_val("wing.mesh", units="m")), "toc": toc_full,
           "plate": comp.plate, "stick": comp.stick, "y_junction": 674.9}
    tag = f"{arc}_{profile}"
    # The sizer places the access cut-out from ONE wing-wide stringer pair
    # (default / alternative) in planformIn.csv, but each bay carries a different
    # stringer range -- bay 16 carries Stg 1..6, bay 19 carries Stg 6..9 -- so no
    # single pair suits every bay on every planform, and the run stops with
    # "bay N has nowhere to put the access cut-out". That is a deck-configuration
    # limit, not a property of the design, so it must not cost the aero result:
    # the weight is recorded as unavailable and the t/c, depth and drag stand.
    w, hist, sizing_error = W_SEED_LB, [], None
    for p in range(1, W_PASSES + 1):
        print(f"  {label} weight pass {p}: W_in {w:.1f} lb", flush=True)
        try:
            wcdeck.write_deck(wcdeck.WC_DECK, Path(LOGS) / f"deck_arc{tag}", mission.MTOW_LB, w, oas=oas)
            w_new = wcdeck.run_wingcalc(Path(LOGS) / f"deck_arc{tag}", Path(LOGS) / f"wc_arc{tag}")
        except Exception as exc:
            sizing_error = f"{type(exc).__name__}: {exc}"
            print(f"  !!! {label} sizing FAILED: {sizing_error}", flush=True)
            break
        hist.append({"pass": p, "w_in_lb": w, "w_wing_lb": w_new, "residual_lb": w_new - w})
        print(f"  >>> {label} p{p}: {w:.1f} -> {w_new:.1f} lb ({w_new - w:+.1f})", flush=True)
        if abs(w_new - w) < W_TOL_LB:
            break
        w += 0.5 * (w_new - w)

    r["sizing_error"] = sizing_error
    r["w_wing_lb"] = hist[-1]["w_wing_lb"] if hist else None
    r["batt_lb"] = mission.battery_lb(r["w_wing_lb"]) if hist else None
    r["R_nmi"] = mission.electric_range_nmi(r["w_wing_lb"], r["drag_N"]) if hist else None
    r["weight_history"] = hist
    r["converged"] = bool(hist) and abs(hist[-1]["residual_lb"]) < W_TOL_LB
    r["arc"], r["profile"] = arc, profile
    r["root_toc_req"], r["ratio_req"] = cp_toc
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc", required=True, choices=sorted(ARCHS))
    ap.add_argument("--profile", required=True, choices=sorted(PROFILES))
    ap.add_argument("--airfoil", default="as-built",
                    help="database section name, or 'as-built' (default)")
    a = ap.parse_args()

    RET, C_MAX_T = section(a.airfoil)
    print(f"section {a.airfoil}: c_max_t {C_MAX_T:.4f}, retention at {SPAR_AIL:.4f}c "
          f"= {RET(SPAR_AIL):.4f}", flush=True)
    res = solve(a.arc, a.profile)
    res["airfoil"] = a.airfoil
    res["c_max_t"] = C_MAX_T
    res["retention_at_spar"] = RET(SPAR_AIL)
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
