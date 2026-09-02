"""Construct Arc A with a STRAIGHT aft spar, rather than optimizing it.

WHY CONSTRUCT RATHER THAN OPTIMIZE. The OAS run minimizes drag so the class figure
can claim "best drag available inside each constraint class". A geometry handed to a
structures reviewer does not need that -- it needs to RESPECT ITS CONSTRAINTS.
Optimality is a property of the comparison; feasibility is a property of the wing.
So this takes the twist, taper and alpha Arc A already converged to, applies the
straight aft spar and the spanwise section, and solves ONE unknown -- taper_B, on the
one requirement that binds. A 1-D root find cannot fail the way SLSQP did.

WHAT A STRAIGHT AFT SPAR ACTUALLY IS. ``wingbox_pct`` (p) is NOT the aft spar.
param.py's own component says so: "the straight line inside the box, not an edge of
it". The spar is the box's aft EDGE, at the scheduled fraction, and it sits at

    x_aft(y) = x_spar + (rear(y) - p) * c(y)

so it is straight -- constant x -- exactly when

    rear(y) = p + K / c(y)          for a constant offset K

``rear == p`` is only the K = 0 case, and that case is INFEASIBLE here: it pins
p * c = 71.49 in, capping the box at p*c*(1 - 0.12/p) <= 62.91 in against the 65 in
the nacelles need. Measured both ways -- constructing gave 59.9 in, and the optimizer
moved from "positive directional derivative" (stuck) to an iteration limit (no
feasible point) once the degenerate design variable was removed.

K is fixed by the nacelle requirement:

    box(nacelle) = (p - 0.12) * c + K = 65.00 in

with p left at its baseline value so region A's chord stays 103.38 in -- which is
also what "constant wing and wingbox chord to the outboard nacelle" asks for.

THE COST, AND IT IS INHERENT. A constant-x spar is a LARGER chord fraction where the
chord is smaller, so in section terms it marches aft going outboard -- precisely
where the 7 in depth is required. That is why the original schedule kinked FORWARD
(0.750 -> 0.550): it was buying retention where the chord runs out. Holding the spar
straight gives that up, and the only way back is chord, carried by taper_B.

Reports every constraint with its margin, and drag against the optimized Arc A so
the price of constructing rather than optimizing is visible.
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

from studies.vsp_planform import config, param                       # noqa: E402
from studies.vsp_planform.degen_csv import read_degen_csv, lifting_surfaces  # noqa: E402
from studies.vsp_planform.coupling import deck as wcdeck              # noqa: E402
from studies.vsp_planform.coupling import mission                     # noqa: E402
import studies.vsp_planform.run_opt as ro                            # noqa: E402
from studies.vsp_planform.run_opt import POINT, trim_alpha           # noqa: E402
import wing2_oas as w2                                               # noqa: E402
import arc_optimal_toc as A                                          # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")
q = 0.5 * config.RHO * config.V_MS**2
M2_FT2 = 10.7639104

# Constraint stations and required box widths, inches. The aileron entry is the depth
# requirement re-encoded as a width, as aileron_90.py does.
BASE_STATIONS = ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0))


def evaluate(taper_B, case, cmt_at, p, schedule, stations, rule="root_le_fixed"):
    """One model evaluation.

    The chord is independent of the rear schedule (param.py: ``c_w`` depends on p and
    taper only), so station chords read off a first pass are exact for building the
    straight-spar schedule on a second pass.

    ``rule`` is region A's rule. This construction needs ``root_le_fixed``, which is
    what both baselines ship as. Arc A's constant-fraction construction needs
    ``preserved``, so the parameter exists rather than the constant it replaced --
    see arc_a_constfrac.py, where the rule is the whole difference between the two.
    """
    w2.REAR_SCHEDULE, w2.WIDTH_STATIONS = schedule, stations
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = stations
    saved = param.REGION_A_RULE[w2.BASELINE]
    param.REGION_A_RULE[w2.BASELINE] = rule
    orig = ro.build_surface

    def _surface(mesh_, stick_, regions_, **kw):
        sd = orig(mesh_, stick_, regions_, **kw)
        ym = np.abs(np.asarray(mesh_)[0, :, 1]) / config.SCALE
        yp = 0.5 * (ym[:-1] + ym[1:])
        sd["c_max_t"] = np.array([cmt_at(v) for v in yp])
        return sd

    ro.build_surface = _surface
    try:
        mesh, stick, regions, planform0 = w2.load_relofted(
            w2.BASELINE, A.REGION_A_AS_BUILT_IN)
        prob, _ = ro.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
    finally:
        ro.build_surface = orig
        param.REGION_A_RULE[w2.BASELINE] = saved

    n_cp = int(np.asarray(prob.get_val("wing.t_over_c_cp")).size)
    cp = np.linspace(case["_cp0"], case["_cp0"] * case["_cpr"], n_cp)
    prob.set_val("wing.t_over_c_cp", cp)
    prob.set_val("wing.wingbox_pct", p)
    prob.set_val("wing.taper_B", taper_B)
    prob.set_val("wing.twist_cp", np.array(case["twist_cp"]), units="deg")
    prob.set_val("alpha", case["alpha"], units="deg")
    prob.run_model()
    s_ref = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    trim_alpha(prob, w2.W / (q * s_ref))          # same lift as every other case

    r = w2.evaluate(prob, regions.y_c_start)
    toc = np.asarray(prob.get_val("wing.t_over_c")).ravel()
    m = np.asarray(prob.get_val("wing.mesh", units="m")) / config.SCALE
    ym = np.abs(m[0, :, 1]); yp = 0.5 * (ym[:-1] + ym[1:])
    r["toc_ail"] = float(np.interp(A.Y_AIL, yp, toc))
    r["station_chord_in"] = np.asarray(prob.get_val("station_chord", units="m")) / config.SCALE
    r["box_width_in"] = np.asarray(prob.get_val("wingbox_width", units="m")) / config.SCALE
    r["taper_B"] = float(prob.get_val("wing.taper_B")[0])
    r["wingbox_pct"] = float(prob.get_val("wing.wingbox_pct")[0])
    r["alpha"] = float(prob.get_val("alpha", units="deg")[0])
    r["twist_cp"] = prob.get_val("wing.twist_cp", units="deg").tolist()
    r["t_over_c_cp"] = cp.tolist()
    r["toc_full"] = toc
    r["mesh"] = np.asarray(prob.get_val("wing.mesh", units="m"))
    return r


PROBE_SCHED = ((356.0, 0.750), (674.9, 0.750))
PROBE_ST = BASE_STATIONS + ((A.Y_AIL, 30.0), (674.9, w2.JUNCTION_BOX_IN))
# The 7 in depth holds from ROOT TO AILERON, so it has to be checked along the span.
# Constraining only the endpoint let Arc A dip to 5.94 in at y = 595 while the aileron
# read exactly 7.00. RegionPlanform emits station_chord wherever it is asked, so these
# are appended as pure REPORTING stations -- zero required width, nothing constrained.
REPORT_Y = tuple(float(v) for v in np.linspace(40.0, A.Y_AIL, 40))
REPORT_ST = tuple((y, 0.0) for y in REPORT_Y)


def pass_at(taper_B, case, ret_at, cmt_at, p, K):
    """Evaluate at this taper with the straight-spar schedule that K implies."""
    r0 = evaluate(taper_B, case, cmt_at, p, PROBE_SCHED, PROBE_ST)
    chords = np.asarray(r0["station_chord_in"], dtype=float)
    ys = [y for y, _ in PROBE_ST]

    # rear(y) = p + K/c(y)  ->  the aft spar is straight at x_spar + K.
    rear = [p + K / c for c in chords]
    sched = tuple((float(y), float(v)) for y, v in zip(ys, rear))
    spar_ail = float(rear[3])
    ret_ail = ret_at(A.Y_AIL, spar_ail)
    c_req = A.DEPTH_REQ / (ret_ail * r0["toc_ail"])
    st = (BASE_STATIONS + ((A.Y_AIL, (spar_ail - w2.FRONT_PCT) * c_req),
                          (674.9, w2.JUNCTION_BOX_IN)) + REPORT_ST)

    r = evaluate(taper_B, case, cmt_at, p, sched, st)
    n_con = len(BASE_STATIONS) + 2
    r["stations"] = ys
    r["width_req_in"] = [v for _, v in st[:n_con]]
    r["box_width_in"] = np.asarray(r["box_width_in"])[:n_con]

    # Depth ALONG the span, from the model's own chord at the reporting stations.
    ch_rep = np.asarray(r["station_chord_in"], dtype=float)[n_con:]
    toc_full = np.asarray(r["toc_full"], dtype=float)
    ymesh = np.abs(np.asarray(r["mesh"])[0, :, 1]) / config.SCALE
    ypan = 0.5 * (ymesh[:-1] + ymesh[1:])
    dep_span, rear_span = [], []
    for y, c in zip(REPORT_Y, ch_rep):
        rr = p + K / c
        rear_span.append(rr)
        dep_span.append(ret_at(y, rr) * float(np.interp(y, ypan, toc_full)) * c)
    dep_span = np.array(dep_span)
    r["depth_span_in"] = dep_span
    r["depth_span_y"] = list(REPORT_Y)
    r["depth_min_in"] = float(dep_span.min())
    r["depth_min_y"] = float(REPORT_Y[int(dep_span.argmin())])
    r["station_chord_in"] = np.asarray(r["station_chord_in"])[:n_con]
    r["rear_schedule"] = [[float(y), float(v)] for y, v in sched]
    r["spar_at_aileron"] = spar_ail
    r["retention_at_spar"] = ret_ail
    r["depth_in"] = ret_ail * r["toc_ail"] * float(r["station_chord_in"][3])
    r["K_in"] = float(K)
    # Straightness check: (rear - p) * c is the spar's offset from the straight
    # construction line, so it must be the same K at every station.
    ch = np.asarray(r["station_chord_in"], dtype=float)
    r["x_aft_spread_in"] = float(np.ptp([(v - p) * c for v, c in zip(rear, ch)]))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="optimal", choices=sorted(A.PROFILES))
    ap.add_argument("--airfoil", default="e694")
    ap.add_argument("--tol", type=float, default=0.02, help="depth tolerance, in")
    a = ap.parse_args()

    cp_toc = A.PROFILES[a.profile]
    A.RET, A.C_MAX_T = A.section(a.airfoil)
    n_in, n_out, f0, f1 = A.SECTION_BLEND["A"]
    ret_at, cmt_at, _ = A.blended_section(n_in, n_out, f0, f1)

    src = os.path.join(LOGS, f"arc_optimal_toc_A_{a.profile}_{a.airfoil}.json")
    case = json.load(open(src))
    case["_cp0"], case["_cpr"] = cp_toc
    p = float(case["wingbox_pct"])

    print(f"seeded from {os.path.basename(src)}")
    print(f"  twist_cp {[round(v, 2) for v in case['twist_cp']]}, "
          f"alpha {case['alpha']:.3f} deg, taper_B {case['taper_B']:.4f}")
    print(f"  wingbox_pct held at its baseline {p:.4f}, so region A's chord is unchanged")
    print(f"  section {n_in} -> {n_out} over {f0*A.SEMI_IN:.0f}-{f1*A.SEMI_IN:.0f} in")
    print(f"  depth floor {A.DEPTH_REQ:.2f} in at y = {A.Y_AIL:.1f} in\n")

    # K is the constant offset that makes the spar straight, and it is set by
    # whichever width station binds HARDEST -- not by a station chosen in advance.
    # box(y) = (p - 0.12) * c(y) + K, so the requirement at station i needs
    # K >= req_i - (p - 0.12) * c_i, and K is the maximum of those. Calibrating on
    # one nacelle instead missed y = 100 by 0.05 in, because its chord is fractionally
    # smaller.
    probe = evaluate(float(case["taper_B"]), case, cmt_at, p, PROBE_SCHED, PROBE_ST)
    ch0 = np.asarray(probe["station_chord_in"], dtype=float)
    reqs = [v for _, v in BASE_STATIONS]
    needs = [(rq - (p - w2.FRONT_PCT) * c, y, c, rq)
             for (y, rq), c, rq2 in zip(BASE_STATIONS, ch0[:len(BASE_STATIONS)], reqs)
             for rq in (rq2,)]
    K = max(n[0] for n in needs)
    drv = max(needs, key=lambda n: n[0])
    print(f"straight aft spar: rear(y) = {p:.4f} + K/c(y)")
    for need, y, c, rq in needs:
        mark = "  <- sets K" if abs(need - K) < 1e-12 else ""
        print(f"  y {y:6.1f}  chord {c:7.2f} in  needs K >= {need:5.2f} in "
              f"for {rq:.0f} in of box{mark}")
    print(f"  K = {K:.2f} in: the spar sits that far aft of the straight construction "
          f"line, at constant x\n")

    t0 = float(case["taper_B"])
    r = pass_at(t0, case, ret_at, cmt_at, p, K)

    def show(t, res):
        print(f"  taper_B {t:.4f} -> MIN depth {res['depth_min_in']:5.2f} in at y "
              f"{res['depth_min_y']:5.0f} ({res['depth_min_in']-A.DEPTH_REQ:+.3f}), "
              f"aileron {res['depth_in']:5.2f}, c_ail "
              f"{float(res['station_chord_in'][3]):5.1f} in, drag {res['drag_N']:9.1f} N, "
              f"S_ref {res['S_ref']*M2_FT2:7.1f} ft2", flush=True)

    show(t0, r)
    # The requirement is a FLOOR over the whole root-to-aileron span, so the MINIMUM
    # is what has to clear it -- not the value at the aileron.
    best, d0 = r, r["depth_min_in"]
    if d0 < A.DEPTH_REQ - a.tol:
        t1 = min(1.0, t0 * A.DEPTH_REQ / max(d0, 1e-6))
        for _ in range(8):
            r1 = pass_at(t1, case, ret_at, cmt_at, p, K)
            show(t1, r1)
            best = r1
            if A.DEPTH_REQ <= r1["depth_min_in"] <= A.DEPTH_REQ + a.tol:
                break
            d1 = r1["depth_min_in"]
            if abs(d1 - d0) < 1e-9:
                break
            t_new = t1 + (A.DEPTH_REQ - d1) * (t1 - t0) / (d1 - d0)
            t0, d0 = t1, d1
            t1 = float(np.clip(t_new, 0.05, 1.0))

    print()
    print("CONSTRAINTS")
    viol = []
    for y, req, got in zip(best["stations"], best["width_req_in"], best["box_width_in"]):
        tag = "<- aileron depth, re-encoded" if abs(y - A.Y_AIL) < 1e-6 else ""
        ok = got >= req - 1e-6
        if not ok:
            viol.append(round(y, 1))
        print(f"  y {y:7.1f}  box {got:6.2f} in  need {req:6.2f}  "
              f"{'ok      ' if ok else 'VIOLATED'}  {tag}")
    dep_ok = best["depth_min_in"] >= A.DEPTH_REQ - a.tol
    print(f"  depth, MINIMUM root->aileron {best['depth_min_in']:5.2f} in at y "
          f"{best['depth_min_y']:.0f}   need {A.DEPTH_REQ:.2f}  "
          f"{'ok' if dep_ok else 'VIOLATED'}")
    print(f"  depth at the aileron itself  {best['depth_in']:5.2f} in")
    print(f"  aft spar straightness: the spar's offset from the construction line "
          f"varies by {best['x_aft_spread_in']:.4f} in (0 = perfectly straight)")

    print()
    print("RESULT")
    print(f"  taper_B         {best['taper_B']:.4f}   (seed {case['taper_B']:.4f})")
    print(f"  wingbox_pct     {best['wingbox_pct']:.4f}   held, not a design variable")
    print(f"  spar at aileron {best['spar_at_aileron']:.4f}c   retention "
          f"{best['retention_at_spar']:.4f}")
    print(f"  S_ref          {best['S_ref']*M2_FT2:8.1f} ft2  (optimized Arc A "
          f"{case['S_ref']*M2_FT2:.1f})")
    print(f"  drag           {best['drag_N']:8.1f} N    (optimized Arc A on the kinked "
          f"0.574c spar {case['drag_N']:.1f}, {100*(best['drag_N']/case['drag_N']-1):+.2f}%)")
    if viol or not dep_ok:
        print(f"  NOT FEASIBLE -- width violated at y = {viol}" if viol
              else "  NOT FEASIBLE -- depth short")

    # ---- weight, so this design can stand beside B and C on the merit function ----
    # Same damped bi-level fixed point arc_optimal_toc uses, and the same deck.
    w_wing = batt = rng = None
    hist, sizing_error = [], None
    if not (viol or not dep_ok):
        comp = list(lifting_surfaces(read_degen_csv(
            config.BASELINES[w2.BASELINE])).values())[0][0]
        oas = {"mesh": best["mesh"], "toc": best["toc_full"],
               "plate": comp.plate, "stick": comp.stick, "y_junction": 674.9}
        w = A.W_SEED_LB
        passes = int(os.environ.get("ARC_W_PASSES", "8"))
        from pathlib import Path
        for i in range(1, passes + 1):
            print(f"  weight pass {i}: W_in {w:.1f} lb", flush=True)
            try:
                wcdeck.write_deck(wcdeck.WC_DECK, Path(LOGS) / "deck_arcA_constructed",
                                  mission.MTOW_LB, w, oas=oas)
                w_new = wcdeck.run_wingcalc(Path(LOGS) / "deck_arcA_constructed",
                                            Path(LOGS) / "wc_arcA_constructed")
            except Exception as exc:
                sizing_error = f"{type(exc).__name__}: {exc}"
                print(f"  !!! sizing FAILED: {sizing_error}", flush=True)
                break
            hist.append({"pass": i, "w_in_lb": w, "w_wing_lb": w_new,
                         "residual_lb": w_new - w})
            print(f"  >>> p{i}: {w:.1f} -> {w_new:.1f} lb ({w_new - w:+.1f})", flush=True)
            if abs(w_new - w) < A.W_TOL_LB:
                break
            w += 0.5 * (w_new - w)
        if hist:
            w_wing = hist[-1]["w_wing_lb"]
            batt = mission.battery_lb(w_wing)
            rng = mission.electric_range_nmi(w_wing, best["drag_N"])
            print(f"\n  W_wing {w_wing:.1f} lb, battery {batt:.1f} lb, "
                  f"range {rng:.1f} nmi", flush=True)

    out = os.path.join(LOGS, f"arc_a_constructed_{a.profile}_{a.airfoil}.json")
    ser = {k: (v.tolist() if hasattr(v, "tolist") else v)
           for k, v in best.items() if not k.startswith("_")}
    ser.update({"arc": "A", "profile": a.profile, "airfoil": a.airfoil,
                "constructed": True, "straight_aft_spar": True,
                "section_blend": {"inboard": n_in, "outboard": n_out,
                                  "f_start": f0, "f_end": f1},
                "feasible": bool(not viol and dep_ok),
                # the schema arc_optimal_toc writes, so downstream needs no special case
                "depth_delivered_in": best["depth_in"],
                "chord_at_aileron_in": float(best["station_chord_in"][3]),
                "toc_delivered_ail": best["toc_ail"],
                "toc_root": float(best["toc_full"][0]),
                "toc_tip": float(best["toc_full"][-1]),
                "root_toc_req": cp_toc[0], "ratio_req": cp_toc[1],
                "depth_req_in": A.DEPTH_REQ,
                "w_wing_lb": w_wing, "batt_lb": batt, "R_nmi": rng,
                "weight_history": hist, "sizing_error": sizing_error,
                "converged": bool(hist) and abs(hist[-1]["residual_lb"]) < A.W_TOL_LB,
                "success": True})
    ser.pop("mesh", None)          # large, and downstream replays the design instead
    json.dump(ser, open(out, "w"), indent=2)
    print(f"  wrote {out}")
