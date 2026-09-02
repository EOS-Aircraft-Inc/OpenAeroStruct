"""ARC A, THIRD CONSTRUCTION: the aft spar straight AND at a constant chord fraction.

THIS IS NOT A NEW ARCHITECTURE. The arc letters name CONSTRAINT CLASSES -- Arc A is
"constant chord over region A" (export_dat.ARCH_NOTE) -- and this design is in Arc
A's class. It holds region A's chord more firmly than arc_a_construct.py does: the
`preserved` rule holds that chord by construction, where `root_le_fixed` only left
it alone because p was pinned at its baseline for that purpose. So Arc A now has
three design points, told apart by HOW they were produced, which is the same way
arc_a_constructed_* is already told apart from arc_optimal_toc_A_*:

    arc_optimal_toc_A_*    drag-optimized, kinking aft spar
    arc_a_constructed_*    constructed, rear(y) = p + K/c(y), K = 6.03 in
    arc_a_constfrac_*      constructed, rear(y) = p,          K = 0     <- this one

WHAT CHANGES. Both constructions hold the aft spar at constant x. The offset one
holds it a fixed 6.03 in aft of the straight construction line, and a fixed
distance is a rising chord fraction on a falling chord -- 0.7500c at the root to
0.8040c at the winglet junction. This one puts the aft spar ON the construction
line, so the fraction is constant. It is the construction Plan L was lofted
around, at 0.60065c.

WHY THE OFFSET CONSTRUCTION CANNOT DO THIS. It uses `root_le_fixed`, which holds
the product p * c invariant. The box is then (p - 0.12) * c = p*c - 0.12*c, and
with p*c frozen at 71.49 in it cannot exceed 71.49 * (1 - 0.12/p) <= 62.91 in at
any p -- below the 65 in the inboard nacelle needs. The 6.03 in offset is how that
construction buys the missing width, and the rising fraction is what it costs. No sweep
angle changes this: sweep is a derived output of the parameterization
(README: tan(sweep_LE,B) = p * c_A * (1 - taper_B) / span_B), and the binding
station y = 100 in sits INSIDE region A, where the chord is constant and the
leading edge is flat.

THE RULE THAT DOES. `preserved` (exponent 0) holds region A's chord and lets p
select which chord fraction is the straight line -- wing7_front_spar.py:24 says so,
and wing 7 already uses it to pin the FRONT spar. Region A keeps the as-built
104.79 in at the nacelle stations instead of shrinking to 95.13 in, and the width
requirement is then met by the fraction alone.

WHAT IT COSTS: THE x ANCHOR. A pinned p under `preserved` re-anchors the planform
in x, by up to 59 in aft if the baseline's own x_spar is reused
(wing7_front_spar.py:42). VLM drag does not change with that translation. The
landing gear and the centre of gravity stations do, and this study does not carry
them. That is the open item on this construction, and it is not an aerodynamic one.

THE SOLVE IS DECOUPLED, AND THAT IS A PROPERTY OF THE RULE. Under `preserved` the
chord does not track p, so the chord at every constraint station inboard of
region A's end is independent of BOTH unknowns. Region A ends at 361.70 in and the
three width stations are at 100, 176 and 356 in, so:

    p        is set by WIDTH alone, and it is the SMALLEST fraction that clears
             every station -- a larger one only moves the spar aft for nothing.
    taper_B  is then set by the 7 in DEPTH floor alone, root to aileron.

Neither feeds back into the other. Arc A's solve could not be split this way,
because `root_le_fixed` makes the chord a function of p.

p is solved by Newton on the MODEL's reported width rather than taken from the
closed form p = 0.12 + max(req/c). The derivative is exact -- the box grows by c
inches for each unit of p, and c does not move with p here -- so it converges in
one step, onto what the model measures rather than onto a reconstruction of it.
Measured, the closed form agrees to 0.0016 in at the binding station, so this is
robustness, not the correction of a known error.

Reports every constraint with its margin, and drag against the other two Arc A
design points, so the price of holding the fraction constant is visible.
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

from studies.vsp_planform import config                              # noqa: E402
from studies.vsp_planform.degen_csv import read_degen_csv, lifting_surfaces  # noqa: E402
from studies.vsp_planform.coupling import deck as wcdeck             # noqa: E402
from studies.vsp_planform.coupling import mission                    # noqa: E402
import wing2_oas as w2                                              # noqa: E402
import arc_optimal_toc as A                                         # noqa: E402
from arc_a_construct import evaluate, BASE_STATIONS, LOGS, M2_FT2    # noqa: E402

# A construction run is long -- model passes then the WingCalc loop -- and it is
# usually watched from a file rather than a terminal. Line buffering makes the
# progress visible as it happens instead of at exit.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:                       # pragma: no cover -- Python < 3.7
    pass

RULE = "preserved"
# The width stations Arc A carries, plus the junction. This construction adds no
# requirement of its own -- it answers the same ones with a different spar rule.
WIDTH_STATIONS = BASE_STATIONS + ((674.9, w2.JUNCTION_BOX_IN),)
N_CON = len(WIDTH_STATIONS)
# The depth requirement holds from ROOT TO AILERON, so it is sampled along the
# span. These are pure REPORTING stations: zero required width, nothing constrained.
REPORT_Y = tuple(float(v) for v in np.linspace(40.0, A.Y_AIL, 40))
TAPER_BOUNDS = (0.15, 1.0)          # README: taper_B design-variable bounds


def pass_at(taper_B, p, case, ret_at, cmt_at):
    """Evaluate at this fraction and this taper, with the flat schedule."""
    sched = ((100.0, p), (674.9, p))          # K = 0: the schedule IS the fraction
    st = WIDTH_STATIONS + tuple((y, 0.0) for y in REPORT_Y)
    r = evaluate(taper_B, case, cmt_at, p, sched, st, rule=RULE)

    ch = np.asarray(r["station_chord_in"], dtype=float)
    toc = np.asarray(r["toc_full"], dtype=float)
    m = np.asarray(r["mesh"]) / config.SCALE
    ym = np.abs(m[0, :, 1]); yp = 0.5 * (ym[:-1] + ym[1:])

    dep = np.array([ret_at(y, p) * float(np.interp(y, yp, toc)) * c
                    for y, c in zip(REPORT_Y, ch[N_CON:])])
    r["depth_span_in"], r["depth_span_y"] = dep, list(REPORT_Y)
    r["depth_min_in"] = float(dep.min())
    r["depth_min_y_in"] = float(REPORT_Y[int(dep.argmin())])
    r["depth_in"] = float(dep[-1])
    r["c_ail_in"] = float(ch[N_CON:][-1])     # the LAST reporting station IS the aileron
    r["stations"] = [y for y, _ in WIDTH_STATIONS]
    r["width_req_in"] = [v for _, v in WIDTH_STATIONS]
    r["box_width_in"] = np.asarray(r["box_width_in"])[:N_CON]
    r["station_chord_in"] = ch[:N_CON]
    r["rear_schedule"] = [[float(y), float(v)] for y, v in sched]
    r["spar_at_aileron"] = p
    r["retention_at_spar"] = ret_at(A.Y_AIL, p)
    r["toc_ail"] = float(np.interp(A.Y_AIL, yp, toc))
    r["K_in"] = 0.0
    # Straightness, measured on the MESH rather than on the rule. The rule is
    # straight by identity here -- rear == p -- so the only number worth printing
    # is the departure the baseline loft itself carries through.
    keep = ym <= 674.95 + 1e-6
    x_aft = m[0, keep, 0] + p * (m[-1, keep, 0] - m[0, keep, 0])
    r["x_aft_spread_in"] = float(np.ptp(x_aft))
    return r


def solve_p(case, ret_at, cmt_at, taper_B, p0, tol_w, bounds):
    """The smallest constant fraction that clears every width station.

    Newton on the worst margin. The derivative is exact to the cosine correction:
    the box grows by ``c`` inches for each unit of ``p``, and ``c`` at these
    stations does not move with ``p`` under `preserved`.
    """
    p, hist = float(p0), []
    for _ in range(6):
        r = pass_at(taper_B, p, case, ret_at, cmt_at)
        box = np.asarray(r["box_width_in"], dtype=float)
        req = np.asarray(r["width_req_in"], dtype=float)
        ch = np.asarray(r["station_chord_in"], dtype=float)
        i = int(np.argmin(box - req))
        marg = float(box[i] - req[i])
        hist.append({"p": p, "worst_y_in": r["stations"][i], "margin_in": marg})
        print(f"  p {p:.6f}c -> worst width margin {marg:+.3f} in at y "
              f"{r['stations'][i]:.1f} (box {box[i]:.2f}, need {req[i]:.2f})", flush=True)
        if 0.0 <= marg <= tol_w:
            return p, r, hist
        p = float(np.clip(p - marg / ch[i], bounds[0], bounds[1]))
        if abs(p - hist[-1]["p"]) < 1e-9:
            break
    return p, pass_at(taper_B, p, case, ret_at, cmt_at), hist


def solve_taper(case, ret_at, cmt_at, p, t0, tol_d):
    """Drive the MINIMUM depth root-to-aileron onto the floor, from either side.

    Two-sided on purpose. The requirement is a floor, so a design that clears it
    with slack is carrying chord it does not need, and chord is area and drag.
    """
    r0 = pass_at(t0, p, case, ret_at, cmt_at)
    show(t0, r0)
    if A.DEPTH_REQ <= r0["depth_min_in"] <= A.DEPTH_REQ + tol_d:
        return t0, r0
    d0 = r0["depth_min_in"]
    t1 = float(np.clip(t0 * A.DEPTH_REQ / max(d0, 1e-6), *TAPER_BOUNDS))
    best = r0
    for _ in range(10):
        r1 = pass_at(t1, p, case, ret_at, cmt_at)
        show(t1, r1)
        d1 = r1["depth_min_in"]
        if d1 >= A.DEPTH_REQ:
            best = r1                       # only a feasible pass may be kept
        if A.DEPTH_REQ <= d1 <= A.DEPTH_REQ + tol_d:
            return t1, r1
        if abs(d1 - d0) < 1e-9:
            break
        t_new = t1 + (A.DEPTH_REQ - d1) * (t1 - t0) / (d1 - d0)
        t0, d0 = t1, d1
        t1 = float(np.clip(t_new, *TAPER_BOUNDS))
    return float(best["taper_B"]), best


def show(t, res):
    print(f"  taper_B {t:.4f} -> MIN depth {res['depth_min_in']:5.2f} in at y "
          f"{res['depth_min_y_in']:5.0f} ({res['depth_min_in']-A.DEPTH_REQ:+.3f}), "
          f"aileron {res['depth_in']:5.2f}, c_ail "
          f"{res['c_ail_in']:5.1f} in, drag {res['drag_N']:9.1f} N, "
          f"S_ref {res['S_ref']*M2_FT2:7.1f} ft2", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="optimal", choices=sorted(A.PROFILES))
    ap.add_argument("--airfoil", default="e694")
    ap.add_argument("--tol-depth", type=float, default=0.02, help="depth tolerance, in")
    ap.add_argument("--tol-width", type=float, default=0.03, help="width tolerance, in")
    a = ap.parse_args()

    cp_toc = A.PROFILES[a.profile]
    A.RET, A.C_MAX_T = A.section(a.airfoil)
    n_in, n_out, f0, f1 = A.SECTION_BLEND["A"]
    ret_at, cmt_at, _ = A.blended_section(n_in, n_out, f0, f1)

    src = os.path.join(LOGS, f"arc_optimal_toc_A_{a.profile}_{a.airfoil}.json")
    case = json.load(open(src))
    case["_cp0"], case["_cpr"] = cp_toc
    p_bounds = tuple(float(v) for v in config.WINGBOX_CHORD_PCT_BOUNDS)

    print("ARC A, constant-fraction construction -- the aft spar straight AND at "
          "a constant chord fraction")
    print(f"seeded from {os.path.basename(src)}")
    print(f"  twist_cp {[round(v, 2) for v in case['twist_cp']]}, "
          f"alpha {case['alpha']:.3f} deg, taper_B {case['taper_B']:.4f}")
    print(f"  region A rule '{RULE}', so p selects the straight line and region A's "
          f"chord is held")
    print(f"  section {n_in} -> {n_out} over {f0*A.SEMI_IN:.0f}-{f1*A.SEMI_IN:.0f} in")
    print(f"  depth floor {A.DEPTH_REQ:.2f} in, root to y = {A.Y_AIL:.1f} in")
    print(f"  wingbox_pct bounds {p_bounds[0]:.4f} - {p_bounds[1]:.4f}\n")

    t_seed = float(case["taper_B"])
    print("STEP 1  the fraction, from WIDTH alone")
    p, r_p, p_hist = solve_p(case, ret_at, cmt_at, t_seed,
                             float(case["wingbox_pct"]), a.tol_width, p_bounds)
    at_bound = abs(p - p_bounds[1]) < 1e-9
    print(f"  p = {p:.5f}c{'   AT THE UPPER BOUND' if at_bound else ''}\n")

    print("STEP 2  the taper, from the DEPTH floor alone")
    t_best, best = solve_taper(case, ret_at, cmt_at, p, t_seed, a.tol_depth)

    # The decoupling is a claim about the model, so it is checked rather than
    # asserted: the width stations must read the same at the solved taper.
    w0 = np.asarray(r_p["box_width_in"], dtype=float)
    w1 = np.asarray(best["box_width_in"], dtype=float)
    drift = float(np.max(np.abs(w1[:3] - w0[:3])))
    print(f"\n  decoupling check: the three inboard widths moved {drift:.4f} in "
          f"when taper_B changed {t_seed:.4f} -> {t_best:.4f}")

    print("\nCONSTRAINTS")
    viol = []
    for y, req, got in zip(best["stations"], best["width_req_in"], best["box_width_in"]):
        ok = got >= req - 1e-6
        if not ok:
            viol.append(round(y, 1))
        print(f"  y {y:7.1f}  box {got:6.2f} in  need {req:6.2f}  "
              f"{'ok      ' if ok else 'VIOLATED'}")
    dep_ok = best["depth_min_in"] >= A.DEPTH_REQ - a.tol_depth
    print(f"  depth, MINIMUM root->aileron {best['depth_min_in']:5.2f} in at y "
          f"{best['depth_min_y_in']:.0f}   need {A.DEPTH_REQ:.2f}  "
          f"{'ok' if dep_ok else 'VIOLATED'}")
    print(f"  depth at the aileron itself  {best['depth_in']:5.2f} in")
    print(f"  aft spar chord fraction: {p:.5f}c at EVERY station, constant by "
          f"construction (K = 0)")
    print(f"  aft spar straightness on the mesh: x varies by "
          f"{best['x_aft_spread_in']:.3f} in over the wingbox")

    print("\nRESULT")
    print(f"  wingbox_pct     {p:.5f}   = the aft spar fraction, not a line inside the box")
    print(f"  taper_B         {best['taper_B']:.4f}   (seed {t_seed:.4f})")
    print(f"  retention at the spar {best['retention_at_spar']:.4f}  "
          f"(Arc A at 0.7930c: {ret_at(A.Y_AIL, 0.79296):.4f})")
    print(f"  S_ref          {best['S_ref']*M2_FT2:8.1f} ft2")
    print(f"  drag           {best['drag_N']:8.1f} N")
    aa = os.path.join(LOGS, f"arc_a_constructed_{a.profile}_{a.airfoil}.json")
    if os.path.exists(aa):
        c_aa = json.load(open(aa))
        print(f"    vs offset-spar Arc A {c_aa['drag_N']:.1f} N "
              f"({100*(best['drag_N']/c_aa['drag_N']-1):+.2f}%), "
              f"S_ref {c_aa['S_ref']*M2_FT2:.1f} ft2")
    print(f"    vs Arc A optimized   {case['drag_N']:.1f} N "
          f"({100*(best['drag_N']/case['drag_N']-1):+.2f}%), "
          f"S_ref {case['S_ref']*M2_FT2:.1f} ft2")
    if viol or not dep_ok:
        print(f"  NOT FEASIBLE -- width violated at y = {viol}" if viol
              else "  NOT FEASIBLE -- depth short")

    # ---- weight, so this design can be compared with the other two on range ----
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
                wcdeck.write_deck(wcdeck.WC_DECK, Path(LOGS) / "deck_arcA_constfrac",
                                  mission.MTOW_LB, w, oas=oas)
                w_new = wcdeck.run_wingcalc(Path(LOGS) / "deck_arcA_constfrac",
                                            Path(LOGS) / "wc_arcA_constfrac")
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

    out = os.path.join(LOGS, f"arc_a_constfrac_{a.profile}_{a.airfoil}.json")
    ser = {k: (v.tolist() if hasattr(v, "tolist") else v)
           for k, v in best.items() if not k.startswith("_")}
    ser.update({"arc": "A", "profile": a.profile, "airfoil": a.airfoil,
                "constructed": True, "straight_aft_spar": True,
                "constant_aft_fraction": True,
                "region_a_rule": RULE,
                "p_solve_history": p_hist,
                "width_drift_in": drift,
                "section_blend": {"inboard": n_in, "outboard": n_out,
                                  "f_start": f0, "f_end": f1},
                "feasible": bool(not viol and dep_ok),
                # the schema arc_optimal_toc writes, so downstream needs no special case
                "depth_delivered_in": best["depth_in"],
                "chord_at_aileron_in": best["c_ail_in"],
                "toc_delivered_ail": best["toc_ail"],
                "toc_root": float(best["toc_full"][0]),
                "toc_tip": float(best["toc_full"][-1]),
                "root_toc_req": cp_toc[0], "ratio_req": cp_toc[1],
                "depth_req_in": A.DEPTH_REQ,
                "w_wing_lb": w_wing, "batt_lb": batt, "R_nmi": rng,
                "weight_history": hist, "sizing_error": sizing_error,
                "converged": bool(hist) and abs(hist[-1]["residual_lb"]) < A.W_TOL_LB,
                "success": True})
    for k in ("mesh", "depth_span_in", "depth_span_y"):
        ser.pop(k, None)          # large, and downstream replays the design instead
    json.dump(ser, open(out, "w"), indent=2)
    print(f"  wrote {out}")
