"""Rear-spar kink sweep in full OAS (queued item 5 in HANDOFF.md).

The v1 spar sweep ranked candidates on thickness at 0.75c and is marked KNOWN
BAD; it also could not run a real aero case, because `param.py` had a single
scalar spar fraction and a kinking spar simply could not be expressed. Both
blockers are gone: the spar is a schedule and region B can be re-lofted.

This sweeps the OUTBOARD end of the kink -- the spar fraction at the winglet
junction -- holding 0.750c inboard, and re-optimizes the planform in full OAS at
each step. The trade it exposes:

  * spar aft  -> box is a bigger fraction of chord -> junction chord can shrink
                 -> less wetted area
  * spar aft  -> lands on thinner section          -> spar depth falls

Drag comes from OAS. Depth is checked afterwards against the as-built section's
own thickness distribution, so a row can be aerodynamically better and still be
structurally inadmissible -- which is the whole point of running both.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))
sys.path.insert(0, os.path.dirname(_HERE))

from studies.vsp_planform import config  # noqa: E402
import studies.vsp_planform.run_opt as ro  # noqa: E402
from studies.vsp_planform.run_opt import POINT, trim_alpha  # noqa: E402

import wing2_oas as w2  # noqa: E402
from doe_v3 import asbuilt  # noqa: E402

JUNCTION_SPAR = [0.35, 0.40, 0.45, 0.499, 0.55, 0.60, 0.65, 0.70]
DEPTH_REQ_IN = 7.0

OUT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs", "spar_sweep_oas.json")

q = 0.5 * config.RHO * config.V_MS**2


def run_at(p_junction, af):
    schedule = ((356.0, 0.750), (674.9, p_junction))
    stations = ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0), (674.9, w2.JUNCTION_BOX_IN))

    w2.REAR_SCHEDULE = schedule
    w2.WIDTH_STATIONS = stations
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = stations

    mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    y_c_in = regions.y_c_start

    prob, _ = ro.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
    prob.run_model()
    s0 = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    alpha0 = trim_alpha(prob, w2.W / (q * s0))
    ro.add_optimization(prob, "plan_l", mesh, planform0, s0, mode="fixed_lift", weight=w2.W)
    prob.set_val("alpha", alpha0, units="deg")
    prob.run_model()
    prob.run_driver()

    r = w2.evaluate(prob, y_c_in)
    r["junction_spar_xc"] = p_junction
    r["success"] = bool(prob.driver.result.success)
    r["exit_status"] = str(prob.driver.result.exit_status)

    # Structural check, after the fact and independent of the aero.
    t_frac = float(af.local_thickness(x_over_c=p_junction))
    r["t_frac_at_spar"] = t_frac
    r["depth_in"] = t_frac * r["junction_chord_in"]
    r["depth_ok"] = bool(r["depth_in"] >= DEPTH_REQ_IN)
    return r


if __name__ == "__main__":
    af = asbuilt()
    print("=" * 96)
    print(f"Rear-spar kink sweep in full OAS -- junction spar fraction, {w2.JUNCTION_BOX_IN:.0f} in box, as-built section")
    print("=" * 96)

    results = []
    for p in JUNCTION_SPAR:
        r = run_at(p, af)
        results.append(r)
        flag = "" if r["depth_ok"] else "   <-- DEPTH SHORT"
        print(
            f"\n  spar {p:.3f}c -> junction chord {r['junction_chord_in']:6.2f} in"
            f"   S_ref {r['S_ref']:7.3f}   drag {r['drag_N']:9.1f} N   [{r['exit_status']}]"
        )
        print(f"    t/c at spar {r['t_frac_at_spar']:.4f} -> depth {r['depth_in']:5.2f} in (need {DEPTH_REQ_IN}){flag}")

    ref = next(r for r in results if abs(r["junction_spar_xc"] - 0.499) < 1e-9)
    print("\n" + "=" * 96)
    print(f"  {'spar x/c':>9} {'junc chord':>11} {'S_ref m2':>9} {'drag N':>10} {'vs 0.499':>9} {'depth in':>9} {'ok':>4}")
    for r in results:
        print(
            f"  {r['junction_spar_xc']:9.3f} {r['junction_chord_in']:11.2f} {r['S_ref']:9.3f} {r['drag_N']:10.1f} "
            f"{r['drag_N'] / ref['drag_N'] - 1:+8.2%} {r['depth_in']:9.2f} {'yes' if r['depth_ok'] else 'NO':>4}"
        )
    print("=" * 96)

    ok = [r for r in results if r["depth_ok"]]
    if ok:
        best = min(ok, key=lambda r: r["drag_N"])
        print(f"  best admissible: spar {best['junction_spar_xc']:.3f}c, drag {best['drag_N']:.1f} N, "
              f"junction chord {best['junction_chord_in']:.2f} in, depth {best['depth_in']:.2f} in")
    else:
        print("  NO row meets the 7 in depth on the as-built section.")

    with open(OUT, "w") as fh:
        json.dump({"cases": results, "meta": {"depth_req_in": DEPTH_REQ_IN, "box_in": w2.JUNCTION_BOX_IN}}, fh, indent=2)
    print(f"\n  wrote {OUT}")
