"""What inboard box margin actually costs (queued item 2, corrected).

`root_chord_sweep.py` asked this by widening the baseline root chord and found
the question was degenerate: under the "root_le_fixed" rule `wingbox_pct` scales
every chord together, so the optimizer undid the widening exactly (chords
identical to 3 decimals from 105 to 114 in, `wingbox_pct` tracking 1/k) and
handed back the same wing. The optimizer will never *buy* margin -- it drives the
binding stations to zero slack by construction.

So price margin by requiring it. The inboard requirement is raised together at
y = 100 and y = 176 in, everything else held at the wing 2 design point, and the
planform is re-optimized in full OAS at each step. That makes the reported number
the true cost of the requirement rather than the cost of a design-variable bound.
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

INBOARD_REQ_IN = [65.0, 67.0, 69.0, 71.0, 73.0, 75.0]

OUT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs", "margin_sweep.json")

q = 0.5 * config.RHO * config.V_MS**2


def run_at(req_in):
    """Re-optimize with the two inboard stations requiring ``req_in`` of box."""
    stations = ((100.0, req_in), (176.0, req_in), (356.0, 55.0), (674.9, w2.JUNCTION_BOX_IN))
    w2.WIDTH_STATIONS = stations
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = w2.REAR_SCHEDULE
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
    r["inboard_req_in"] = req_in
    r["success"] = bool(prob.driver.result.success)
    r["exit_status"] = str(prob.driver.result.exit_status)
    return r


if __name__ == "__main__":
    print("=" * 88)
    print("Cost of inboard box margin (y = 100 and y = 176 raised together), full OAS")
    print("=" * 88)

    results = []
    for req in INBOARD_REQ_IN:
        r = run_at(req)
        results.append(r)
        print(
            f"\n  req {req:5.1f} in -> S_ref {r['S_ref']:7.3f} m^2  CL {r['CL']:.4f}"
            f"  drag {r['drag_N']:9.1f} N  [{r['exit_status']}]"
        )
        print(f"    chords " + "  ".join(f"{v:7.2f}" for v in r["station_chord_in"]))
        print(f"    margin " + "  ".join(f"{v:+7.2f}" for v in r["box_margin_in"]))

    base = results[0]["drag_N"]
    print("\n" + "=" * 88)
    print(f"  {'req in':>7} {'margin vs as-built':>19} {'S_ref m2':>9} {'drag N':>10} {'vs 65 in':>9} {'binding':>28}")
    for r in results:
        names = ["y=100", "y=176", "y=356", "junction"]
        binding = ",".join(n for n, m in zip(names, r["box_margin_in"]) if m < 0.02)
        print(
            f"  {r['inboard_req_in']:7.1f} {r['inboard_req_in'] / 65.0 - 1:+18.1%} {r['S_ref']:9.3f} "
            f"{r['drag_N']:10.1f} {r['drag_N'] / base - 1:+8.2%} {binding:>28}"
        )
    print("=" * 88)

    with open(OUT, "w") as fh:
        json.dump({"cases": results, "meta": {"weight_N": w2.W, "junction_box_in": w2.JUNCTION_BOX_IN}}, fh, indent=2)
    print(f"\n  wrote {OUT}")
