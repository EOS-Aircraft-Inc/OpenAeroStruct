"""Ailerons moved inboard to ~90% semi-span (user, 2026-08-15).

The aft-spar depth requirement houses the aileron actuator. At the winglet
junction (y = 674.9 in, 95.3% semi-span) it was crippling: the junction chord is
set by the 20 in box requirement, and at that chord the as-built section's
maximum thickness is only 6.25 in, so 7 in of depth was unreachable at ANY spar
station. Forcing it cost ~1.7% of cruise drag.

The requirement does not go away -- it moves. Ailerons at 90% semi-span put it at
y = 637.2 in, where the chord is larger and the section is thicker in absolute
terms.

HOW THE CONSTRAINT IS APPLIED
-----------------------------
The model has no depth constraint, only box width. But at a FIXED spar fraction
depth is strictly proportional to chord:

    depth(y) = retention(spar(y)) * toc(y) * chord(y)

and `toc` is not a design variable in this study, so `retention * toc` is a
constant at a given station. A depth requirement is therefore exactly a minimum
chord requirement, which is exactly a minimum box width requirement:

    width_equivalent = (spar(y) - front) * depth_req / (retention * toc)

So the depth constraint is imposed by adding one more entry to
`WINGBOX_WIDTH_STATIONS`. This is an exact re-encoding, not an approximation --
but it does mean the reported "box width" at the aileron station is a proxy for
depth and should not be read as a real width requirement.
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
from studies.vsp_planform.param import rear_spar_fraction  # noqa: E402

import wing2_oas as w2  # noqa: E402
from doe_v3 import asbuilt  # noqa: E402

SEMI_IN = 118.0 * 12.0 / 2.0  # 708 in
AILERON_FRAC = 0.90
Y_AIL = AILERON_FRAC * SEMI_IN  # 637.2 in

JUNCTION_SPAR = [0.499, 0.550, 0.600, 0.650, 0.700, 0.750]
DEPTH_REQS = [7.0, 6.0]

OUT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs", "aileron_90.json")

q = 0.5 * config.RHO * config.V_MS**2


def retention_fn(af):
    xs = np.linspace(0.05, 0.95, 300)
    t = np.array([float(af.local_thickness(x_over_c=x)) for x in xs])
    return lambda x: float(np.interp(x, xs, t / t.max()))


def run(p_junction, depth_req, ret_of, toc_at):
    schedule = ((356.0, 0.750), (674.9, p_junction))
    spar_ail = float(rear_spar_fraction(Y_AIL, schedule))
    ret = ret_of(spar_ail)
    toc = toc_at(Y_AIL)

    # Depth -> minimum chord -> equivalent minimum box width at the aileron.
    c_req = depth_req / (ret * toc)
    w_equiv = (spar_ail - w2.FRONT_PCT) * c_req

    stations = (
        (100.0, 65.0),
        (176.0, 65.0),
        (356.0, 55.0),
        (Y_AIL, w_equiv),
        (674.9, w2.JUNCTION_BOX_IN),
    )
    w2.REAR_SCHEDULE = schedule
    w2.WIDTH_STATIONS = stations
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = stations

    mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    prob, _ = ro.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
    prob.run_model()
    s0 = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    alpha0 = trim_alpha(prob, w2.W / (q * s0))
    ro.add_optimization(prob, "plan_l", mesh, planform0, s0, mode="fixed_lift", weight=w2.W)
    prob.set_val("alpha", alpha0, units="deg")
    prob.run_model()
    prob.run_driver()

    r = w2.evaluate(prob, regions.y_c_start)
    r["junction_spar_xc"] = p_junction
    r["depth_req_in"] = depth_req
    r["spar_at_aileron"] = spar_ail
    r["chord_req_at_aileron_in"] = c_req
    r["chord_at_aileron_in"] = r["station_chord_in"][3]
    r["depth_at_aileron_in"] = ret * toc * r["station_chord_in"][3]
    r["success"] = bool(prob.driver.result.success)
    r["exit_status"] = str(prob.driver.result.exit_status)
    return r


if __name__ == "__main__":
    af = asbuilt()
    ret_of = retention_fn(af)

    # Baseline t/c distribution, from the re-lofted loft (t_over_c is not a DV).
    w2.apply_wing2_box()
    _, stick0, _, _ = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    y_s = np.abs(np.asarray(stick0.le[:, 1], dtype=float))
    toc_at = lambda y: float(np.interp(y, y_s, stick0.toc))

    print("=" * 100)
    print(f"Ailerons at {AILERON_FRAC:.0%} semi-span -> depth required at y = {Y_AIL:.1f} in")
    print(f"  (was y = 674.9 in, {674.9 / SEMI_IN:.1%} semi-span)")
    print(f"  local t/c there = {toc_at(Y_AIL):.4f}")
    print("=" * 100)

    base = 10736.1  # as-built, full OAS
    res = []
    for depth_req in DEPTH_REQS:
        print(f"\n--- {depth_req:.0f} in depth at y = {Y_AIL:.1f} in " + "-" * 55)
        print(f"  {'junc spar':>10} {'spar@ail':>9} {'c req':>8} {'c got':>8} {'depth':>7} "
              f"{'junc ch':>8} {'S_ref':>8} {'drag N':>9} {'vs built':>9}")
        for p in JUNCTION_SPAR:
            r = run(p, depth_req, ret_of, toc_at)
            res.append(r)
            print(
                f"  {p:10.3f} {r['spar_at_aileron']:9.3f} {r['chord_req_at_aileron_in']:8.2f} "
                f"{r['chord_at_aileron_in']:8.2f} {r['depth_at_aileron_in']:7.2f} "
                f"{r['junction_chord_in']:8.2f} {r['S_ref']:8.3f} {r['drag_N']:9.1f} "
                f"{r['drag_N'] / base - 1:+8.2%}"
            )

    print("\n" + "=" * 100)
    for depth_req in DEPTH_REQS:
        rows = [r for r in res if r["depth_req_in"] == depth_req]
        best = min(rows, key=lambda r: r["drag_N"])
        print(
            f"  BEST at {depth_req:.0f} in: junction spar {best['junction_spar_xc']:.3f}c, "
            f"drag {best['drag_N']:.1f} N ({best['drag_N'] / base - 1:+.2%} vs as-built), "
            f"aileron chord {best['chord_at_aileron_in']:.2f} in, junction chord {best['junction_chord_in']:.2f} in"
        )
    print("=" * 100)

    with open(OUT, "w") as fh:
        json.dump({"cases": res, "meta": {"y_aileron_in": Y_AIL, "semi_span_in": SEMI_IN,
                                          "junction_box_in": w2.JUNCTION_BOX_IN}}, fh, indent=2)
    print(f"\n  wrote {OUT}")
