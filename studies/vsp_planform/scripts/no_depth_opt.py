"""Wing 2 with the aft-spar DEPTH requirement removed (user, 2026-08-15).

The 7 in depth at the winglet junction existed to house an aileron actuator. The
ailerons can move inboard, so the requirement goes away at that station -- and it
was the single most expensive constraint in the study, worth ~1.3-1.7% of cruise
drag (see spar_sweep_oas.py).

Two things change once it is gone:

1. The kink loses its purpose. The spar was swept FORWARD outboard to find
   thickness; with no depth to find, it wants to go AFT, because a further-aft
   spar makes the box a bigger fraction of the chord and so needs less chord. A
   straight 0.750c spar needs only 20/(0.75-0.12) = 31.7 in of junction chord
   against 40 in as-built, so the junction box may stop binding entirely.

2. The requirement does not vanish, it MOVES. Wherever the ailerons end up still
   needs actuator depth. So this also maps depth against span and reports how far
   outboard the ailerons can sit for 7 in and for 6 in.

Depth is computed as retention(x) * toc(y) * chord(y): the as-built section's own
thickness distribution, normalized to its peak, scaled by the local t/c and the
local chord. That is the same convention the earlier depth notes used, extended
from one station to the whole span.
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

OUT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs", "no_depth_opt.json")

q = 0.5 * config.RHO * config.V_MS**2

# Junction box width stays at the user's 20 in. Only DEPTH was withdrawn; if the
# actuator also drove the box WIDTH there, this should be revisited too.
CASES = {
    "straight_0750": ((356.0, 0.750), (674.9, 0.750)),
    "kink_0600": ((356.0, 0.750), (674.9, 0.600)),
    "kink_0499_old_design_point": ((356.0, 0.750), (674.9, 0.499)),
}

DEPTH_REQS = (7.0, 6.0)


def run(schedule):
    stations = ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0), (674.9, w2.JUNCTION_BOX_IN))
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
    r["success"] = bool(prob.driver.result.success)
    r["exit_status"] = str(prob.driver.result.exit_status)

    # Spanwise depth map on the converged geometry.
    m = prob.get_val("wing.mesh", units="m")
    y_in = np.abs(m[0, :, 1]) / config.SCALE
    chord_in = (m[-1, :, 0] - m[0, :, 0]) / config.SCALE
    # t/c from the baseline loft rather than the model: `t_over_c` is stored per
    # panel with a chordwise axis, and t/c is not a design variable here, so the
    # loft's own distribution is both simpler and exactly what the model carries.
    y_s = np.abs(np.asarray(stick.le[:, 1], dtype=float))
    toc = np.interp(y_in, y_s, np.asarray(stick.toc, dtype=float))

    r["y_in"] = y_in.tolist()
    r["chord_in"] = chord_in.tolist()
    r["depth_in"] = None  # filled by caller, needs the section
    r["toc"] = toc.tolist()
    r["schedule"] = schedule
    return r


def depth_map(r, af):
    """Depth available at the spar, station by station."""
    xs = np.linspace(0.05, 0.95, 200)
    t = np.array([float(af.local_thickness(x_over_c=x)) for x in xs])
    retention = t / t.max()

    y = np.array(r["y_in"])
    chord = np.array(r["chord_in"])
    toc = np.array(r["toc"])
    spar = np.array([float(rear_spar_fraction(v, r["schedule"])) for v in y])
    ret = np.interp(spar, xs, retention)
    return ret * toc * chord, spar


def outboard_limit(y, depth, req):
    """Outboard-most station with at least ``req`` of depth."""
    ok = depth >= req
    if not ok.any():
        return None
    # Walk in from the tip to the last contiguous run that satisfies it.
    idx = np.flatnonzero(ok)
    return float(y[idx[-1]]), float(depth[idx[-1]])


if __name__ == "__main__":
    af = asbuilt()
    print("=" * 92)
    print(f"Wing 2 with the junction DEPTH requirement removed (box width still {w2.JUNCTION_BOX_IN:.0f} in)")
    print("=" * 92)

    res = {}
    for name, sched in CASES.items():
        r = run(sched)
        d, spar = depth_map(r, af)
        r["depth_in"] = d.tolist()
        r["spar_xc"] = spar.tolist()
        res[name] = r

        print(f"\n  {name}: spar {sched[0][1]:.3f}c inboard -> {sched[-1][1]:.3f}c at the junction")
        print(f"    S_ref {r['S_ref']:7.3f} m^2   CL {r['CL']:.4f}   drag {r['drag_N']:9.1f} N   [{r['exit_status']}]")
        print(f"    junction chord {r['junction_chord_in']:6.2f} in")
        print("    box margins " + "  ".join(f"{m:+6.2f}" for m in r["box_margin_in"]))

    base = 10736.1  # as-built, full OAS
    print("\n" + "=" * 92)
    print(f"  {'case':>28} {'S_ref m2':>9} {'drag N':>10} {'vs as-built':>12} {'junction ch':>12}")
    for name, r in res.items():
        print(f"  {name:>28} {r['S_ref']:9.3f} {r['drag_N']:10.1f} {r['drag_N'] / base - 1:+11.2%} {r['junction_chord_in']:12.2f}")
    print("=" * 92)

    print("\n  HOW FAR OUTBOARD THE AILERONS CAN SIT (as-built section, depth at the spar):")
    semi = 118.0 * 12.0 / 2.0
    for name, r in res.items():
        y = np.array(r["y_in"])
        d = np.array(r["depth_in"])
        print(f"\n    {name}:")
        for req in DEPTH_REQS:
            lim = outboard_limit(y, d, req)
            if lim is None:
                print(f"      {req:.0f} in depth: NOWHERE on the span")
            else:
                yl, dl = lim
                print(f"      {req:.0f} in depth: out to y = {yl:6.1f} in  ({yl / semi:5.1%} semi-span), depth {dl:.2f} in there")

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n  wrote {OUT}")
