"""What does it cost to actually DELIVER 6 or 7 in of aft-spar depth?

The study asks for depth at the aileron but constrains box WIDTH there, at a
chord derived from the BASELINE STICK's t/c:

    c_req = depth / (retention(spar x/c) * t/c)

The optimizer meets that width exactly. But the model's order-4 t/c SplineComp
delivers 0.1253 at that station against the stick's 0.1317 -- 4.9% low -- so the
realized depth is 5.71 in, not the 6.00 asked for. Wings 3-6 and 8 all inherit
it; wing 7 clears 6 in only because its chord there is larger anyway.

So "is 6 in reachable" has never actually been tested. This closes the loop: the
requirement is re-derived from the t/c the model DELIVERS and the case re-run
until the delivered depth converges on the target. One correction pass is enough
-- the delivered t/c at that station is set by the spline and the mesh, neither
of which the design variables move much -- but it iterates to a tolerance rather
than assuming that.

Three architectures x two depths, all at MTOW on the wing_comparison basis:

  free            wing 3's parameterization: A|B re-lofted to 176 in, p free
  straight fwd    p pinned at the front spar's 0.12c, `preserved` region A rule
  constant chord  the ConstChord loft's own breakpoint at 361.7 in

Writes out/logs/depth_feasibility.json.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

from studies.vsp_planform import config, param                  # noqa: E402
import studies.vsp_planform.run_opt as ro                       # noqa: E402
from studies.vsp_planform.run_opt import POINT, trim_alpha      # noqa: E402
from studies.vsp_planform.param import baseline_planform, rear_spar_fraction  # noqa: E402
import wing2_oas as w2                                          # noqa: E402
from doe_v3 import asbuilt                                      # noqa: E402
from wing8_constchord_toc import REGION_A_AS_BUILT_IN           # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")
q = 0.5 * config.RHO * config.V_MS**2
SEMI_IN, Y_AIL = 708.0, 0.90 * 708.0
JUNCTION_SPAR = 0.550
SCHEDULE = ((356.0, 0.750), (674.9, JUNCTION_SPAR))
DEPTHS = (6.0, 7.0)
TOL_IN, MAX_PASS = 0.005, 6

# (label, region A end, region A rule, pinned straight-line fraction or None)
ARCHS = [
    ("free", w2.REGION_A_END_IN, "root_le_fixed", None),
    ("straight fwd spar", w2.REGION_A_END_IN, "preserved", w2.FRONT_PCT),
    ("constant chord", REGION_A_AS_BUILT_IN, "root_le_fixed", None),
]


def retention_fn():
    af = asbuilt()
    xs = np.linspace(0.05, 0.95, 300)
    t = np.array([float(af.local_thickness(x_over_c=x)) for x in xs])
    return lambda x: float(np.interp(x, xs, t / t.max()))


RET = retention_fn()
SPAR_AIL = float(rear_spar_fraction(Y_AIL, SCHEDULE))


def run(y_a_in, rule, pin_p, c_req):
    """Optimize with the aileron station's box width set by ``c_req``."""
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
        prob, _ = ro.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
        prob.run_model()
        s0 = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
        alpha0 = trim_alpha(prob, w2.W / (q * s0))
        ro.add_optimization(prob, "plan_l", mesh, planform0, s0, mode="fixed_lift", weight=w2.W)
        if pin_p is not None:
            prob.set_val("wing.wingbox_pct", pin_p)
        prob.set_val("alpha", alpha0, units="deg")
        prob.run_model()
        prob.run_driver()
    finally:
        param.REGION_A_RULE[w2.BASELINE] = saved
        config.WINGBOX_CHORD_PCT_BOUNDS = pct0

    r = w2.evaluate(prob, regions.y_c_start)
    # delivered t/c and chord at the aileron, from the model
    m = np.asarray(prob.get_val("wing.mesh", units="m")) / config.SCALE
    ym = np.abs(m[0, :, 1]); yp = 0.5 * (ym[:-1] + ym[1:])
    toc_del = float(np.interp(Y_AIL, yp, np.asarray(prob.get_val("wing.t_over_c")).ravel()))
    chord_del = float(np.asarray(prob.get_val("station_chord", units="m"))[3] / config.SCALE)
    r["toc_delivered"] = toc_del
    r["chord_at_aileron_in"] = chord_del
    r["depth_delivered_in"] = RET(SPAR_AIL) * toc_del * chord_del
    r["c_req_in"] = c_req
    r["success"] = bool(prob.driver.result.success)
    return r


if __name__ == "__main__":
    # the t/c the requirement was ORIGINALLY derived from: the baseline stick's
    w2.apply_wing2_box()
    _, stick0, _, _ = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    ys = np.abs(np.asarray(stick0.le[:, 1], dtype=float))
    toc_stick = float(np.interp(Y_AIL, ys, stick0.toc))
    print(f"spar at aileron {SPAR_AIL:.4f}c, retention {RET(SPAR_AIL):.4f}, "
          f"stick t/c {toc_stick:.4f}\n")

    out = {}
    for label, y_a, rule, pin_p in ARCHS:
        for depth in DEPTHS:
            key = f"{label} @ {depth:.0f} in"
            print(f"=== {key} ===", flush=True)
            toc_use = toc_stick
            hist = []
            for p in range(1, MAX_PASS + 1):
                c_req = depth / (RET(SPAR_AIL) * toc_use)
                r = run(y_a, rule, pin_p, c_req)
                err = r["depth_delivered_in"] - depth
                hist.append({"pass": p, "c_req_in": c_req, "toc_assumed": toc_use,
                             "toc_delivered": r["toc_delivered"],
                             "depth_delivered_in": r["depth_delivered_in"],
                             "drag_N": r["drag_N"], "S_ref": r["S_ref"]})
                print(f"  pass {p}: c_req {c_req:6.2f} in -> chord {r['chord_at_aileron_in']:6.2f}, "
                      f"t/c {r['toc_delivered']:.4f}, depth {r['depth_delivered_in']:5.2f} in "
                      f"({err:+.3f}), drag {r['drag_N']:9.1f} N, S_ref {r['S_ref']:.3f}", flush=True)
                if abs(err) < TOL_IN:
                    break
                toc_use = r["toc_delivered"]
            r["history"] = hist
            out[key] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}

    with open(os.path.join(LOGS, "depth_feasibility.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {os.path.join(LOGS, 'depth_feasibility.json')}")

    print("\n" + "=" * 96)
    print(f"{'case':26} {'chord @ ail':>12} {'depth':>8} {'S_ref':>9} {'drag N':>10} {'vs 6 in free':>13}")
    base = out["free @ 6 in"]["drag_N"]
    for k, v in out.items():
        print(f"{k:26} {v['chord_at_aileron_in']:>12.2f} {v['depth_delivered_in']:>8.2f} "
              f"{v['S_ref']:>9.3f} {v['drag_N']:>10.1f} {100*(v['drag_N']/base-1):>+12.2f}%")
    print("=" * 96)
