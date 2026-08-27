"""WING 6: wing 5 with the twist constrained to fall monotonically root -> tip.

Wing 6 is wing 5's parent: same planform parameterization, same inboard-thickened
t/c control points, same kinking rear spar and the same 6 in aileron-depth station
set. The only change is that `twist_abs` is required to be non-increasing outboard
from the root to the winglet junction, which is what makes the distribution
buildable and justifiable.

The constraint is on `twist_abs`, not `twist_cp`: a monotone set of B-spline
control points does not give a monotone spline, and it is the physical twist that
has to be manufacturable. It runs root -> junction rather than over region B
alone -- restricting it to B lets the inboard bay climb first, which produced a
"monotonic" wing 4 draft that still rose from 3.4 deg at the root to ~5 deg at
y = 120 in.

Run at MTOW on the wing_comparison basis, so wing 6 is directly comparable with
the wing 3 / wing 5 numbers in wing5_design_point.json rather than with the
cruise-weight coupled runs.

Writes out/logs/wing6_design_point.json.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

from studies.vsp_planform import config                      # noqa: E402
import studies.vsp_planform.run_opt as ro                    # noqa: E402
from studies.vsp_planform.run_opt import POINT, trim_alpha   # noqa: E402
import wing2_oas as w2                                       # noqa: E402

from monotonic_twist import TwistSlope, sign_changes         # noqa: E402
from wing5_mtow import stations_and_schedule                 # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")
q = 0.5 * config.RHO * config.V_MS**2


def run(cp_toc, monotonic, label):
    schedule, stations, c_req = stations_and_schedule()
    w2.REAR_SCHEDULE = schedule
    w2.WIDTH_STATIONS = stations
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = stations

    mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)

    def attach(model, mesh_, regions_):
        y_in = np.abs(mesh_[0, :, 1]) / config.SCALE
        idx = np.flatnonzero(y_in <= regions_.y_c_start + 1e-6)
        model.add_subsystem("twist_slope", TwistSlope(ny=y_in.size, idx=idx),
                            promotes_inputs=["twist_abs"], promotes_outputs=["dtwist"])
        print(f"    monotonicity over {idx.size} stations, y = {y_in[idx[0]]:.1f} to "
              f"{y_in[idx[-1]]:.1f} in (root -> junction)", flush=True)

    prob, _ = ro.build_problem(w2.BASELINE, mesh, stick, regions, planform0,
                               extra=attach if monotonic else None)
    if cp_toc is not None:
        prob.set_val("wing.t_over_c_cp", np.asarray(cp_toc))
    prob.run_model()
    s0 = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    alpha0 = trim_alpha(prob, w2.W / (q * s0))

    # MUST precede add_optimization: it ends with its own setup(), and a
    # constraint added after that setup() is silently discarded.
    if monotonic:
        prob.model.add_constraint("dtwist", upper=0.0, units="deg", ref=1.0)

    ro.add_optimization(prob, "plan_l", mesh, planform0, s0, mode="fixed_lift", weight=w2.W)
    if cp_toc is not None:
        prob.set_val("wing.t_over_c_cp", np.asarray(cp_toc))   # setup() reset it
    prob.set_val("alpha", alpha0, units="deg")
    prob.run_model()
    prob.run_driver()

    r = w2.evaluate(prob, regions.y_c_start)
    r["alpha"] = float(prob.get_val("alpha", units="deg")[0])
    r["twist_cp"] = prob.get_val("wing.twist_cp", units="deg").tolist()
    r["twist_abs"] = prob.get_val("twist_abs", units="deg").tolist()
    r["y_in"] = (np.abs(mesh[0, :, 1]) / config.SCALE).tolist()
    r["t_over_c_cp"] = (np.asarray(cp_toc).tolist() if cp_toc is not None
                        else np.asarray(prob.get_val("wing.t_over_c_cp")).tolist())
    toc = np.asarray(prob.get_val("wing.t_over_c")).ravel()
    r["toc_root"], r["toc_tip"] = float(toc[0]), float(toc[-1])
    r["chord_req_ail_in"] = c_req
    r["monotonic"] = bool(monotonic)
    r["success"] = bool(prob.driver.result.success)
    r["exit_status"] = str(prob.driver.result.exit_status)
    print(f"  {label}: drag {r['drag_N']:.1f} N, S_ref {r['S_ref']:.3f} m2, CL {r['CL']:.4f}, "
          f"twist {r['twist_root']:+.3f} -> {r['twist_tip']:+.3f} deg  [{r['exit_status']}]", flush=True)
    return r


if __name__ == "__main__":
    dp5 = json.load(open(os.path.join(LOGS, "wing5_design_point.json")))
    cp5 = dp5["wing5_mtow"]["t_over_c_cp"]
    print(f"wing 5 t/c cp = {[round(c, 4) for c in cp5]}")

    print("\nrunning wing 5 at MTOW (free twist, control) ...", flush=True)
    w5 = run(cp5, False, "wing 5")
    print("running WING 6 at MTOW (monotonic twist, root -> junction) ...", flush=True)
    w6 = run(cp5, True, "wing 6")

    y = w6["y_in"]
    print("\n" + "=" * 84)
    print(f"{'':22} {'wing 5':>13} {'wing 6':>13} {'delta':>12}")
    for k in ("drag_N", "induced_N", "viscous_N", "S_ref", "CL", "twist_root", "twist_tip"):
        print(f"{k:22} {w5[k]:>13.4f} {w6[k]:>13.4f} {w6[k]-w5[k]:>+12.4f}")
    print(f"{'sign changes r->junc':22} {sign_changes(w5['twist_abs'], y, 0.0, 674.9):>13d} "
          f"{sign_changes(w6['twist_abs'], y, 0.0, 674.9):>13d}")
    d = w6["drag_N"] - w5["drag_N"]
    print(f"\nCOST OF MONOTONICITY ON WING 5: {d:+.1f} N ({d / w5['drag_N']:+.3%})")
    print(f"wing 6 vs the ConstChord as-built 10736.1 N: {w6['drag_N'] / 10736.1 - 1:+.2%}")
    print("=" * 84)

    out = {"wing5_mtow": w5, "wing6_mtow": w6,
           "basis": {"weight_N": w2.W, "note": "MTOW, same basis as wing_comparison / wing5_design_point"}}
    with open(os.path.join(LOGS, "wing6_design_point.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {os.path.join(LOGS, 'wing6_design_point.json')}")
