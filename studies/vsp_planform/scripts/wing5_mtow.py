"""Wing 5 at MTOW, on the wing_comparison basis.

The coupled runs that produced wing 5 were trimmed to CRUISE weight (370,537 N).
`compare_wings.py` runs every case at MTOW 382,547 N. Mixing the two in one table
is exactly the error the study's caveats warn about, so wing 5 is re-run here at
MTOW with the same station set wing 3 uses -- wing 5 does not change t/c at the
aileron, so its depth-equivalent width is wing 3's, unchanged.

Writes wing5_design_point.json in the format compare_wings.rebuild_from consumes,
plus the t/c control points, which that helper needs to be taught to set.
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
from studies.vsp_planform.param import rear_spar_fraction    # noqa: E402
import wing2_oas as w2                                       # noqa: E402
from doe_v3 import asbuilt                                   # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")
SEMI_IN, Y_AIL = 708.0, 0.90 * 708.0
DEPTH_REQ, JUNCTION_SPAR = 6.0, 0.550
q = 0.5 * config.RHO * config.V_MS**2


def stations_and_schedule():
    """Wing 3's box exactly: same retention/toc basis as aileron_90.py."""
    af = asbuilt()
    xs = np.linspace(0.05, 0.95, 300)
    t = np.array([float(af.local_thickness(x_over_c=x)) for x in xs])
    ret_of = lambda x: float(np.interp(x, xs, t / t.max()))

    w2.apply_wing2_box()
    _, stick0, _, _ = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    y_s = np.abs(np.asarray(stick0.le[:, 1], dtype=float))
    toc_at = lambda y: float(np.interp(y, y_s, stick0.toc))

    schedule = ((356.0, 0.750), (674.9, JUNCTION_SPAR))
    spar_ail = float(rear_spar_fraction(Y_AIL, schedule))
    ret, toc = ret_of(spar_ail), toc_at(Y_AIL)
    c_req = DEPTH_REQ / (ret * toc)
    stations = ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0),
                (Y_AIL, (spar_ail - w2.FRONT_PCT) * c_req),
                (674.9, w2.JUNCTION_BOX_IN))
    return schedule, stations, c_req


def run(cp_toc, label):
    schedule, stations, c_req = stations_and_schedule()
    w2.REAR_SCHEDULE = schedule
    w2.WIDTH_STATIONS = stations
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = stations

    mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    prob, _ = ro.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
    if cp_toc is not None:
        prob.set_val("wing.t_over_c_cp", np.asarray(cp_toc))
    prob.run_model()
    s0 = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    alpha0 = trim_alpha(prob, w2.W / (q * s0))
    ro.add_optimization(prob, "plan_l", mesh, planform0, s0, mode="fixed_lift", weight=w2.W)
    if cp_toc is not None:
        prob.set_val("wing.t_over_c_cp", np.asarray(cp_toc))   # setup() reset it
    prob.set_val("alpha", alpha0, units="deg")
    prob.run_model()
    prob.run_driver()

    r = w2.evaluate(prob, regions.y_c_start)
    r["alpha"] = float(prob.get_val("alpha", units="deg")[0])
    r["twist_cp"] = prob.get_val("wing.twist_cp", units="deg").tolist()
    r["t_over_c_cp"] = (np.asarray(cp_toc).tolist() if cp_toc is not None
                        else np.asarray(prob.get_val("wing.t_over_c_cp")).tolist())
    toc = np.asarray(prob.get_val("wing.t_over_c")).ravel()
    r["toc_root"], r["toc_tip"] = float(toc[0]), float(toc[-1])
    r["chord_req_ail_in"] = c_req
    r["success"] = bool(prob.driver.result.success)
    print(f"  {label}: drag {r['drag_N']:.1f} N, S_ref {r['S_ref']:.3f} m2, "
          f"CL {r['CL']:.4f}, t/c root {r['toc_root']:.4f}", flush=True)
    return r


if __name__ == "__main__":
    # The t/c control points are DETERMINISTIC -- wing5.wing5_cp fits them to the
    # crossover-informed target profile (root 0.220, ratio 0.58, blend to the
    # as-built loft by WS 447) using nothing but the OAS spline basis. WingCalc
    # enters wing5.py only for the weight loop, and cp is held fixed across its
    # passes. So the geometry regenerates without the structural tool: prefer the
    # coupled log when it exists, otherwise recompute the same vector here.
    w5log = os.path.join(LOGS, "wing5.json")
    if os.path.exists(w5log):
        cp5 = json.load(open(w5log))["wing5"][-1]["cp"]
        print(f"wing 5 t/c cp = {[round(c, 4) for c in cp5]}  (from wing5.json)")
    else:
        from wing5 import wing5_cp
        w2.apply_wing2_box()
        _m, _s, _r, _pf = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
        _p, _ = ro.build_problem(w2.BASELINE, _m, _s, _r, _pf)
        _p.run_model()
        cp5, _t, _y = wing5_cp(_p, int(np.asarray(_p.get_val("wing.t_over_c_cp")).size))
        cp5 = list(map(float, cp5))
        print(f"wing 5 t/c cp = {[round(c, 4) for c in cp5]}  (recomputed, no WingCalc)")

    print("\nrunning wing 3 at MTOW (as-built t/c, control) ...")
    w3 = run(None, "wing 3")
    print("running wing 5 at MTOW (thickened inboard) ...")
    w5 = run(cp5, "wing 5")

    def jsonable(v):
        return v.tolist() if hasattr(v, "tolist") else v

    out = {k: {kk: jsonable(vv) for kk, vv in v.items()} for k, v in
           (("wing3_mtow", w3), ("wing5_mtow", w5))}
    out["basis"] = {"weight_N": w2.W, "note": "MTOW, same basis as wing_comparison"}
    with open(os.path.join(LOGS, "wing5_design_point.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 78)
    print(f"{'':20} {'wing 3':>14} {'wing 5':>14} {'delta':>12}")
    for k in ("drag_N", "S_ref", "CL", "induced_N", "viscous_N", "toc_root"):
        print(f"{k:20} {w3[k]:>14.4f} {w5[k]:>14.4f} {w5[k]-w3[k]:>+12.4f}")
    print("=" * 78)
    print(f"drag {100*(w5['drag_N']/w3['drag_N']-1):+.2f}% vs wing 3 at MTOW")
