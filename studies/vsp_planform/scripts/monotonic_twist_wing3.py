"""Cost of monotonic twist through region B, ON WING 3.

`monotonic_twist.py` measured +0.297% -- but on wing 2's configuration (0.499c
kink, 7 in depth at the winglet junction). Wing 3 has a different spar schedule
(0.550c), a different binding constraint (6 in depth at 90% semi-span) and a
different optimum, so the number has to be re-measured rather than carried over.

Same constraint as before: twist non-increasing outboard across region B, applied
to `twist_abs` (the physical distribution) rather than `twist_cp`.
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
from monotonic_twist import TwistSlope, sign_changes  # noqa: E402
from aileron_90 import Y_AIL, retention_fn  # noqa: E402
from doe_v3 import asbuilt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs", "monotonic_twist_wing3.json")

q = 0.5 * config.RHO * config.V_MS**2

JUNCTION_SPAR = 0.550  # wing 3
DEPTH_REQ = 6.0


def wing3_stations(ret_of, toc_at):
    """Wing 3's constraint set: box widths plus the depth-equivalent station."""
    schedule = ((356.0, 0.750), (674.9, JUNCTION_SPAR))
    spar_ail = float(rear_spar_fraction(Y_AIL, schedule))
    c_req = DEPTH_REQ / (ret_of(spar_ail) * toc_at(Y_AIL))
    w_equiv = (spar_ail - w2.FRONT_PCT) * c_req
    stations = (
        (100.0, 65.0),
        (176.0, 65.0),
        (356.0, 55.0),
        (Y_AIL, w_equiv),
        (674.9, w2.JUNCTION_BOX_IN),
    )
    return schedule, stations


def build(monotonic, schedule, stations):
    w2.REAR_SCHEDULE = schedule
    w2.WIDTH_STATIONS = stations
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = stations

    mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)

    def attach(model, mesh_, regions_):
        # ROOT to junction, not region B only. Monotonic twist means the twist
        # falls continuously from the root outboard; restricting the constraint
        # to region B leaves the inboard bay free to climb first, which is what
        # an earlier version did -- it produced a "monotonic" wing that still
        # rose from 3.43 deg at the root to ~5 deg at y = 120 in.
        y_in = np.abs(mesh_[0, :, 1]) / config.SCALE
        idx = np.flatnonzero(y_in <= regions_.y_c_start + 1e-6)
        model.add_subsystem("twist_slope", TwistSlope(ny=y_in.size, idx=idx),
                            promotes_inputs=["twist_abs"], promotes_outputs=["dtwist"])
        print(f"    monotonicity over {idx.size} stations, y = {y_in[idx[0]]:.1f} to {y_in[idx[-1]]:.1f} in (root -> junction)")

    prob, _ = ro.build_problem(w2.BASELINE, mesh, stick, regions, planform0,
                               extra=attach if monotonic else None)
    prob.run_model()
    s0 = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    alpha0 = trim_alpha(prob, w2.W / (q * s0))

    # MUST precede add_optimization -- it ends with its own setup(), and a
    # constraint added after a setup() is silently discarded.
    if monotonic:
        prob.model.add_constraint("dtwist", upper=0.0, units="deg", ref=1.0)

    ro.add_optimization(prob, "plan_l", mesh, planform0, s0, mode="fixed_lift", weight=w2.W)
    prob.set_val("alpha", alpha0, units="deg")
    prob.run_model()
    prob.run_driver()

    r = w2.evaluate(prob, regions.y_c_start)
    r["success"] = bool(prob.driver.result.success)
    r["exit_status"] = str(prob.driver.result.exit_status)
    r["twist_abs"] = prob.get_val("twist_abs", units="deg").tolist()
    r["y_in"] = (np.abs(mesh[0, :, 1]) / config.SCALE).tolist()
    return r


if __name__ == "__main__":
    af = asbuilt()
    ret_of = retention_fn(af)

    w2.apply_wing2_box()
    _, stick0, _, _ = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    y_s = np.abs(np.asarray(stick0.le[:, 1], dtype=float))
    toc_at = lambda y: float(np.interp(y, y_s, stick0.toc))

    schedule, stations = wing3_stations(ret_of, toc_at)

    print("=" * 86)
    print(f"Monotonic twist on WING 3 (spar 0.750c -> {JUNCTION_SPAR:.3f}c, "
          f"{DEPTH_REQ:.0f} in depth at y = {Y_AIL:.1f} in)")
    print("=" * 86)

    res = {}
    for key, mono in (("free", False), ("monotonic", True)):
        r = build(mono, schedule, stations)
        res[key] = r
        print(f"\n  {key}: S_ref {r['S_ref']:7.3f} m^2  CL {r['CL']:.4f}  "
              f"drag {r['drag_N']:9.1f} N  [{r['exit_status']}]")
        print(f"    twist root {r['twist_root']:+.3f} -> tip {r['twist_tip']:+.3f} deg")
        print(f"    twist_cp " + "  ".join(f"{v:+.3f}" for v in r["twist_cp"]))

    y = res["free"]["y_in"]
    # Report over the constrained span: root to junction.
    y_a, y_c = 0.0, 674.9
    d = res["monotonic"]["drag_N"] - res["free"]["drag_N"]
    base = 10736.1

    print("\n" + "=" * 86)
    for k in ("free", "monotonic"):
        print(f"  {k:>10}  drag {res[k]['drag_N']:9.1f} N  ({res[k]['drag_N'] / base - 1:+.2%} vs as-built)"
              f"   root->junction sign changes: {sign_changes(res[k]['twist_abs'], y, y_a, y_c)}")
    print(f"  COST OF MONOTONICITY ON WING 3: {d:+.1f} N  ({d / res['free']['drag_N']:+.3%})")
    print("  (wing 2 measured +31.1 N, +0.297%)")
    print("=" * 86)

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n  wrote {OUT}")
