"""WING 7: the front spar is the straight one, and the aft spar may move.

Every design in this study so far has been built around a straight, unswept
*aft* wingbox spar -- that is what both VSP baselines were lofted to, and
`RegionPlanform` enforces it by construction rather than by constraint:

    x_le(y) = x_spar - p * c(y)

`p` is `wingbox_pct`, the chord fraction of the line held straight -- NOT an edge
of the box (`param.RegionPlanform` docstring). The box edges are the front
fraction and the rear *schedule*, and they follow the local chord. The baselines
fit p0 = 0.601 (Plan L) / 0.681 (ConstChord), which is why the aft spar is the
straight one as built.

WHICH RULE. `p` alone is not enough, because `REGION_A_RULE` decides how the
chord distribution answers when `p` moves, and both baselines ship as
`root_le_fixed` (exponent 1). That rule holds `p * c` invariant: it FREEZES the
leading-edge line and makes `p` a chord scale, `c -> c * p0/p`. Pinning p = 0.12
under it does not move the straight line to the front spar at all -- it multiplies
every chord by 5.674 (measured: S_ref 77.09 -> 344.82 m2, drag +90.7%). That run
is a bug, not a design.

The rule in which `p` means what is wanted here is `preserved` (exponent 0):
region A's chord is held, `taper_B` alone sets the chord distribution, and `p`
only selects WHICH chord fraction is the straight line. So wing 7 is `preserved`
with p pinned at 0.12. The leading edge becomes x_spar_fwd - 0.12*c, the front
spar is straight by construction, and the aft spar goes wherever the rear
schedule and the taper put it. Region B's leading-edge sweep follows

    tan(sweep_LE,B) = p * c_a0 * (1 - taper_B) / span_B

(with the exponent 0, no p0/p factor), so pinning p 0.681 -> 0.120 takes most of
the LE sweep out and pushes the sweep into the trailing edge instead. taper_B and
the twist are still free, and the box-width / aileron-depth station set is wing
3's, so the two are comparable.

The control is wing 3 exactly as published -- `root_le_fixed`, p free -- so the
comparison carries a rule change as well as a pinned p. That is unavoidable:
under wing 3's own rule the question "which spar is straight" cannot be posed,
since the LE line is frozen by construction. The two rules coincide at p = p0.

WHAT IS NOT CLAIMED. The absolute chordwise position of the straight line is a
convention, not a result: pinning p re-anchors the wing to whatever x the 0.12c
line is held at, which translates the whole planform ~59 in aft of wing 3 if the
baseline's own x_spar is reused. VLM drag is invariant to that translation, so it
does not touch any number here, but it does move the wing against the gear and cg
stations -- exactly the trap `coupling/deck.py` records for `Fwd spar X at BL0`.
`x_shift_in` therefore reports how far the planform sits from wing 3's, measured
at the root leading edge, so the translation is explicit rather than silent. It is
NOT removed from the model: VLM drag does not depend on it, and re-anchoring
would have to be done against the gear and cg stations, which this study does not
carry.

Runs at MTOW on the wing_comparison basis, alongside wing 3 as the control.
Writes out/logs/wing7_design_point.json.
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
from studies.vsp_planform.param import baseline_planform      # noqa: E402
import wing2_oas as w2                                       # noqa: E402
from wing5_mtow import stations_and_schedule                 # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")
q = 0.5 * config.RHO * config.V_MS**2

P_FRONT = w2.FRONT_PCT          # 0.12 -- the front spar fraction
PCT_BOUNDS0 = config.WINGBOX_CHORD_PCT_BOUNDS


def run(pin_p, label):
    """Optimize at MTOW. ``pin_p`` is None -- wing 3 as published, `root_le_fixed`
    with p free -- or the chord fraction of the line to hold straight, which also
    switches region A to the `preserved` rule so that p means that."""
    schedule, stations, c_req = stations_and_schedule()
    w2.REAR_SCHEDULE = schedule
    w2.WIDTH_STATIONS = stations
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = stations
    # add_optimization reads the bounds off config, so pinning is a degenerate
    # interval rather than a special case in the driver.
    config.WINGBOX_CHORD_PCT_BOUNDS = PCT_BOUNDS0 if pin_p is None else (pin_p, pin_p)

    mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    if pin_p is not None:
        # Re-measure the baseline under `preserved`. Only the rule flag changes --
        # wingbox_pct is the same fitted p0 -- but it is what makes p select the
        # straight line instead of scaling every chord by p0/p.
        planform0 = baseline_planform(stick, regions, rule="preserved")
    prob, _ = ro.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
    prob.run_model()
    s0 = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    alpha0 = trim_alpha(prob, w2.W / (q * s0))
    ro.add_optimization(prob, "plan_l", mesh, planform0, s0, mode="fixed_lift", weight=w2.W)
    if pin_p is not None:
        prob.set_val("wing.wingbox_pct", pin_p)   # setup() reset it to p0
    prob.set_val("alpha", alpha0, units="deg")
    prob.run_model()
    prob.run_driver()

    r = w2.evaluate(prob, regions.y_c_start)
    r["alpha"] = float(prob.get_val("alpha", units="deg")[0])
    r["twist_cp"] = prob.get_val("wing.twist_cp", units="deg").tolist()
    r["chord_req_ail_in"] = c_req
    r["straight_line_pct"] = float(prob.get_val("wing.wingbox_pct")[0])
    r["region_a_rule"] = planform0["rule"]
    r["sweep_B_deg"] = float(prob.get_val("wing.sweep_B", units="deg")[0])
    r["success"] = bool(prob.driver.result.success)

    # How far the planform sits from the baseline's fore/aft station, measured at
    # the root leading edge. Reported, not removed -- see the module docstring.
    m = np.asarray(prob.get_val("wing.mesh", units="m"))
    r["x_shift_in"] = float((m[0, 0, 0] - mesh[0, 0, 0]) / config.SCALE)

    print(f"  {label}: drag {r['drag_N']:.1f} N, S_ref {r['S_ref']:.3f} m2, CL {r['CL']:.4f}, "
          f"straight line at {r['straight_line_pct']:.4f}c, LE sweep_B {r['sweep_B_deg']:.3f} deg, "
          f"taper_B {r['taper_B']:.4f}, x shift {r['x_shift_in']:+.1f} in", flush=True)
    return r


if __name__ == "__main__":
    try:
        print("\nrunning wing 3 at MTOW (aft spar straight, p free -- the control) ...")
        w3 = run(None, "wing 3")
        print("running wing 7 at MTOW (FRONT spar straight, p pinned at 0.12) ...")
        w7 = run(P_FRONT, "wing 7")
    finally:
        config.WINGBOX_CHORD_PCT_BOUNDS = PCT_BOUNDS0

    def jsonable(v):
        return v.tolist() if hasattr(v, "tolist") else v

    out = {k: {kk: jsonable(vv) for kk, vv in v.items()} for k, v in
           (("wing3_mtow", w3), ("wing7_mtow", w7))}
    out["basis"] = {"weight_N": w2.W, "front_pct": P_FRONT,
                    "note": "MTOW, same basis and station set as wing_comparison"}
    with open(os.path.join(LOGS, "wing7_design_point.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {os.path.join(LOGS, 'wing7_design_point.json')}")

    print("\n" + "=" * 82)
    print(f"{'':22} {'wing 3':>14} {'wing 7':>14} {'delta':>14}")
    for k in ("drag_N", "induced_N", "viscous_N", "S_ref", "CL", "taper_B",
              "straight_line_pct", "sweep_B_deg", "junction_chord_in"):
        print(f"{k:22} {w3[k]:>14.4f} {w7[k]:>14.4f} {w7[k]-w3[k]:>+14.4f}")
    print("=" * 82)
    print(f"drag {100*(w7['drag_N']/w3['drag_N']-1):+.2f}% vs wing 3 at MTOW")
    print(f"box margins in, wing 7: {[round(x, 2) for x in w7['box_margin_in']]}")
