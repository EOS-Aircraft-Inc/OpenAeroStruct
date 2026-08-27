"""WING 8: the CONSTANT-CHORD planform, with thickness added instead of chord.

Every design from wing 2 on re-lofts region A from the baseline's 361.7 in down
to 176 in (the inboard nacelle), so none of them is a constant-chord wing -- the
constant bay survives over 25% of the semi-span rather than 51%. That re-loft is
where most of wing 3's -6.52% comes from, and wing 5 inherits it untouched. So
"the constant-chord wing with a higher t/c" has never actually been run, and the
question it answers is which half of wing 5 is doing the work: the planform, or
the thickness.

Wing 8 keeps the ConstChord loft's own A|B breakpoint. `reloft_region_a` is
documented as the identity when handed the breakpoint the baseline already has,
so the same code path is used rather than a second one -- the wing that comes out
is the as-built chord distribution, and `taper_B` and the twist optimize from
there exactly as in every other case.

Everything else is held to wing 3 / wing 5: MTOW 382,547 N, the kinking rear spar
(0.750 -> 0.550), and the same box-width and 6 in aileron-depth station set. The
station set is deliberately NOT recomputed from the thicker sections -- it is the
constraint the other cases were run against, and moving it would confound the
comparison with a constraint change. That makes the aileron requirement
conservative for the thickened variants, which is the safe direction.

Three cases, all at MTOW on the wing_comparison basis:

  base       constant chord, as-built t/c (0.178 root -> 0.100 tip)
  inboard    wing 5's profile: root to 0.220, blended back to the as-built loft
             by WS 447, outer 37% of the span untouched
  taper      the swept optimum over the WHOLE span: a straight t/c taper from
             0.220 at the root to 0.128 at the tip, no blend

`inboard` isolates wing 5's thickness treatment on a constant-chord planform;
`taper` is the uniform root-to-tip version wing 5 rejected inboard-only, priced
here on aero alone. Neither carries a weight: the structural half of that trade
needs WingCalc, and this script is pure OAS.

Writes out/logs/wing8_design_point.json.
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import lsq_linear

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

from studies.vsp_planform import config                      # noqa: E402
import studies.vsp_planform.run_opt as ro                    # noqa: E402
from studies.vsp_planform.run_opt import POINT, trim_alpha   # noqa: E402
import wing2_oas as w2                                       # noqa: E402
from wing5 import ROOT_TOC_OPT, RATIO_OPT, Y_CROSS, _spline_basis   # noqa: E402
from wing5_mtow import stations_and_schedule                 # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")
q = 0.5 * config.RHO * config.V_MS**2

# The ConstChord loft's own breakpoint. reloft_region_a snaps to the nearest
# native section, so this reproduces the baseline exactly.
REGION_A_AS_BUILT_IN = 361.70495639569657


def build(cp_toc=None):
    """The constant-chord baseline under wing 3's box, ready to optimize."""
    schedule, stations, c_req = stations_and_schedule()
    w2.REAR_SCHEDULE = schedule
    w2.WIDTH_STATIONS = stations
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = stations

    mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, REGION_A_AS_BUILT_IN)
    prob, _ = ro.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
    if cp_toc is not None:
        prob.set_val("wing.t_over_c_cp", np.asarray(cp_toc))
    prob.run_model()
    return prob, mesh, stick, regions, planform0, c_req


def toc_targets(prob, n_cp):
    """Control points for the two thickened profiles, on THIS planform.

    Same construction as wing5.wing5_cp -- fit the spline to a target sampled on
    the model's own panel stations -- but the basis is rebuilt here because the
    spanwise station distribution follows the breakpoint, which has moved.
    """
    toc_ab = np.asarray(prob.get_val("wing.t_over_c")).ravel()
    y = np.abs(np.asarray(prob.get_val("wing.mesh", units="m"))[0, :, 1]) / config.SCALE
    yp = 0.5 * (y[:-1] + y[1:])

    B = _spline_basis(prob, n_cp)
    toc_opt = B @ np.linspace(ROOT_TOC_OPT, ROOT_TOC_OPT * RATIO_OPT, n_cp)

    blend = np.clip((Y_CROSS - yp) / Y_CROSS, 0.0, 1.0)
    cp_inboard = lsq_linear(B, toc_ab + blend * (toc_opt - toc_ab), bounds=(0.08, 0.30)).x
    cp_taper = lsq_linear(B, toc_opt, bounds=(0.08, 0.30)).x
    return list(map(float, cp_inboard)), list(map(float, cp_taper)), toc_ab, yp


def run(cp_toc, label):
    prob, mesh, stick, regions, planform0, c_req = build(cp_toc)
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
    r["t_over_c_cp"] = (list(map(float, cp_toc)) if cp_toc is not None
                        else np.asarray(prob.get_val("wing.t_over_c_cp")).ravel().tolist())
    toc = np.asarray(prob.get_val("wing.t_over_c")).ravel()
    r["toc_root"], r["toc_tip"] = float(toc[0]), float(toc[-1])
    r["region_a_end_in"] = float(regions.y_a_end)
    r["chord_req_ail_in"] = c_req
    r["success"] = bool(prob.driver.result.success)
    print(f"  {label}: drag {r['drag_N']:.1f} N, S_ref {r['S_ref']:.3f} m2, CL {r['CL']:.4f}, "
          f"t/c {r['toc_root']:.4f} -> {r['toc_tip']:.4f}, taper_B {r['taper_B']:.4f}", flush=True)
    return r


if __name__ == "__main__":
    print("fitting the t/c profiles on the constant-chord planform ...")
    prob0, *_ = build()
    n_cp = int(np.asarray(prob0.get_val("wing.t_over_c_cp")).size)
    cp_in, cp_tap, toc_ab, yp = toc_targets(prob0, n_cp)
    print(f"  as-built t/c   {toc_ab[0]:.4f} -> {toc_ab[-1]:.4f}")
    print(f"  inboard cp     {[round(c, 4) for c in cp_in]}")
    print(f"  full-taper cp  {[round(c, 4) for c in cp_tap]}")

    print("\nrunning constant chord, as-built t/c (control) ...")
    base = run(None, "base")
    print("running constant chord, wing 5 inboard t/c ...")
    inb = run(cp_in, "inboard")
    print("running constant chord, t/c tapered root to tip ...")
    tap = run(cp_tap, "taper")

    def jsonable(v):
        return v.tolist() if hasattr(v, "tolist") else v

    out = {k: {kk: jsonable(vv) for kk, vv in v.items()} for k, v in
           (("constchord_asbuilt", base), ("constchord_toc_inboard", inb),
            ("constchord_toc_taper", tap))}
    out["basis"] = {"weight_N": w2.W, "region_a_end_in": REGION_A_AS_BUILT_IN,
                    "note": "MTOW, wing 3 station set and rear-spar schedule, no re-loft"}
    with open(os.path.join(LOGS, "wing8_design_point.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {os.path.join(LOGS, 'wing8_design_point.json')}")

    print("\n" + "=" * 88)
    print(f"{'':20} {'as-built':>14} {'inboard t/c':>14} {'tapered t/c':>14}")
    for k in ("drag_N", "induced_N", "viscous_N", "S_ref", "CL", "toc_root", "toc_tip", "taper_B"):
        print(f"{k:20} {base[k]:>14.4f} {inb[k]:>14.4f} {tap[k]:>14.4f}")
    print("=" * 88)
    for n, c in (("inboard", inb), ("taper", tap)):
        print(f"{n:8} drag {100*(c['drag_N']/base['drag_N']-1):+.2f}% vs constant chord, as-built t/c")
