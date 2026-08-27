"""Wing 5: wing 3 with thickness added only where it pays.

Wing 3 is wing 5's parent -- same planform parameterization, same 6 in aileron
depth at 90% semi-span, same kinking rear spar. The only change is the spanwise
t/c distribution.

Where the profile comes from. Decomposing a uniform 1.00 -> 1.13 thickening bay
by bay, each bay's battery gain against its share of the drag cost, the two cross
at WS 447 in (63% semi-span): structural benefit falls ~40x root to tip because
bending moment does, while drag cost falls only ~2.5x with chord. So wing 5
thickens the root to the swept optimum and blends back to the as-built loft by
WS 447, leaving the outer 37% of the span alone.

What it gives up, and why this has to be run rather than argued. The uniform
thickening also shrank the wing: c_req = 6 in / (retention * t/c) fell, releasing
the constraint that pins S_ref, worth part of its +2.90%. Wing 5 leaves the
aileron station at as-built thickness, so it forgoes that area relief. Whether
the drag saved by not thickening the outer wing covers the loss is the question.

Both designs are run through the same converged weight loop so the comparison is
like for like.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coupled_loop as cl

Y_CROSS = 447.0
ROOT_TOC_OPT, RATIO_OPT = 0.220, 0.58
MTOW_LB, K_LB, PAYLOAD_LB, FUEL_LB = 86_000.0, 56_000.0, 17_100.0, 5_400.0
K_EX_BATT_LB = K_LB - 16_665.6
ETA, E_STAR, LB_KG, NMI_M = 0.80, 300.0, 0.45359237, 1852.0
PASSES, TOL_LB = 4, 25.0
OUT = Path(__file__).resolve().parent.parent / "out" / "logs" / "wing5.json"


def _spline_basis(prob, n_cp):
    """Columns of d(t_over_c)/d(cp_i), so a target profile can be fitted."""
    saved = np.asarray(prob.get_val("wing.t_over_c_cp")).copy()
    cols = []
    for i in range(n_cp):
        e = np.zeros(n_cp); e[i] = 1.0
        prob.set_val("wing.t_over_c_cp", e)
        prob.run_model()
        cols.append(np.asarray(prob.get_val("wing.t_over_c")).ravel())
    prob.set_val("wing.t_over_c_cp", saved)
    prob.run_model()
    return np.column_stack(cols)


def wing5_cp(prob, n_cp):
    """Fit control points to the crossover-informed target profile."""
    from studies.vsp_planform import config
    toc_ab = np.asarray(prob.get_val("wing.t_over_c")).ravel()
    y = np.abs(np.asarray(prob.get_val("wing.mesh", units="m"))[0, :, 1]) / config.SCALE
    yp = 0.5 * (y[:-1] + y[1:])

    # the swept optimum, on the same stations
    B = _spline_basis(prob, n_cp)
    cp_opt = np.linspace(ROOT_TOC_OPT, ROOT_TOC_OPT * RATIO_OPT, n_cp)
    toc_opt = B @ cp_opt

    blend = np.clip((Y_CROSS - yp) / Y_CROSS, 0.0, 1.0)
    target = toc_ab + blend * (toc_opt - toc_ab)

    cp = lsq_linear(B, target, bounds=(0.08, 0.30)).x
    return cp, target, yp


def run(cp_vector, w_wing_in, tag):
    from studies.vsp_planform import run_opt, config
    from studies.vsp_planform.degen_csv import read_degen_csv, lifting_surfaces
    from studies.vsp_planform.param import rear_spar_fraction
    import wing2_oas as w2
    from doe_v3 import asbuilt

    cruise = MTOW_LB - 0.5 * FUEL_LB
    af = asbuilt()
    xs = np.linspace(0.05, 0.95, 300)
    t = np.array([float(af.local_thickness(x_over_c=x)) for x in xs])
    ret = float(np.interp(
        float(rear_spar_fraction(cl.Y_AIL, ((356.0, 0.750), (674.9, cl.JUNCTION_SPAR)))),
        xs, t / t.max()))
    spar_ail = float(rear_spar_fraction(cl.Y_AIL, ((356.0, 0.750), (674.9, cl.JUNCTION_SPAR))))

    w2.apply_wing2_box()
    schedule = ((356.0, 0.750), (674.9, cl.JUNCTION_SPAR))
    base_st = ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0),
               (cl.Y_AIL, 0.0), (674.9, w2.JUNCTION_BOX_IN))
    w2.REAR_SCHEDULE = schedule
    w2.WIDTH_STATIONS = base_st
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = base_st

    mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    pre, _ = run_opt.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
    n_cp = int(np.asarray(pre.get_val("wing.t_over_c_cp")).size)
    cp = cp_vector if cp_vector is not None else np.asarray(pre.get_val("wing.t_over_c_cp")).copy()
    pre.set_val("wing.t_over_c_cp", cp)
    pre.run_model()
    tv = np.asarray(pre.get_val("wing.t_over_c")).ravel()
    yy = np.abs(np.asarray(pre.get_val("wing.mesh", units="m"))[0, :, 1]) / config.SCALE
    yp = 0.5 * (yy[:-1] + yy[1:]); o = np.argsort(yp)
    toc_ail = float(np.interp(cl.Y_AIL, yp[o], tv[o]))

    c_req = cl.DEPTH_REQ_IN / (ret * toc_ail)
    stations = ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0),
                (cl.Y_AIL, (spar_ail - w2.FRONT_PCT) * c_req),
                (674.9, w2.JUNCTION_BOX_IN))
    w2.WIDTH_STATIONS = stations
    config.WINGBOX_WIDTH_STATIONS = stations

    mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    prob, _ = run_opt.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
    prob.set_val("wing.t_over_c_cp", cp)
    prob.run_model()
    q = 0.5 * config.RHO * config.V_MS**2
    s0 = float(prob.get_val(f"{run_opt.POINT}.wing.S_ref")[0])
    alpha0 = run_opt.trim_alpha(prob, cruise * cl.LB / (q * s0))
    run_opt.add_optimization(prob, "plan_l", mesh, planform0, s0,
                             mode="fixed_lift", weight=cruise * cl.LB)
    prob.set_val("wing.t_over_c_cp", cp)
    prob.set_val("alpha", alpha0, units="deg")
    prob.run_model()
    prob.run_driver()

    st = run_opt._state(prob)
    toc_final = np.asarray(prob.get_val("wing.t_over_c")).ravel()
    comp = list(lifting_surfaces(read_degen_csv(config.BASELINES[w2.BASELINE])).values())[0][0]
    oas = {"mesh": np.asarray(prob.get_val("wing.mesh", units="m")), "toc": toc_final,
           "plate": comp.plate, "stick": comp.stick, "y_junction": 674.9}
    cl.write_deck(cl.WC_DECK, OUT.parent / f"deck_{tag}", MTOW_LB, w_wing_in, oas=oas)
    w_new = cl.run_wingcalc(OUT.parent / f"deck_{tag}", OUT.parent / f"wc_{tag}")

    m_batt = MTOW_LB - K_EX_BATT_LB - PAYLOAD_LB - FUEL_LB - w_new
    return {"drag_N": st["drag_N"], "S_ref": st["S_ref"], "CL": st["CL"], "L/D": st["L/D"],
            "w_wing_lb": w_new, "m_batt_lb": m_batt, "toc_ail": toc_ail,
            "chord_req_ail_in": c_req, "wingbox_pct": st["wingbox_pct"],
            "toc_root": float(toc_final[0]), "toc_tip": float(toc_final[-1]),
            "R_nmi": ETA * (m_batt * LB_KG * E_STAR * 3600.0) / st["drag_N"] / NMI_M,
            "cp": list(map(float, cp)), "success": bool(prob.driver.result.success)}


def converge(cp, label):
    w = 8440.1
    hist = []
    for p in range(1, PASSES + 1):
        print(f"\n{'#'*78}\n# {label} pass {p}: W_wing in {w:.1f}\n{'#'*78}", flush=True)
        r = run(cp, w, f"{label}_p{p}")
        r["pass"], r["w_wing_in"] = p, w
        r["residual_lb"] = r["w_wing_lb"] - w
        hist.append(r)
        print(f">>> {label} p{p}: W_wing {w:.1f} -> {r['w_wing_lb']:.1f} "
              f"({r['residual_lb']:+.1f}) | drag {r['drag_N']:.1f} N | "
              f"S_ref {r['S_ref']:.3f} | R {r['R_nmi']:.2f} nmi", flush=True)
        if abs(r["residual_lb"]) < TOL_LB:
            break
        w += 0.5 * r["residual_lb"]
    return hist


def main():
    from studies.vsp_planform import run_opt
    import wing2_oas as w2
    w2.apply_wing2_box()
    mesh, stick, regions, pf0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    prob, _ = run_opt.build_problem(w2.BASELINE, mesh, stick, regions, pf0)
    prob.run_model()
    n_cp = int(np.asarray(prob.get_val("wing.t_over_c_cp")).size)
    cp5, target, yp = wing5_cp(prob, n_cp)
    print(f"wing 5 cp = {np.round(cp5, 4).tolist()}")

    out = {"wing3": converge(None, "wing3"), "wing5": converge(cp5, "wing5")}
    OUT.write_text(json.dumps(out, indent=2))

    a, b = out["wing3"][-1], out["wing5"][-1]
    print("\n" + "=" * 96)
    print(f"{'':22} {'wing 3':>14} {'wing 5':>14} {'delta':>14}")
    for k, f in [("t/c root", "toc_root"), ("t/c at aileron", "toc_ail"),
                 ("chord req ail in", "chord_req_ail_in"), ("S_ref m2", "S_ref"),
                 ("drag N", "drag_N"), ("L/D", "L/D"), ("wing weight lb", "w_wing_lb"),
                 ("battery lb", "m_batt_lb"), ("elec range nmi", "R_nmi")]:
        print(f"{k:22} {a[f]:>14.4f} {b[f]:>14.4f} {b[f]-a[f]:>+14.4f}")
    print("=" * 96)
    print(f"wing 5 vs wing 3: {100*(b['R_nmi']/a['R_nmi']-1):+.2f}% electric range, "
          f"{b['w_wing_lb']-a['w_wing_lb']:+.1f} lb wing, {b['drag_N']-a['drag_N']:+.1f} N drag")


if __name__ == "__main__":
    main()
