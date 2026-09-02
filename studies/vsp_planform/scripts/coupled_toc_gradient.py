"""Where along the span does thickness still pay?

The level sweeps answer "how thick" with one number for the whole wing. This
answers "thick WHERE" by finite-differencing each t/c spline control point:

    dR/d(t/c_i)  for i = 0..4,  at spanwise fractions 0, 0.25, 0.5, 0.75, 1.0

Each perturbation is one full coupled evaluation -- OAS re-optimizes the planform,
the geometry is exported, WingCalc re-sizes all 20 bays -- so the derivative
carries both currencies, drag from the VLM and weight from the structure.

This is the gradient the bi-level loop structurally cannot produce. The loop
passes a scalar (wing weight) between the tools; a direction needs one number per
design variable, so it has to be measured by perturbation or supplied by a
surrogate. Five stations, five runs.

Reading it: a positive dR means thickness at that station still buys range. The
station where it crosses zero is where thickening stops being worth it, and the
profile between the crossings is the shape the wing actually wants.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coupled_loop as cl
import coupled_toc_profile as tp

BASE_ROOT_TOC = 0.200        # start from the best of the level sweep
BASE_RATIO = 0.58            # as-built-like taper
DELTA_TOC = 0.010            # absolute t/c bump on one control point
OUT = Path(__file__).resolve().parent.parent / "out" / "logs" / "coupled_toc_gradient.json"


def run_cp(cp_vector, tag):
    """One coupled evaluation at an explicit t/c control-point vector."""
    from studies.vsp_planform import run_opt, config
    from studies.vsp_planform.degen_csv import read_degen_csv, lifting_surfaces
    from studies.vsp_planform.param import rear_spar_fraction
    import wing2_oas as w2
    from doe_v3 import asbuilt

    cruise = tp.MTOW_LB - 0.5 * tp.FUEL_LB

    af = asbuilt()
    xs = np.linspace(0.05, 0.95, 300)
    t = np.array([float(af.local_thickness(x_over_c=x)) for x in xs])
    ret_of = lambda x: float(np.interp(x, xs, t / t.max()))

    w2.apply_wing2_box()
    schedule = ((356.0, 0.750), (674.9, cl.JUNCTION_SPAR))
    spar_ail = float(rear_spar_fraction(cl.Y_AIL, schedule))
    ret = ret_of(spar_ail)

    base_st = ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0),
               (cl.Y_AIL, 0.0), (674.9, w2.JUNCTION_BOX_IN))
    w2.REAR_SCHEDULE = schedule
    w2.WIDTH_STATIONS = base_st
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = base_st

    mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    pre, _ = run_opt.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
    pre.set_val("wing.t_over_c_cp", cp_vector)
    pre.run_model()
    toc_v = np.asarray(pre.get_val("wing.t_over_c")).ravel()
    y_nodes = np.abs(np.asarray(pre.get_val("wing.mesh", units="m"))[0, :, 1]) / config.SCALE
    y_panel = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    o = np.argsort(y_panel)
    toc_ail = float(np.interp(cl.Y_AIL, y_panel[o], toc_v[o]))

    c_req = cl.DEPTH_REQ_IN / (ret * toc_ail)
    stations = ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0),
                (cl.Y_AIL, (spar_ail - w2.FRONT_PCT) * c_req),
                (674.9, w2.JUNCTION_BOX_IN))
    w2.WIDTH_STATIONS = stations
    config.WINGBOX_WIDTH_STATIONS = stations

    mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    prob, _ = run_opt.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
    prob.set_val("wing.t_over_c_cp", cp_vector)
    prob.run_model()
    q = 0.5 * config.RHO * config.V_MS**2
    s0 = float(prob.get_val(f"{run_opt.POINT}.wing.S_ref")[0])
    alpha0 = run_opt.trim_alpha(prob, cruise * cl.LB / (q * s0))
    run_opt.add_optimization(prob, "plan_l", mesh, planform0, s0,
                             mode="fixed_lift", weight=cruise * cl.LB)
    prob.set_val("wing.t_over_c_cp", cp_vector)      # add_optimization re-ran setup()
    prob.set_val("alpha", alpha0, units="deg")
    prob.run_model()
    prob.run_driver()

    st = run_opt._state(prob)
    toc_final = np.asarray(prob.get_val("wing.t_over_c")).ravel()
    comp = list(lifting_surfaces(read_degen_csv(config.BASELINES[w2.BASELINE])).values())[0][0]
    oas = {"mesh": np.asarray(prob.get_val("wing.mesh", units="m")), "toc": toc_final,
           "plate": comp.plate, "stick": comp.stick, "y_junction": 674.9}

    cl.write_deck(cl.WC_DECK, OUT.parent / f"deck_{tag}", tp.MTOW_LB, tp.W_WING_SEED, oas=oas)
    w_new = cl.run_wingcalc(OUT.parent / f"deck_{tag}", OUT.parent / f"wc_{tag}")

    m_batt = tp.MTOW_LB - tp.K_EX_BATT_LB - tp.PAYLOAD_LB - tp.FUEL_LB - w_new
    r_nmi = tp.ETA * (m_batt * tp.LB_KG * tp.E_STAR * 3600.0) / st["drag_N"] / tp.NMI_M
    return {"tag": tag, "cp": list(map(float, cp_vector)), "drag_N": st["drag_N"],
            "S_ref": st["S_ref"], "w_wing_lb": w_new, "m_batt_lb": m_batt,
            "R_nmi": r_nmi, "toc_ail": toc_ail, "chord_req_ail_in": c_req,
            "CL": st["CL"], "wingbox_pct": st["wingbox_pct"]}


def main():
    n_cp = 5
    base_cp = np.linspace(BASE_ROOT_TOC, BASE_ROOT_TOC * BASE_RATIO, n_cp)
    frac = np.linspace(0.0, 1.0, n_cp)
    semi_in = 708.0

    print(f"baseline cp = {np.round(base_cp, 4).tolist()}", flush=True)
    print(f"{'#'*78}\n# BASELINE\n{'#'*78}", flush=True)
    res = [run_cp(base_cp, "grad_base")]
    print(f">>> baseline: drag {res[0]['drag_N']:.1f} N | W_wing {res[0]['w_wing_lb']:.1f} lb "
          f"| R {res[0]['R_nmi']:.2f} nmi", flush=True)
    OUT.write_text(json.dumps(res, indent=2))

    for i in range(n_cp):
        cp = base_cp.copy()
        cp[i] += DELTA_TOC
        y = frac[i] * semi_in
        print(f"\n{'#'*78}\n# cp{i} (span frac {frac[i]:.2f}, y ~ {y:.0f} in) "
              f"+{DELTA_TOC:.3f} t/c\n{'#'*78}", flush=True)
        t0 = time.perf_counter()
        r = run_cp(cp, f"grad_cp{i}")
        r.update({"cp_index": i, "span_frac": float(frac[i]), "y_in": float(y),
                  "seconds": time.perf_counter() - t0})
        res.append(r)
        b = res[0]
        print(f"\n>>> cp{i}: drag {r['drag_N']-b['drag_N']:+.1f} N | "
              f"W_wing {r['w_wing_lb']-b['w_wing_lb']:+.1f} lb | "
              f"R {r['R_nmi']-b['R_nmi']:+.3f} nmi", flush=True)
        OUT.write_text(json.dumps(res, indent=2))

    b = res[0]
    print("\n" + "=" * 104)
    print(f"{'cp':>4} {'span':>6} {'y in':>7} {'d drag N':>10} {'d W_wing':>10} "
          f"{'d R nmi':>9} {'dR/dtoc':>10} {'verdict':>12}")
    for r in res[1:]:
        dR = r["R_nmi"] - b["R_nmi"]
        print(f"{r['cp_index']:>4} {r['span_frac']:>6.2f} {r['y_in']:>7.0f} "
              f"{r['drag_N']-b['drag_N']:>+10.1f} {r['w_wing_lb']-b['w_wing_lb']:>+10.1f} "
              f"{dR:>+9.3f} {dR/DELTA_TOC:>+10.1f} "
              f"{'pays' if dR > 0 else 'costs':>12}")
    print("=" * 104)
    print(f"baseline: drag {b['drag_N']:.1f} N, W_wing {b['w_wing_lb']:.1f} lb, "
          f"R {b['R_nmi']:.2f} nmi. Bump = +{DELTA_TOC:.3f} absolute t/c at one control point.")


if __name__ == "__main__":
    main()
