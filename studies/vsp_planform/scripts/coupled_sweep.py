"""Price a structural requirement in drag AND wing weight at the same time.

The study has always had to price a box requirement in drag alone, because the
structural side was a hand-derived width proxy. With WingCalc coupled, each point
returns both currencies: the drag the requirement costs, and the wing weight it
buys or spends. That trade has never been on one page.

Swept knob: the box width required at y = 356 in. The study's own handoff calls
this "the binding inboard station ... as soft a pick as the 25 in junction was",
with "cost of moving it unmeasured" as open item #1.

One coupled pass per point, not a full fixed point: the wing-weight loop gain is
measured at -0.03 (W_out moved -21.8 lb for +683 lb of W_in), so a single pass
seeded near the converged weight lands within ~1%. Each point reports its own
residual so a point that needs another pass is visible rather than assumed.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coupled_loop as cl

WIDTHS_356 = [45.0, 50.0, 55.0, 60.0, 65.0]   # 55.0 is the current requirement
W_WING_SEED = 8540.8                          # converged pass-1 value
OUT = Path(__file__).resolve().parent.parent / "out" / "logs" / "coupled_sweep.json"


def run_point(w356):
    """One coupled evaluation at this inboard box width."""
    import numpy as np
    from studies.vsp_planform import run_opt, config
    from studies.vsp_planform.degen_csv import read_degen_csv, lifting_surfaces
    from studies.vsp_planform.param import rear_spar_fraction
    import wing2_oas as w2
    from doe_v3 import asbuilt

    mtow, cruise = cl.weights(W_WING_SEED)

    af = asbuilt()
    xs = np.linspace(0.05, 0.95, 300)
    t = np.array([float(af.local_thickness(x_over_c=x)) for x in xs])
    ret_of = lambda x: float(np.interp(x, xs, t / t.max()))

    w2.apply_wing2_box()
    _, stick0, _, _ = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    y_s = np.abs(np.asarray(stick0.le[:, 1], dtype=float))
    toc_at = lambda y: float(np.interp(y, y_s, stick0.toc))

    schedule = ((356.0, 0.750), (674.9, cl.JUNCTION_SPAR))
    spar_ail = float(rear_spar_fraction(cl.Y_AIL, schedule))
    ret, toc = ret_of(spar_ail), toc_at(cl.Y_AIL)
    w_equiv = (spar_ail - w2.FRONT_PCT) * (cl.DEPTH_REQ_IN / (ret * toc))

    stations = ((100.0, 65.0), (176.0, 65.0), (356.0, w356),
                (cl.Y_AIL, w_equiv), (674.9, w2.JUNCTION_BOX_IN))
    w2.REAR_SCHEDULE = schedule
    w2.WIDTH_STATIONS = stations
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = stations

    mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    prob, _ = run_opt.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
    prob.run_model()
    q = 0.5 * config.RHO * config.V_MS**2
    s0 = float(prob.get_val(f"{run_opt.POINT}.wing.S_ref")[0])
    alpha0 = run_opt.trim_alpha(prob, cruise * cl.LB / (q * s0))
    run_opt.add_optimization(prob, "plan_l", mesh, planform0, s0,
                             mode="fixed_lift", weight=cruise * cl.LB)
    prob.set_val("alpha", alpha0, units="deg")
    prob.run_model()
    prob.run_driver()
    st = run_opt._state(prob)

    comp = list(lifting_surfaces(read_degen_csv(config.BASELINES[w2.BASELINE])).values())[0][0]
    oas = {"mesh": np.asarray(prob.get_val("wing.mesh", units="m")),
           "toc": np.asarray(prob.get_val("wing.t_over_c")).ravel(),
           "plate": comp.plate, "stick": comp.stick, "y_junction": 674.9}

    tag = f"w356_{int(w356)}"
    deck = OUT.parent / f"deck_{tag}"
    cl.write_deck(cl.WC_DECK, deck, mtow, W_WING_SEED, oas=oas)
    w_new = cl.run_wingcalc(deck, OUT.parent / f"wc_{tag}")

    return {"w356_in": w356, "drag_N": st["drag_N"], "S_ref": st["S_ref"],
            "CL": st["CL"], "L/D": st["L/D"], "wingbox_pct": st["wingbox_pct"],
            "taper_B": st["taper_B"], "w_wing_lb": w_new,
            "residual_lb": w_new - W_WING_SEED,
            "mtow_lb": mtow - W_WING_SEED + w_new,
            "success": bool(prob.driver.result.success)}


def main():
    res = []
    for w in WIDTHS_356:
        print(f"\n{'#'*78}\n# y=356 box width = {w:.1f} in\n{'#'*78}", flush=True)
        t0 = time.perf_counter()
        r = run_point(w)
        r["seconds"] = time.perf_counter() - t0
        res.append(r)
        print(f"\n>>> w356 {w:5.1f} in | drag {r['drag_N']:9.1f} N | S_ref {r['S_ref']:7.3f} "
              f"| W_wing {r['w_wing_lb']:8.1f} lb | MTOW {r['mtow_lb']:9.1f} lb", flush=True)
        OUT.write_text(json.dumps(res, indent=2))

    base = next((x for x in res if x["w356_in"] == 55.0), res[0])
    print("\n" + "=" * 104)
    print(f"{'w356':>6} {'drag N':>10} {'d drag':>9} {'S_ref':>8} {'W_wing':>9} "
          f"{'d W_wing':>9} {'MTOW':>10} {'wb_pct':>7} {'resid':>8}")
    for r in res:
        print(f"{r['w356_in']:>6.1f} {r['drag_N']:>10.1f} "
              f"{r['drag_N']-base['drag_N']:>+9.1f} {r['S_ref']:>8.3f} "
              f"{r['w_wing_lb']:>9.1f} {r['w_wing_lb']-base['w_wing_lb']:>+9.1f} "
              f"{r['mtow_lb']:>10.1f} {r['wingbox_pct']:>7.4f} {r['residual_lb']:>+8.1f}")
    print("=" * 104)
    print(f"(deltas vs the current {base['w356_in']:.0f} in requirement)")


if __name__ == "__main__":
    main()
