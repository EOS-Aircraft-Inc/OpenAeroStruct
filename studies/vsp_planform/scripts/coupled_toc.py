"""Sweep t/c along the span and price it in BOTH currencies.

OAS prices thickness in drag (Raymer form factor, viscous_drag.py:103).
WingCalc prices it in weight (real airfoil contours -> real box depth).
Neither needs the other's job -- the sweep just puts the two curves side by side.

t/c is NOT made a design variable. OAS sees thickness only as a penalty, so a
drag-only optimizer drives it straight to its lower bound; that answers nothing.
Instead the whole baseline distribution is scaled by a factor and held, and each
point is optimized, exported and sized.

Three effects move together as t/c rises:

  drag   up    -- thicker sections, higher form factor
  weight down  -- deeper box carries the moment with less material
  S_ref  down  -- the 6 in aileron depth needs chord c_req = depth/(retention*t/c),
                  so thickening RELAXES the constraint that has pinned S_ref at
                  77.093 m^2 in every run so far

The third is the one worth watching: it is a drag saving bought BY thickening,
working against the direct viscous penalty.

Points are ranked on ELECTRIC RANGE. Wing weight is traded against battery at
fixed MTOW, so

    m_batt  = MTOW - K_ex_batt - payload - fuel - W_wing
    R_elec  = eta * e_star * m_batt / D      ->  proportional to m_batt / D

eta and e_star are constants, so m_batt/D ranks designs without needing the
battery specific energy at all. It is the one scalar that sees both sides: a
heavier wing cuts the numerator, more drag grows the denominator.

Calibration check against the aircraft mass export (Batteries 16,665.6 lb): at
its Wing structure total of 7,460 lb this gives 16,705.6 lb, 0.24% apart.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coupled_loop as cl

# Multipliers on the as-built distribution (root 0.177, tip 0.125), chosen to keep
# the ROOT inside the conventional 0.10-0.20 band: 0.150 / 0.168 / 0.177 / 0.186 /
# 0.200. The as-built root is already 0.177, so there is little headroom upward --
# which is itself part of the answer about how much thickness is available.
TOC_SCALES = [0.85, 0.95, 1.00, 1.05, 1.13]
W_WING_SEED = 8422.8          # converged value from the fixed-t/c loop
MTOW_LB, K_LB, PAYLOAD_LB = 86_000.0, 56_000.0, 17_100.0
BATT_LB_BOOK = 16_665.6            # from the aircraft mass export
K_EX_BATT_LB = K_LB - BATT_LB_BOOK # non-wing OEW with the battery taken out
FUEL_LB = 5_400.0                  # fixed: wing weight now trades against battery
ETA, NMI_M = 0.80, 1852.0
OUT = Path(__file__).resolve().parent.parent / "out" / "logs" / "coupled_toc.json"


def run(scale):
    from studies.vsp_planform import run_opt, config
    from studies.vsp_planform.degen_csv import read_degen_csv, lifting_surfaces
    from studies.vsp_planform.param import rear_spar_fraction
    import wing2_oas as w2
    from doe_v3 import asbuilt

    fuel = FUEL_LB
    cruise = MTOW_LB - 0.5 * fuel

    af = asbuilt()
    xs = np.linspace(0.05, 0.95, 300)
    t = np.array([float(af.local_thickness(x_over_c=x)) for x in xs])
    ret_of = lambda x: float(np.interp(x, xs, t / t.max()))

    w2.apply_wing2_box()
    _, stick0, _, _ = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    y_s = np.abs(np.asarray(stick0.le[:, 1], dtype=float))

    schedule = ((356.0, 0.750), (674.9, cl.JUNCTION_SPAR))
    spar_ail = float(rear_spar_fraction(cl.Y_AIL, schedule))
    ret = ret_of(spar_ail)

    # The depth requirement tracks the scaled t/c. At fixed t/c the width proxy is
    # exact (depth is strictly proportional to chord), so scaling t/c here keeps it
    # exact rather than leaving the constraint keyed to the baseline loft.
    toc_ail = float(np.interp(cl.Y_AIL, y_s, stick0.toc)) * scale
    c_req = cl.DEPTH_REQ_IN / (ret * toc_ail)
    w_equiv = (spar_ail - w2.FRONT_PCT) * c_req

    stations = ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0),
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

    # add_optimization ends with its own setup(), which resets inputs -- so the
    # scaled t/c has to be applied AFTER it and before the driver runs.
    cp0 = np.asarray(prob.get_val("wing.t_over_c_cp")).copy()
    prob.set_val("wing.t_over_c_cp", cp0 * scale)
    prob.set_val("alpha", alpha0, units="deg")
    prob.run_model()
    prob.run_driver()

    st = run_opt._state(prob)
    toc_final = np.asarray(prob.get_val("wing.t_over_c")).ravel()
    chords = prob.get_val("station_chord", units="m") / config.SCALE
    depth_in = ret * toc_ail * float(chords[3])

    comp = list(lifting_surfaces(read_degen_csv(config.BASELINES[w2.BASELINE])).values())[0][0]
    oas = {"mesh": np.asarray(prob.get_val("wing.mesh", units="m")), "toc": toc_final,
           "plate": comp.plate, "stick": comp.stick, "y_junction": 674.9}

    tag = f"toc{int(round(scale*100))}"
    cl.write_deck(cl.WC_DECK, OUT.parent / f"deck_{tag}", MTOW_LB, W_WING_SEED, oas=oas)
    w_new = cl.run_wingcalc(OUT.parent / f"deck_{tag}", OUT.parent / f"wc_{tag}")

    return {"scale": scale, "drag_N": st["drag_N"], "S_ref": st["S_ref"],
            "CL": st["CL"], "L/D": st["L/D"], "w_wing_lb": w_new,
            "toc_root": float(toc_final[0]), "toc_tip": float(toc_final[-1]),
            "toc_ail": toc_ail, "chord_req_ail_in": c_req,
            "chord_ail_in": float(chords[3]), "depth_ail_in": depth_in,
            "wingbox_pct": st["wingbox_pct"], "taper_B": st["taper_B"],
            "E_kWh_nmi": (st["drag_N"] / ETA) * NMI_M / 3.6e6,
            "m_batt_lb": MTOW_LB - K_EX_BATT_LB - PAYLOAD_LB - FUEL_LB - w_new,
            "range_factor": (MTOW_LB - K_EX_BATT_LB - PAYLOAD_LB - FUEL_LB - w_new) / st["drag_N"],
            "success": bool(prob.driver.result.success)}


def main():
    res = []
    for s in TOC_SCALES:
        print(f"\n{'#'*78}\n# t/c scale = {s:.2f}\n{'#'*78}", flush=True)
        t0 = time.perf_counter()
        r = run(s)
        r["seconds"] = time.perf_counter() - t0
        res.append(r)
        print(f"\n>>> scale {s:.2f} | t/c ail {r['toc_ail']:.4f} | c_req {r['chord_req_ail_in']:.2f} in "
              f"| S_ref {r['S_ref']:.3f} | drag {r['drag_N']:.1f} N | W_wing {r['w_wing_lb']:.1f} lb", flush=True)
        OUT.write_text(json.dumps(res, indent=2))

    base = next((x for x in res if x["scale"] == 1.00), res[0])
    print("\n" + "=" * 118)
    print(f"{'scale':>6} {'t/c rt':>7} {'t/c tip':>8} {'t/c ail':>8} {'c_req':>8} {'S_ref':>8} "
          f"{'drag N':>9} {'d drag':>8} {'W_wing':>9} {'d W':>8} {'m_batt':>9} {'R_elec':>9}")
    for r in res:
        dr = 100.0 * (r["range_factor"] / base["range_factor"] - 1.0)
        print(f"{r['scale']:>6.2f} {r['toc_root']:>7.4f} {r['toc_tip']:>8.4f} "
              f"{r['toc_ail']:>8.4f} {r['chord_req_ail_in']:>8.2f} "
              f"{r['S_ref']:>8.3f} {r['drag_N']:>9.1f} {r['drag_N']-base['drag_N']:>+8.1f} "
              f"{r['w_wing_lb']:>9.1f} {r['w_wing_lb']-base['w_wing_lb']:>+8.1f} "
              f"{r['m_batt_lb']:>9.1f} {dr:>+8.2f}%")
    print("=" * 118)
    print("R_elec = electric range vs the as-built t/c, proportional to m_batt/D.")
    best = max(res, key=lambda r: r["range_factor"])
    print(f"best: t/c scale {best['scale']:.2f} -> "
          f"{100*(best['range_factor']/base['range_factor']-1):+.2f}% electric range "
          f"(drag {best['drag_N']-base['drag_N']:+.1f} N, wing {best['w_wing_lb']-base['w_wing_lb']:+.1f} lb)")


if __name__ == "__main__":
    main()
