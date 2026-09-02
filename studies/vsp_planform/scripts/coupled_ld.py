"""Aircraft L/D at cruise weight, with the wing sized by the structural tool.

Objective is L/D, not range: the aircraft is not range limited, so what a lighter
wing buys is fuel burn and electric range, not distance.

The accounting, with MTOW held at the aircraft's design weight:

    MTOW    = K + W_wing + payload + fuel      (K = non-wing OEW, fixed)
    fuel    = MTOW - K - payload - W_wing      <- a heavier wing eats fuel
    cruise  = MTOW - 0.5 * fuel                <- what OAS trims lift to
    W_wing  = WingCalc sized at MTOW           <- structure sees the design weight

So the wing is SIZED at MTOW but FLOWN at cruise weight. Using one for both is
worth a few percent in either the margins or the drag.

The reported metric is ENERGY, not L/D: the aircraft is not range limited, so
what a design change is worth is the energy it takes to fly, which is what burns
fuel and what drains the battery.

    P_shaft = D * V / eta          eta = 0.80, constant
    E per distance = D / eta       (J/m -- independent of speed)

Drag is the VLM's own CDi + CDv + CDw. Note this is WING drag: the OAS model
carries no fuselage, nacelles or tails, so the absolute energy is the wing's
share only. It is the same basis as every drag number already in the study, so
differences between designs are directly comparable even though the absolute
is not an aircraft-level figure.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coupled_loop as cl

MTOW_LB = 86_000.0        # fixed: this is the aircraft
K_LB = 56_000.0           # non-wing OEW (63,500 nominal OEW - 7,500 nominal wing)
PAYLOAD_LB = 17_100.0
ETA_PROP = 0.80           # propulsive efficiency, constant
NMI_M = 1852.0

W_WING_START = 7_500.0
MAX_PASSES = 5
TOL_LB = 25.0
OUT = Path(__file__).resolve().parent.parent / "out" / "logs" / "coupled_ld.json"


def mission(w_wing_lb):
    fuel = MTOW_LB - K_LB - PAYLOAD_LB - w_wing_lb
    cruise = MTOW_LB - 0.5 * fuel
    return fuel, cruise


def main():
    import numpy as np
    from studies.vsp_planform import config

    q = 0.5 * config.RHO * config.V_MS**2
    w_wing = W_WING_START
    hist = []

    for p in range(1, MAX_PASSES + 1):
        fuel, cruise = mission(w_wing)
        print(f"\n{'#'*78}\n# PASS {p}: W_wing {w_wing:.1f} -> fuel {fuel:.1f}, "
              f"cruise {cruise:.1f} lbm (MTOW fixed {MTOW_LB:.0f})\n{'#'*78}", flush=True)

        t0 = time.perf_counter()
        oas = cl.run_oas(cruise)          # optimizes at CRUISE weight
        t_oas = time.perf_counter() - t0
        st = oas["optimized"]

        d_wing = st["drag_N"]
        lift_N = cruise * cl.LB
        ld_wing = lift_N / d_wing         # straight from the VLM solution
        p_shaft_kw = d_wing * config.V_MS / ETA_PROP / 1000.0
        e_per_nmi_kwh = (d_wing / ETA_PROP) * NMI_M / 3.6e6
        e_per_km_kwh = (d_wing / ETA_PROP) * 1000.0 / 3.6e6

        deck = OUT.parent / f"ld_deck_pass{p}"
        cl.write_deck(cl.WC_DECK, deck, MTOW_LB, w_wing, oas=oas)   # SIZED at MTOW
        t0 = time.perf_counter()
        w_new = cl.run_wingcalc(deck, OUT.parent / f"ld_wc_pass{p}")
        t_wc = time.perf_counter() - t0

        resid = w_new - w_wing
        hist.append({
            "pass": p, "w_wing_in": w_wing, "w_wing_out": w_new, "residual_lb": resid,
            "fuel_lb": fuel, "cruise_lb": cruise, "mtow_lb": MTOW_LB,
            "S_ref": st["S_ref"], "CL": st["CL"], "CD_wing": st["CD"],
            "drag_wing_N": d_wing, "LD_wing": ld_wing,
            "P_shaft_kW": p_shaft_kw, "E_kWh_per_nmi": e_per_nmi_kwh,
            "E_kWh_per_km": e_per_km_kwh, "eta_prop": ETA_PROP,
            "wingbox_pct": st["wingbox_pct"], "taper_B": st["taper_B"],
            "depth_ail_in": oas["depth_ail_in"],
            "oas_s": t_oas, "wc_s": t_wc, "oas_success": oas["success"],
        })
        print(f"\n>>> PASS {p}: W_wing {w_wing:.1f} -> {w_new:.1f} ({resid:+.1f}) | "
              f"fuel {fuel:.0f} | cruise {cruise:.0f} | S_ref {st['S_ref']:.3f} | "
              f"CL {st['CL']:.4f} | L/D {ld_wing:.3f} | P {p_shaft_kw:.1f} kW "
              f"| E {e_per_nmi_kwh:.3f} kWh/nmi", flush=True)
        OUT.write_text(json.dumps(hist, indent=2))

        if abs(resid) < TOL_LB:
            print(f"\nCONVERGED after pass {p}")
            break
        w_wing = w_wing + 0.5 * resid

    print("\n" + "=" * 112)
    print(f"{'pass':>4} {'W_wing':>9} {'fuel':>8} {'cruise':>9} {'S_ref':>8} {'CL':>7} "
          f"{'drag N':>9} {'L/D':>8} {'P_kW':>9} {'kWh/nmi':>9} {'resid':>8}")
    for h in hist:
        print(f"{h['pass']:>4} {h['w_wing_out']:>9.1f} {h['fuel_lb']:>8.1f} "
              f"{h['cruise_lb']:>9.1f} {h['S_ref']:>8.3f} {h['CL']:>7.4f} "
              f"{h['drag_wing_N']:>9.1f} {h['LD_wing']:>8.3f} "
              f"{h['P_shaft_kW']:>9.1f} {h['E_kWh_per_nmi']:>9.3f} {h['residual_lb']:>+8.1f}")
    print("=" * 112)
    b, e = hist[0], hist[-1]
    print(f"eta = {ETA_PROP}, V = {config.V_MS:.3f} m/s ({config.KTAS:.0f} KTAS at "
          f"{config.ALTITUDE_FT:.0f} ft). Energy is the WING's share only.")
    print(f"seed -> converged: W_wing {b['w_wing_out']:.0f} -> {e['w_wing_out']:.0f} lb, "
          f"fuel {b['fuel_lb']:.0f} -> {e['fuel_lb']:.0f} lb, "
          f"E {b['E_kWh_per_nmi']:.3f} -> {e['E_kWh_per_nmi']:.3f} kWh/nmi "
          f"({100*(e['E_kWh_per_nmi']/b['E_kWh_per_nmi']-1):+.2f}%)")


if __name__ == "__main__":
    main()
