"""Electric cruise range from a coupled sweep result.

    m_batt = MTOW - K_ex_batt - payload - fuel - W_wing      [lb -> kg]
    R      = eta * e_star * m_batt / D                       [J/N = m]

D = D_wing + D0.

  D_wing  the VLM's own wing drag, recomputed from the geometry at every point,
          so it moves with t/c, area and CL.
  D0      a FIXED allowance in newtons for everything OAS does not model --
          fuselage, nacelles, tails. Set once from CD0 * q * S_ref_baseline and
          then held constant: those components do not shrink when the wing does,
          so carrying D0 as a coefficient on a moving S_ref would invent an area
          penalty that is not physical.

Both columns are reported because the choice changes the ANSWER, not just the
absolute. R = eta*E/D, so a constant D0 in the denominator dilutes the effect of
a wing-drag change while leaving the battery term untouched -- with D0 included,
WEIGHT matters relatively more than drag, and a design picked on wing-only drag
is biased toward the thin, low-drag, heavy end.
"""

import json
import sys
from pathlib import Path

LB_KG = 0.45359237
ETA = 0.80
E_STAR_WH_KG = 300.0
NMI_M = 1852.0
KM_M = 1000.0

MTOW_LB, K_LB, PAYLOAD_LB, FUEL_LB = 86_000.0, 56_000.0, 17_100.0, 5_400.0
BATT_LB_BOOK = 16_665.6
K_EX_BATT_LB = K_LB - BATT_LB_BOOK
CD0_OTHER = 0.03            # fuselage + nacelles + tails, on the baseline S_ref
S_REF_BASE = 77.093         # m^2, the reference area D0 is fixed against


def batt_lb(w_wing_lb):
    return MTOW_LB - K_EX_BATT_LB - PAYLOAD_LB - FUEL_LB - w_wing_lb


def range_m(m_batt_lb, drag_N):
    e_j = m_batt_lb * LB_KG * E_STAR_WH_KG * 3600.0
    return ETA * e_j / drag_N


def main(path):
    rows = json.loads(Path(path).read_text())
    from studies.vsp_planform import config
    q = 0.5 * config.RHO * config.V_MS**2

    d0 = CD0_OTHER * q * S_REF_BASE      # fixed, once
    for r in rows:
        mb = batt_lb(r["w_wing_lb"])
        r["m_batt_lb"] = mb
        r["D0_N"] = d0
        r["drag_total_N"] = r["drag_N"] + d0
        r["R_wing_nmi"] = range_m(mb, r["drag_N"]) / NMI_M
        r["R_nmi"] = range_m(mb, r["drag_N"] + d0) / NMI_M
        r["R_km"] = range_m(mb, r["drag_N"] + d0) / KM_M

    key = "scale" if "scale" in rows[0] else "pass"
    base = next((x for x in rows if x.get("scale") == 1.00), rows[0])

    print(f"e* = {E_STAR_WH_KG:.0f} Wh/kg, eta = {ETA}, fuel {FUEL_LB:.0f} lb fixed, "
          f"MTOW {MTOW_LB:.0f} lb")
    print("=" * 112)
    print(f"D0 = {CD0_OTHER} * q * {S_REF_BASE:.3f} = {d0:.1f} N, fixed "
          f"(fuselage + nacelles + tails)")
    print("=" * 120)
    print(f"{key:>7} {'W_wing':>9} {'m_batt':>9} {'D_wing':>9} {'D_tot':>9} {'S_ref':>8} "
          f"{'R nmi':>8} {'vs base':>8} {'R wing':>8} {'vs base':>8}")
    for r in rows:
        d = 100.0 * (r["R_nmi"] / base["R_nmi"] - 1.0)
        dw = 100.0 * (r["R_wing_nmi"] / base["R_wing_nmi"] - 1.0)
        print(f"{r[key]:>7} {r['w_wing_lb']:>9.1f} {r['m_batt_lb']:>9.1f} "
              f"{r['drag_N']:>9.1f} {r['drag_total_N']:>9.1f} {r['S_ref']:>8.3f} "
              f"{r['R_nmi']:>8.1f} {d:>+7.2f}% {r['R_wing_nmi']:>8.1f} {dw:>+7.2f}%")
    print("=" * 120)
    b = max(rows, key=lambda r: r["R_nmi"])
    bw = max(rows, key=lambda r: r["R_wing_nmi"])
    print(f"best with D0    : {key} {b[key]} -> {b['R_nmi']:.1f} nmi "
          f"({100*(b['R_nmi']/base['R_nmi']-1):+.2f}%), drag {b['drag_N']:.1f} N, "
          f"wing {b['w_wing_lb']:.1f} lb")
    print(f"best wing-only  : {key} {bw[key]} -> {bw['R_wing_nmi']:.1f} nmi")
    if b[key] != bw[key]:
        print("  NOTE: the two disagree -- D0 shifts the optimum toward the lighter wing.")
    Path(path).with_name(Path(path).stem + "_range.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         str(Path(__file__).resolve().parent.parent / "out" / "logs" / "coupled_toc.json"))
