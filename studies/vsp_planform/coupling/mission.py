"""Weight, battery and range bookkeeping for the coupled study.

Only two things move: the wing, and whatever it displaces. Everything else is a
single calibrated constant, because only its total matters and that total is
known -- summing a dozen guessed part weights would add error, not information.

    MTOW    = K + W_wing + payload + fuel        K = non-wing OEW, fixed
    m_batt  = MTOW - K_ex_batt - payload - fuel - W_wing

Wing weight trades against BATTERY at fixed MTOW, so electric range is

    R = eta * e_star * m_batt / D

and since eta and e_star are constants, m_batt/D ranks designs without needing
the battery specific energy at all.

Calibration: K_ex_batt against the aircraft mass export (Batteries 16,665.6 lb)
reproduces the book's battery mass to 0.24% at its 7,460 lb wing.
"""

LB_N = 4.4482216
LB_KG = 0.45359237
NMI_M = 1852.0

MTOW_LB = 86_000.0
K_LB = 56_000.0                      # non-wing OEW (nominal 63,500 OEW - 7,500 wing)
PAYLOAD_LB = 17_100.0
FUEL_LB = 5_400.0
BATT_LB_BOOK = 16_665.6              # aircraft mass export
K_EX_BATT_LB = K_LB - BATT_LB_BOOK

ETA_PROP = 0.80
E_STAR_WH_KG = 300.0


def cruise_weight_lb(fuel_lb=FUEL_LB, mtow_lb=MTOW_LB):
    """Mid-cruise weight. The wing is SIZED at MTOW but FLOWN at this."""
    return mtow_lb - 0.5 * fuel_lb


def battery_lb(w_wing_lb, mtow_lb=MTOW_LB, fuel_lb=FUEL_LB):
    return mtow_lb - K_EX_BATT_LB - PAYLOAD_LB - fuel_lb - w_wing_lb


def electric_range_nmi(w_wing_lb, drag_N, **kw):
    """Electric cruise range. Drag is whatever basis the caller used -- the OAS
    model is wing-only, so absolutes understate aircraft drag and overstate
    range. Comparisons on one basis are the point; absolutes are not."""
    e_j = battery_lb(w_wing_lb, **kw) * LB_KG * E_STAR_WH_KG * 3600.0
    return ETA_PROP * e_j / drag_N / NMI_M


def shaft_power_kW(drag_N, v_ms):
    return drag_N * v_ms / ETA_PROP / 1000.0
