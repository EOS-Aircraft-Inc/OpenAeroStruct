"""1976 US Standard Atmosphere, troposphere and lower stratosphere.

Enough to turn "260 KTAS at 25 kft" into the density, Mach number and unit
Reynolds number that OpenAeroStruct wants.
"""

import numpy as np

# ISA sea-level reference values
T0 = 288.15  # K
P0 = 101325.0  # Pa
LAPSE = 0.0065  # K/m, troposphere
H_TROPOPAUSE = 11000.0  # m
T_TROPOPAUSE = T0 - LAPSE * H_TROPOPAUSE  # 216.65 K

G0 = 9.80665  # m/s^2
R_AIR = 287.0528  # J/(kg K)
GAMMA = 1.4

# Sutherland's law constants for air
MU_REF = 1.716e-5  # Pa s
T_REF = 273.15  # K
S_SUTHERLAND = 110.4  # K

FT_TO_M = 0.3048
KNOT_TO_MS = 0.514444


def atmosphere(altitude_m):
    """Return ISA properties at a geopotential altitude in metres.

    Returns a dict with temperature (K), pressure (Pa), density (kg/m^3),
    speed of sound (m/s) and dynamic viscosity (Pa s).
    """
    h = np.asarray(altitude_m, dtype=float)

    if np.any(h > 20000.0):
        raise ValueError("only valid to 20 km")

    # Troposphere: linear temperature lapse.
    T_trop = T0 - LAPSE * h
    p_trop = P0 * (T_trop / T0) ** (G0 / (R_AIR * LAPSE))

    # Lower stratosphere: isothermal.
    p_tropopause = P0 * (T_TROPOPAUSE / T0) ** (G0 / (R_AIR * LAPSE))
    T_strat = np.full_like(T_trop, T_TROPOPAUSE)
    p_strat = p_tropopause * np.exp(-G0 * (h - H_TROPOPAUSE) / (R_AIR * T_TROPOPAUSE))

    below = h <= H_TROPOPAUSE
    T = np.where(below, T_trop, T_strat)
    p = np.where(below, p_trop, p_strat)

    rho = p / (R_AIR * T)
    a = np.sqrt(GAMMA * R_AIR * T)
    mu = MU_REF * (T / T_REF) ** 1.5 * (T_REF + S_SUTHERLAND) / (T + S_SUTHERLAND)

    return {
        "temperature": float(T),
        "pressure": float(p),
        "density": float(rho),
        "speed_of_sound": float(a),
        "viscosity": float(mu),
    }


def flight_condition(ktas, altitude_ft):
    """Turn true airspeed in knots and altitude in feet into an OAS-ready dict.

    Returns v (m/s), rho, Mach_number, re (Reynolds per metre), speed_of_sound,
    plus the atmospheric state it came from.
    """
    altitude_m = altitude_ft * FT_TO_M
    atm = atmosphere(altitude_m)

    v = ktas * KNOT_TO_MS
    mach = v / atm["speed_of_sound"]
    re_per_m = atm["density"] * v / atm["viscosity"]

    return {
        "v": v,
        "rho": atm["density"],
        "Mach_number": mach,
        "re": re_per_m,
        "speed_of_sound": atm["speed_of_sound"],
        "altitude_m": altitude_m,
        "altitude_ft": altitude_ft,
        "ktas": ktas,
        **{k: atm[k] for k in ("temperature", "pressure", "viscosity")},
    }


if __name__ == "__main__":
    fc = flight_condition(260.0, 25000.0)
    for key in ("ktas", "altitude_ft", "temperature", "pressure", "rho", "speed_of_sound", "viscosity"):
        print(f"{key:>16s} = {fc[key]:.6g}")
    print(f"{'v (m/s)':>16s} = {fc['v']:.6g}")
    print(f"{'Mach':>16s} = {fc['Mach_number']:.6g}")
    print(f"{'Re per m':>16s} = {fc['re']:.6g}")
