"""Configuration for the VSP -> OpenAeroStruct planform study.

Everything that is a choice rather than a consequence lives here.
"""

import os

from studies.vsp_planform.atmosphere import flight_condition

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

BASELINES = {
    "plan_l": os.path.join(DATA_DIR, "Plan_L_DegenGeom.csv"),
    "const_chord": os.path.join(DATA_DIR, "Plan_L_ConstChord_DegenGeom.csv"),
}

# The OpenVSP models are in inches. OAS works in SI.
SCALE = 0.0254

# ---------------------------------------------------------------------------
# Meshing
# ---------------------------------------------------------------------------

# Spanwise stations per half after resampling. The two baselines arrive at very
# different native resolutions (27 vs 95 sections), so they get re-lofted onto a
# common distribution to be comparable and to keep the VLM affordable.
N_SPANWISE_HALF = 35

# Fraction of spanwise stations placed in the winglet (region C). The winglet is
# a small span fraction but has strong curvature, so it needs a share out of
# proportion to its span.
WINGLET_STATION_FRACTION = 0.2

# ---------------------------------------------------------------------------
# Flight condition
# ---------------------------------------------------------------------------
# 260 KTAS at 25,000 ft, ISA. Derived rather than hardcoded so changing either
# number propagates correctly; see atmosphere.py.

KTAS = 260.0
ALTITUDE_FT = 25000.0

_FC = flight_condition(KTAS, ALTITUDE_FT)

V_MS = _FC["v"]  # 133.755 m/s
RHO = _FC["rho"]  # 0.548946 kg/m^3
MACH = _FC["Mach_number"]  # 0.431930
RE_PER_M = _FC["re"]  # 4.76863e6 per metre
SPEED_OF_SOUND = _FC["speed_of_sound"]  # 309.669 m/s
ALTITUDE_M = _FC["altitude_m"]

# Target lift coefficient for the drag optimization.
CL_TARGET = 0.5

# ---------------------------------------------------------------------------
# Aerodynamic model
# ---------------------------------------------------------------------------

K_LAM = 0.05  # fraction of chord with laminar flow
WITH_VISCOUS = True
WITH_WAVE = True

# ---------------------------------------------------------------------------
# Design variables
# ---------------------------------------------------------------------------

N_TWIST_CP = 5

# Absolute twist limits along the span. Both baselines already sit inside this
# window (Plan_L 2.12 -> 0.13 deg, ConstChord 4.00 -> -1.00 deg).
TWIST_BOUNDS = (-1.0, 5.0)  # deg

SWEEP_B_BOUNDS = (0.0, 35.0)  # deg, leading-edge sweep of the tapered region
TAPER_B_BOUNDS = (0.15, 1.0)  # tip/root chord ratio of the tapered region

# Rear-spar location as a fraction of chord. The aft wingbox must stay straight
# (unswept). Both baselines already satisfy this: the unswept chord line sits at
# 0.60 for Plan_L and ~0.68 for ConstChord.
WINGBOX_CHORD_PCT_BOUNDS = (0.45, 0.75)

# ---------------------------------------------------------------------------
# Region detection overrides
# ---------------------------------------------------------------------------
# Set to (idx_a_end, idx_c_start) to bypass automatic detection for a baseline.

REGION_OVERRIDES = {}
