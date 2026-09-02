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
TAPER_B_BOUNDS = (0.05, 1.0)  # tip/root chord ratio of the tapered region

# Ceiling on cruise CL. In fixed_lift mode the wing area is free, so this is
# what stops the optimizer shrinking the wing indefinitely: it sets a floor on
# area of W / (q * MAX_CRUISE_CL). Do not try to free the area under fixed_cl
# instead -- with CL pinned and area free, shrinking the wing shrinks the lift,
# and the optimizer buys a lower CD by carrying less rather than by flying
# better. That was measured, not assumed: it ran straight to the area floor.
MAX_CRUISE_CL = 1.05

# Location of the *straight* chord line, as a fraction of chord. This is the
# line that has to stay unswept, and it is what sets region B's sweep. It is NOT
# an edge of the structural box: measured off the Wingbox geom, it sits at 68.1%
# (straight to 0.09 in) while the box's rear edge is a separate line at 74.2%
# carrying 0.33 deg of sweep. The straight line lies inside the box.
WINGBOX_CHORD_PCT_BOUNDS = (0.45, 0.75)

# The structural box, as fractions of chord. Fixed, and independent of the
# straight line above -- conflating the two is what previously made the width
# constraint look infeasible. Measured as-built box: 12.5% to 74.2% of chord.
WINGBOX_FRONT_PCT = 0.125
WINGBOX_REAR_PCT = 0.75

# The rear spar is allowed to *kink*: it is a piecewise-linear schedule of
# (y in inches, chord fraction) breakpoints rather than one number. Outside the
# breakpoints the end values are held. A single breakpoint is the constant-
# fraction spar the study ran until the wing 2 design point, and reproduces the
# old behaviour exactly.
#
# Note this is the box's rear *edge*, not the straight sweep-driving line
# (WINGBOX_CHORD_PCT_BOUNDS above). The two were decoupled deliberately; a
# kinking rear edge does not make the sweep-driving line kink.
WINGBOX_REAR_SCHEDULE = ((0.0, WINGBOX_REAR_PCT),)

# Where the box width is required, as (y in inches, minimum width in inches).
# The chord falls monotonically outboard, so on a constant-fraction box only the
# outboard end of a run can bind; with a kinking rear spar every station named
# here has to be checked on its own.
WINGBOX_WIDTH_STATIONS = ((100.0, 65.0),)

# ---------------------------------------------------------------------------
# Region detection overrides
# ---------------------------------------------------------------------------
# Set to (idx_a_end, idx_c_start) to bypass automatic detection for a baseline.

REGION_OVERRIDES = {}
