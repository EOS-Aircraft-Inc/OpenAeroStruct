"""Standalone NACA 4-digit airfoil design-of-experiments using AeroSandbox/NeuralFoil.

This module is deliberately self-contained: it does *not* import OpenAeroStruct.
It sweeps the NACA 4-digit family through NeuralFoil at this study's flight
condition (260 KTAS at 25,000 ft ISA) and writes a tidy results table plus a
per-(airfoil, Re) summary of the quantities that would later feed a wing model.

Run it directly::

    python studies/vsp_planform/airfoil_doe.py

Requires ``aerosandbox`` (``pip install aerosandbox``). AeroSandbox bundles
NeuralFoil, so no XFoil binary is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

AEROSANDBOX_INSTALL_HINT = (
    "This study requires AeroSandbox (which bundles NeuralFoil).\n"
    "Install it with:\n\n"
    "    pip install aerosandbox\n\n"
    "It is pure Python; no XFoil binary is required."
)

try:
    import aerosandbox as asb

    HAS_AEROSANDBOX = True
except ImportError:  # pragma: no cover - exercised only without the dependency
    asb = None
    HAS_AEROSANDBOX = False


# ----------------------------------------------------------------------------
# Flight condition: ISA, 260 KTAS at 25,000 ft. See atmosphere.py for the
# derivation; the numbers are hard-coded here to keep this module standalone.
# ----------------------------------------------------------------------------
V_INF = 133.755  # m/s true airspeed
RHO_INF = 0.548946  # kg/m^3
MACH_INF = 0.431930  # -
SPEED_OF_SOUND = 309.669  # m/s
UNIT_REYNOLDS = 4.76863e6  # per metre

# The real wings' chords run roughly 0.36 m (tip) to 3.53 m (root), so chord
# Reynolds numbers span about 1.7e6 to 1.7e7.
DEFAULT_REYNOLDS = (1.7e6, 3.0e6, 5.0e6, 1.0e7, 1.7e7)

# The real wings' t/c runs 0.100 to 0.178, so the sweep brackets that fully.
DEFAULT_THICKNESSES = (10, 12, 14, 15, 16, 17, 18)
DEFAULT_CAMBERS = (0, 1, 2, 3, 4)
DEFAULT_CAMBER_POSITIONS = (2, 4, 6)

# Wide enough to capture the drag bucket on the low side and stall onset on the
# high side, fine enough that CL_max and (L/D)_max are resolved.
DEFAULT_ALPHAS = np.arange(-8.0, 20.01, 0.5)

NEURALFOIL_MODEL_SIZE = "large"

DATA_DIR = Path(__file__).resolve().parent / "data"
DOE_CSV = DATA_DIR / "airfoil_doe.csv"
SUMMARY_CSV = DATA_DIR / "airfoil_doe_summary.csv"
POLAR_PNG = DATA_DIR / "airfoil_doe_polars.png"
LD_PNG = DATA_DIR / "airfoil_doe_ld_vs_thickness.png"
CD_PNG = DATA_DIR / "airfoil_doe_cd_at_fixed_cl.png"

#: Representative section lift coefficient for the cruise point.
CL_CRUISE = 0.5


def _require_aerosandbox():
    """Raise a clear, actionable error if AeroSandbox is missing."""
    if not HAS_AEROSANDBOX:
        raise RuntimeError(AEROSANDBOX_INSTALL_HINT)


# ----------------------------------------------------------------------------
# Geometry parameterization
# ----------------------------------------------------------------------------
def naca4_name(camber: int, camber_pos: int, thickness: int) -> str:
    """Return the NACA 4-digit designation, e.g. (2, 4, 12) -> ``"naca2412"``.

    ``camber`` is max camber in percent chord (first digit), ``camber_pos`` is
    the chordwise location of max camber in tenths of chord (second digit), and
    ``thickness`` is max thickness in percent chord (last two digits).
    """
    camber = int(camber)
    camber_pos = int(camber_pos)
    thickness = int(thickness)

    if not 0 <= camber <= 9:
        raise ValueError(f"camber must be a single digit 0-9, got {camber}")
    if not 0 <= camber_pos <= 9:
        raise ValueError(f"camber_pos must be a single digit 0-9, got {camber_pos}")
    if not 1 <= thickness <= 99:
        raise ValueError(f"thickness must be 1-99 percent, got {thickness}")

    # An uncambered section has no meaningful camber location; NACA writes 00xx.
    if camber == 0:
        camber_pos = 0

    return f"naca{camber}{camber_pos}{thickness:02d}"


def parse_naca4(name: str) -> tuple[int, int, int]:
    """Inverse of :func:`naca4_name`: ``"naca2412"`` -> ``(2, 4, 12)``."""
    digits = name.lower().removeprefix("naca")
    if len(digits) != 4 or not digits.isdigit():
        raise ValueError(f"not a NACA 4-digit designation: {name!r}")
    return int(digits[0]), int(digits[1]), int(digits[2:])


def naca4_grid(cambers, camber_positions, thicknesses) -> list[str]:
    """Full-factorial NACA 4-digit grid, de-duplicated and sorted.

    Symmetric sections (camber 0) collapse to a single ``naca00xx`` regardless
    of how many camber positions are requested, so the returned list is shorter
    than the naive product whenever ``0`` is in ``cambers``.
    """
    names = []
    seen = set()
    for thickness in thicknesses:
        for camber in cambers:
            for camber_pos in camber_positions:
                name = naca4_name(camber, camber_pos, thickness)
                if name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


# ----------------------------------------------------------------------------
# The sweep
# ----------------------------------------------------------------------------
#: Scalar (per-alpha) NeuralFoil outputs worth keeping in the tidy table. The
#: model also returns ~200 boundary-layer station values, which we drop.
AERO_KEYS = ("CL", "CD", "CM", "Cpmin", "Top_Xtr", "Bot_Xtr", "mach_crit", "analysis_confidence")

DOE_COLUMNS = ("airfoil", "camber", "camber_pos", "thickness", "t_over_c", "Re", "mach", "alpha", *AERO_KEYS)


def run_doe(airfoils, alphas, reynolds, mach=MACH_INF, model_size=NEURALFOIL_MODEL_SIZE) -> pd.DataFrame:
    """Run the DOE and return a tidy DataFrame, one row per (airfoil, alpha, Re).

    NeuralFoil is vectorized over ``alpha``, so this makes one call per
    (airfoil, Re) pair rather than one per row.
    """
    _require_aerosandbox()

    alphas = np.atleast_1d(np.asarray(alphas, dtype=float))
    reynolds = np.atleast_1d(np.asarray(reynolds, dtype=float))
    n_alpha = alphas.size

    frames = []
    for name in airfoils:
        camber, camber_pos, thickness = parse_naca4(name)
        foil = asb.Airfoil(name)  # coordinate generation is the expensive bit; do it once
        for re in reynolds:
            aero = foil.get_aero_from_neuralfoil(alpha=alphas, Re=re, mach=mach, model_size=model_size)
            block = {
                "airfoil": name,
                "camber": camber,
                "camber_pos": camber_pos,
                "thickness": thickness,
                "t_over_c": thickness / 100.0,
                "Re": re,
                "mach": mach,
                "alpha": alphas,
            }
            for key in AERO_KEYS:
                block[key] = np.broadcast_to(np.asarray(aero[key], dtype=float), (n_alpha,))
            frames.append(pd.DataFrame(block))

    df = pd.concat(frames, ignore_index=True)
    return df[list(DOE_COLUMNS)]


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
SUMMARY_COLUMNS = (
    "airfoil",
    "camber",
    "camber_pos",
    "thickness",
    "t_over_c",
    "Re",
    "CL_max",
    "alpha_CL_max",
    "CD_min",
    "LD_max",
    "CL_at_LD_max",
    "alpha_at_LD_max",
    "CM_at_zero_lift",
)


def _stall_index(cl: np.ndarray) -> int:
    """Index of the first local maximum of CL, i.e. stall onset.

    Taking a global ``argmax`` is unsafe here: NeuralFoil's deep-stall
    extrapolation often turns CL back upward past ~18 deg, which is not a
    physical second lift peak.
    """
    falling = np.flatnonzero(np.diff(cl) < 0.0)
    return int(falling[0]) if falling.size else int(cl.size - 1)


def _cm_at_zero_lift(cl: np.ndarray, cm: np.ndarray) -> float:
    """Interpolate CM at CL = 0 using the linear (pre-stall) part of the curve.

    Restricting to the monotonic run of CL that brackets zero keeps the
    interpolation well-posed even though the full sweep goes past stall.
    """
    order = np.argsort(cl)
    cl_sorted, cm_sorted = cl[order], cm[order]
    if cl_sorted[0] > 0.0 or cl_sorted[-1] < 0.0:
        return float("nan")
    return float(np.interp(0.0, cl_sorted, cm_sorted))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the tidy DOE table to one row per (airfoil, Re).

    Reports CL_max, CD_min, the maximum lift-to-drag ratio with the CL and alpha
    at which it occurs, and CM at zero lift.
    """
    records = []
    for (name, re), group in df.groupby(["airfoil", "Re"], sort=True):
        group = group.sort_values("alpha")
        cl = group["CL"].to_numpy()
        cd = group["CD"].to_numpy()
        cm = group["CM"].to_numpy()
        alpha = group["alpha"].to_numpy()

        # Everything below is defined on the pre-stall branch; NeuralFoil's
        # deep-stall extrapolation is not trustworthy enough to optimize over.
        i_stall = _stall_index(cl)
        cl, cd, cm, alpha = cl[: i_stall + 1], cd[: i_stall + 1], cm[: i_stall + 1], alpha[: i_stall + 1]

        ld = np.divide(cl, cd, out=np.zeros_like(cl), where=cd > 0.0)
        i_ld = int(np.argmax(ld))
        i_clmax = int(np.argmax(cl))

        first = group.iloc[0]
        records.append(
            {
                "airfoil": name,
                "camber": int(first["camber"]),
                "camber_pos": int(first["camber_pos"]),
                "thickness": int(first["thickness"]),
                "t_over_c": float(first["t_over_c"]),
                "Re": float(re),
                "CL_max": float(cl[i_clmax]),
                "alpha_CL_max": float(alpha[i_clmax]),
                "CD_min": float(np.min(cd)),
                "LD_max": float(ld[i_ld]),
                "CL_at_LD_max": float(cl[i_ld]),
                "alpha_at_LD_max": float(alpha[i_ld]),
                "CM_at_zero_lift": _cm_at_zero_lift(cl, cm),
            }
        )

    return pd.DataFrame.from_records(records, columns=list(SUMMARY_COLUMNS))


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_polars(df: pd.DataFrame, path: Path, camber: int = 2, camber_pos: int = 4, re: float = 5.0e6) -> None:
    """Drag polars grouped by thickness, at one camber and one Reynolds number."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = df[(df["camber"] == camber) & (df["camber_pos"] == camber_pos) & np.isclose(df["Re"], re)]
    if subset.empty:
        return

    # Two panels: the full polar (post-stall included) and a zoom on the drag
    # bucket, which is where the thickness trade actually lives.
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    thicknesses = sorted(subset["thickness"].unique())
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, len(thicknesses)))
    for color, thickness in zip(colors, thicknesses):
        rows = subset[subset["thickness"] == thickness].sort_values("alpha")
        label = f"t/c = {thickness / 100:.2f}"
        ax_full.plot(rows["CD"], rows["CL"], color=color, label=label)
        ax_zoom.plot(rows["CD"], rows["CL"], color=color, label=label)

    cd_bucket = float(subset["CD"].min())
    for ax in (ax_full, ax_zoom):
        ax.set_xlabel("$C_D$")
        ax.set_ylabel("$C_L$")
        ax.set_xlim(left=0.0)
        ax.grid(alpha=0.3)
    ax_full.set_title("full polar")
    ax_zoom.set_xlim(0.0, 4.0 * cd_bucket)
    ax_zoom.set_ylim(-0.8, 1.2)
    ax_zoom.set_title("drag bucket (zoom)")
    ax_zoom.legend(fontsize="small")
    fig.suptitle(f"NACA {camber}{camber_pos}xx drag polars, Re = {re:.2g}, M = {MACH_INF:.3f}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ld_vs_thickness(summary: pd.DataFrame, path: Path, camber: int = 2, camber_pos: int = 4) -> None:
    """(L/D)_max versus thickness, one line per Reynolds number."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = summary[(summary["camber"] == camber) & (summary["camber_pos"] == camber_pos)]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    reynolds = sorted(subset["Re"].unique())
    colors = plt.get_cmap("plasma")(np.linspace(0.0, 0.85, len(reynolds)))
    for color, re in zip(colors, reynolds):
        rows = subset[np.isclose(subset["Re"], re)].sort_values("thickness")
        ax.plot(rows["t_over_c"], rows["LD_max"], "o-", color=color, label=f"Re = {re:.2g}")

    ax.axvspan(0.100, 0.178, color="0.85", zorder=0, label="study wing t/c range")
    ax.set_xlabel("$t/c$")
    ax.set_ylabel("$(L/D)_{max}$")
    ax.set_title(f"NACA {camber}{camber_pos}xx maximum L/D vs thickness, M = {MACH_INF:.3f}")
    ax.grid(alpha=0.3)
    # Make room below the lowest curve so the legend does not sit on the data.
    low, high = subset["LD_max"].min(), subset["LD_max"].max()
    ax.set_ylim(low - 0.35 * (high - low), high + 0.05 * (high - low))
    ax.legend(loc="lower left", fontsize="small", ncol=2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cd_at_fixed_cl(df: pd.DataFrame, path: Path, cl_cruise: float = 0.5, camber: int = 2, camber_pos: int = 4):
    """Section CD at a fixed cruise CL versus thickness, one line per Reynolds number.

    This is the honest view of the thickness penalty: ``(L/D)_max`` hides it,
    because a thicker section simply reaches its optimum at a higher CL.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = df[(df["camber"] == camber) & (df["camber_pos"] == camber_pos)]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    reynolds = sorted(subset["Re"].unique())
    colors = plt.get_cmap("plasma")(np.linspace(0.0, 0.85, len(reynolds)))
    for color, re in zip(colors, reynolds):
        at_re = subset[np.isclose(subset["Re"], re)]
        tocs, cds = [], []
        for thickness in sorted(at_re["thickness"].unique()):
            rows = at_re[at_re["thickness"] == thickness].sort_values("alpha")
            cl = rows["CL"].to_numpy()
            cd = rows["CD"].to_numpy()
            stall = _stall_index(cl)
            tocs.append(thickness / 100.0)
            cds.append(float(np.interp(cl_cruise, cl[: stall + 1], cd[: stall + 1])))
        ax.plot(tocs, cds, "o-", color=color, label=f"Re = {re:.2g}")

    ax.axvspan(0.100, 0.178, color="0.85", zorder=0, label="study wing t/c range")
    ax.set_xlabel("$t/c$")
    ax.set_ylabel(f"$C_D$ at $C_L$ = {cl_cruise:.2f}")
    ax.set_title(f"NACA {camber}{camber_pos}xx section drag at fixed lift, M = {MACH_INF:.3f}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize="small")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main() -> int:
    if not HAS_AEROSANDBOX:
        print(AEROSANDBOX_INSTALL_HINT, file=sys.stderr)
        return 1

    import time

    airfoils = naca4_grid(DEFAULT_CAMBERS, DEFAULT_CAMBER_POSITIONS, DEFAULT_THICKNESSES)
    alphas = DEFAULT_ALPHAS
    reynolds = DEFAULT_REYNOLDS

    print(f"Flight condition: V = {V_INF} m/s, rho = {RHO_INF} kg/m^3, M = {MACH_INF:.6f}")
    print(f"  a = {SPEED_OF_SOUND} m/s, unit Re = {UNIT_REYNOLDS:.5e} /m")
    print(f"Grid: {len(airfoils)} airfoils x {len(alphas)} alphas x {len(reynolds)} Reynolds numbers")
    print(f"  cambers          {list(DEFAULT_CAMBERS)} %c")
    print(f"  camber positions {list(DEFAULT_CAMBER_POSITIONS)} tenths chord")
    print(f"  thicknesses      {list(DEFAULT_THICKNESSES)} %c")
    print(f"  alpha            {alphas[0]:.1f} to {alphas[-1]:.1f} deg, step {alphas[1] - alphas[0]:.1f}")
    print(f"  Re               {[f'{r:.2g}' for r in reynolds]}")
    print(
        f"  -> {len(airfoils) * len(alphas) * len(reynolds)} rows from "
        f"{len(airfoils) * len(reynolds)} vectorized NeuralFoil calls"
    )

    t0 = time.perf_counter()
    df = run_doe(airfoils, alphas, reynolds, mach=MACH_INF)
    t_doe = time.perf_counter() - t0
    print(f"DOE finished in {t_doe:.1f} s ({len(df)} rows)")

    summary = summarize(df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DOE_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {DOE_CSV}")
    print(f"Wrote {SUMMARY_CSV}")

    plot_polars(df, POLAR_PNG)
    plot_ld_vs_thickness(summary, LD_PNG)
    plot_cd_at_fixed_cl(df, CD_PNG, cl_cruise=CL_CRUISE)
    for path in (POLAR_PNG, LD_PNG, CD_PNG):
        print(f"Wrote {path}")

    # Headline findings.
    for re in reynolds:
        at_re = summary[np.isclose(summary["Re"], re)].sort_values("LD_max", ascending=False)
        best = at_re.iloc[0]
        print(
            f"Re = {re:.2g}: best L/D {best['LD_max']:.1f} from {best['airfoil']} "
            f"at CL = {best['CL_at_LD_max']:.3f}; CL_max = {best['CL_max']:.2f}"
        )

    # Penalty for the real wing's thickest section relative to a thinner one,
    # holding camber fixed. Reported two ways: at each section's own (L/D)_max,
    # and at a common cruise CL, which is the more meaningful comparison.
    ref_summary = summary[(summary["camber"] == 2) & (summary["camber_pos"] == 4)]
    ref_df = df[(df["camber"] == 2) & (df["camber_pos"] == 4)]

    def cd_at_cruise(thickness, re):
        rows = ref_df[(ref_df["thickness"] == thickness) & np.isclose(ref_df["Re"], re)].sort_values("alpha")
        cl, cd = rows["CL"].to_numpy(), rows["CD"].to_numpy()
        stall = _stall_index(cl)
        return float(np.interp(CL_CRUISE, cl[: stall + 1], cd[: stall + 1]))

    for re in reynolds:
        at_re = ref_summary[np.isclose(ref_summary["Re"], re)].set_index("thickness")
        if 18 not in at_re.index or 12 not in at_re.index:
            continue
        thick, thin = at_re.loc[18, "LD_max"], at_re.loc[12, "LD_max"]
        cd_thick, cd_thin = cd_at_cruise(18, re), cd_at_cruise(12, re)
        print(
            f"Re = {re:.2g}: NACA 2418 vs 2412 -> (L/D)_max {thick:.1f} vs {thin:.1f} "
            f"({100 * (thick / thin - 1):+.1f}%), CD at CL={CL_CRUISE:.1f} "
            f"{cd_thick:.5f} vs {cd_thin:.5f} ({100 * (cd_thick / cd_thin - 1):+.1f}%)"
        )

    print(f"Total wall time {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
