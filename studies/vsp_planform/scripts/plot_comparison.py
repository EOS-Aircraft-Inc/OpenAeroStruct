"""Baseline vs optimized, both baselines, everything on one sheet.

Design vectors come from the converged SLSQP run (opt.log), so nothing is
re-optimized here -- the model is just replayed at the two design points.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/alex/repos/OpenAeroStruct")

from studies.vsp_planform import config
from studies.vsp_planform.regions import detect_regions
from studies.vsp_planform.run_opt import POINT, build_problem, load_baseline, trim_alpha

HERE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "figures"
)

BASE_COLOR = "#4C72B0"
OPT_COLOR = "#C44E52"

CONVERGED = {
    "plan_l": {
        "alpha": 0.37774647,
        "taper_B": 0.15,
        "twist_cp": np.array([-2.1185275, -1.25198153, -0.76894147, 3.72635779, 4.21288852]),
        "wingbox_pct": 0.50715943,
    },
    "const_chord": {
        "alpha": -1.55366469,
        "taper_B": 0.37945194,
        "twist_cp": np.array([-2.96242673, -0.94136205, -1.32748637, 2.78465501, 0.05821838]),
        "wingbox_pct": 0.75,
    },
}

ALPHAS = np.linspace(-3.0, 6.0, 19)


def spanwise_load(prob):
    """Sectional lift per unit span, and the panel mid-span stations."""
    forces = prob.get_val(f"{POINT}.aero_states.wing_sec_forces")
    widths = prob.get_val(f"{POINT}.wing.widths")
    lift = forces[:, :, 2].sum(axis=0) / widths
    mesh = prob.get_val("wing.mesh", units="m")
    y_nodes = mesh[0, :, 1]
    y_mid = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    return y_mid, lift


def apply(prob, design):
    prob.set_val("alpha", design["alpha"], units="deg")
    prob.set_val("wing.taper_B", design["taper_B"])
    prob.set_val("wing.twist_cp", design["twist_cp"], units="deg")
    prob.set_val("wing.wingbox_pct", design["wingbox_pct"])


def collect(name):
    """Everything needed for one baseline, at both design points."""
    mesh, stick, regions, planform0, _, _ = load_baseline(name, config.N_SPANWISE_HALF, 9)
    prob, _ = build_problem(name, mesh, stick, regions, planform0)
    out = {"regions": regions}

    for label, design in (("baseline", None), ("optimized", CONVERGED[name])):
        if design is not None:
            apply(prob, design)
            prob.run_model()
        else:
            # Trim the baseline to the same CL the optimizer is held to, or the
            # comparison is against the model's default alpha, not the design point.
            trim_alpha(prob, config.CL_TARGET)

        # Polar sweep at the fixed geometry.
        alpha0 = float(prob.get_val("alpha", units="deg")[0])
        polar = []
        for a in ALPHAS:
            prob.set_val("alpha", a, units="deg")
            prob.run_model()
            polar.append(
                (
                    a,
                    float(prob.get_val(f"{POINT}.wing_perf.CL")[0]),
                    float(prob.get_val(f"{POINT}.wing_perf.CD")[0]),
                )
            )
        prob.set_val("alpha", alpha0, units="deg")
        prob.run_model()

        y_mid, lift = spanwise_load(prob)
        geom_mesh = prob.get_val("wing.mesh", units="m")
        out[label] = {
            "polar": np.array(polar),
            "twist": prob.get_val("twist_abs", units="deg").copy(),
            "y": geom_mesh[0, :, 1].copy(),
            "chord": (geom_mesh[-1, :, 0] - geom_mesh[0, :, 0]).copy(),
            "le_x": geom_mesh[0, :, 0].copy(),
            "te_x": geom_mesh[-1, :, 0].copy(),
            "y_mid": y_mid,
            "lift": lift,
            "CL": float(prob.get_val(f"{POINT}.wing_perf.CL")[0]),
            "CD": float(prob.get_val(f"{POINT}.wing_perf.CD")[0]),
            "alpha": alpha0,
        }
    return out


def elliptical(y, total):
    """Elliptical loading with the same total lift, for reference."""
    semi = y.max()
    shape = np.sqrt(np.clip(1.0 - (y / semi) ** 2, 0.0, None))
    return shape * total / np.trapezoid(shape, y)


def main():
    data = {name: collect(name) for name in config.BASELINES}

    fig = plt.figure(figsize=(15, 19))
    fig.suptitle(
        "Baseline vs optimized — min CD at CL = 0.5, fixed reference area\n"
        "260 KTAS at 25,000 ft (M = 0.432), VLM 9 x 35 per half",
        fontsize=15,
    )
    grid = fig.add_gridspec(5, 2, hspace=0.34, wspace=0.22, top=0.935, bottom=0.055, left=0.075, right=0.97)

    for col, name in enumerate(config.BASELINES):
        d = data[name]
        regions = d["regions"]
        semi = d["baseline"]["y"][-1]

        # --------------------------------------------------------- drag polar
        ax = fig.add_subplot(grid[0, col])
        for label, color in (("baseline", BASE_COLOR), ("optimized", OPT_COLOR)):
            polar = d[label]["polar"]
            ax.plot(polar[:, 2] * 1e4, polar[:, 1], "-", color=color, lw=1.7, label=label)
            ax.plot(d[label]["CD"] * 1e4, d[label]["CL"], "o", color=color, ms=7, mfc="white", mew=1.8)
        ax.axhline(config.CL_TARGET, color="0.75", ls="--", lw=1)
        ax.annotate(
            f"design CL = {config.CL_TARGET}",
            xy=(0.02, config.CL_TARGET),
            xycoords=("axes fraction", "data"),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=8,
            color="0.45",
        )
        ax.set_title(f"{name} — drag polar", fontsize=11)
        ax.set_xlabel(r"$C_D$ [counts]")
        ax.set_ylabel(r"$C_L$")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="lower right")

        # ------------------------------------------------------------ L over D
        ax = fig.add_subplot(grid[1, col])
        for label, color in (("baseline", BASE_COLOR), ("optimized", OPT_COLOR)):
            polar = d[label]["polar"]
            ax.plot(polar[:, 1], polar[:, 1] / polar[:, 2], "-", color=color, lw=1.7, label=label)
            ax.plot(d[label]["CL"], d[label]["CL"] / d[label]["CD"], "o", color=color, ms=7, mfc="white", mew=1.8)
        ax.axvline(config.CL_TARGET, color="0.75", ls="--", lw=1)
        ax.set_title(f"{name} — L/D", fontsize=11)
        ax.set_xlabel(r"$C_L$")
        ax.set_ylabel("L / D")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="lower right")

        # ------------------------------------------------------------- twist
        ax = fig.add_subplot(grid[2, col])
        for label, color, marker in (("baseline", BASE_COLOR, "o"), ("optimized", OPT_COLOR, "s")):
            ax.plot(
                d[label]["y"] / semi, d[label]["twist"], marker + "-", ms=3.2, lw=1.6, color=color, label=label
            )
        for frac, tag in (
            (regions.y_a_end * config.SCALE / semi, "A|B"),
            (regions.y_c_start * config.SCALE / semi, "B|C"),
        ):
            ax.axvline(frac, color="0.75", ls="--", lw=1)
            ax.annotate(
                tag,
                xy=(frac, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(3, -12),
                textcoords="offset points",
                fontsize=8,
                color="0.45",
            )
        ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
        ax.set_title(f"{name} — spanwise twist", fontsize=11)
        ax.set_xlabel("y / semi-span")
        ax.set_ylabel("twist [deg]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

        # ----------------------------------------------------- spanwise load
        ax = fig.add_subplot(grid[3, col])
        for label, color in (("baseline", BASE_COLOR), ("optimized", OPT_COLOR)):
            y_mid, lift = d[label]["y_mid"], d[label]["lift"]
            keep = y_mid >= 0
            ax.plot(y_mid[keep] / semi, lift[keep], "-", color=color, lw=1.7, label=label)
        y_ref = d["baseline"]["y_mid"]
        keep = y_ref >= 0
        total = np.trapezoid(d["baseline"]["lift"][keep], y_ref[keep])
        ax.plot(
            y_ref[keep] / semi,
            elliptical(y_ref[keep], total),
            ls=(0, (4, 3)),
            color="0.45",
            lw=1.2,
            label="elliptical (same lift)",
        )
        ax.set_title(f"{name} — spanwise loading", fontsize=11)
        ax.set_xlabel("y / semi-span")
        ax.set_ylabel("lift per unit span [N/m]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

        # --------------------------------------------------------- planform
        ax = fig.add_subplot(grid[4, col])
        for label, color in (("baseline", BASE_COLOR), ("optimized", OPT_COLOR)):
            y = d[label]["y"]
            keep = y >= 0
            ax.plot(y[keep], d[label]["le_x"][keep], "-", color=color, lw=1.6, label=label)
            ax.plot(y[keep], d[label]["te_x"][keep], "-", color=color, lw=1.6)
        ax.set_title(f"{name} — planform (half)", fontsize=11)
        ax.set_xlabel("y [m]")
        ax.set_ylabel("x [m]")
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    lines = []
    for name in config.BASELINES:
        b, o = data[name]["baseline"], data[name]["optimized"]
        lines.append(
            f"{name:12s}  CD {b['CD'] * 1e4:7.1f} -> {o['CD'] * 1e4:7.1f} counts "
            f"({(o['CD'] / b['CD'] - 1) * 100:+.2f}%)   "
            f"L/D {b['CL'] / b['CD']:6.2f} -> {o['CL'] / o['CD']:6.2f}   "
            f"alpha {b['alpha']:+.3f} -> {o['alpha']:+.3f} deg"
        )
    fig.text(0.5, 0.021, "\n".join(lines), ha="center", fontsize=10, family="monospace")
    fig.text(
        0.5,
        0.004,
        "Polars are swept at frozen geometry; markers are the design points. "
        "Plan_L's taper and ConstChord's spar both finished on a bound.",
        ha="center",
        fontsize=9,
        color="0.4",
    )

    path = os.path.join(HERE, "baseline_vs_optimized.png")
    fig.savefig(path, dpi=120)
    print(path)


if __name__ == "__main__":
    main()
