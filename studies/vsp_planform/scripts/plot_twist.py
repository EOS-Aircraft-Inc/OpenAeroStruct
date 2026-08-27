"""Spanwise twist distribution, baseline against optimized.

Design vectors are read from the SLSQP log rather than re-optimizing, so this
plots exactly what the run converged to.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from studies.vsp_planform import config
from studies.vsp_planform.regions import detect_regions
from studies.vsp_planform.run_opt import build_problem, load_baseline

HERE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "figures"
)

# Converged design vectors, from opt.log.
CONVERGED = {
    "plan_l": {
        "alpha": 0.37774647,
        "taper_B": 0.15,
        "twist_cp": np.array([-2.1185275, -1.25198153, -0.76894147, 3.72635779, 4.21288852]),
        "wingbox_pct": 0.50715943,
        "status": "converged  |  CD -1.85%  |  taper_B at its lower bound",
    },
    "const_chord": {
        "alpha": -1.55366469,
        "taper_B": 0.37945194,
        "twist_cp": np.array([-2.96242673, -0.94136205, -1.32748637, 2.78465501, 0.05821838]),
        "wingbox_pct": 0.75,
        "status": "converged  |  CD -3.59%  |  spar at its upper bound",
    },
}


def states(name, design):
    """Baseline and optimized twist distributions, plus the spanwise stations."""
    mesh, stick, regions, planform0, _, _ = load_baseline(name, config.N_SPANWISE_HALF, 9)
    prob, _ = build_problem(name, mesh, stick, regions, planform0)

    prob.run_model()
    baseline = prob.get_val("twist_abs", units="deg").copy()
    y_baseline = prob.get_val("wing.mesh", units="m")[0, :, 1].copy()

    if design is None:
        return regions, y_baseline, baseline, None, None

    prob.set_val("alpha", design["alpha"], units="deg")
    prob.set_val("wing.taper_B", design["taper_B"])
    prob.set_val("wing.twist_cp", design["twist_cp"], units="deg")
    prob.set_val("wing.wingbox_pct", design["wingbox_pct"])
    prob.run_model()
    optimized = prob.get_val("twist_abs", units="deg").copy()
    y_opt = prob.get_val("wing.mesh", units="m")[0, :, 1].copy()
    return regions, y_baseline, baseline, y_opt, optimized


def main():
    names = list(config.BASELINES)
    fig, axes = plt.subplots(1, len(names), figsize=(14, 5.6), sharey=True)

    for ax, name in zip(np.atleast_1d(axes), names):
        design = CONVERGED.get(name)
        regions, y_b, twist_b, y_o, twist_o = states(name, design)

        semi = y_b[-1]
        ax.plot(y_b / semi, twist_b, "o-", ms=3.5, lw=1.6, color="#4C72B0", label="baseline (as built)")
        if twist_o is not None:
            ax.plot(y_o / semi, twist_o, "s-", ms=3.5, lw=1.6, color="#C44E52", label="optimized")

        # Region boundaries, in the same normalized coordinate.
        for y_break, label in (
            (regions.y_a_end * config.SCALE / semi, "A|B"),
            (regions.y_c_start * config.SCALE / semi, "B|C"),
        ):
            ax.axvline(y_break, color="0.7", ls="--", lw=1)
            ax.annotate(
                label,
                xy=(y_break, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(3, -12),
                textcoords="offset points",
                fontsize=8,
                color="0.45",
            )

        ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
        status = design["status"] if design else "optimization still running"
        ax.set_title(f"{name}\n{status}", fontsize=11)
        ax.set_xlabel("y / semi-span")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    np.atleast_1d(axes)[0].set_ylabel("geometric twist [deg], positive leading edge up")
    fig.suptitle("Spanwise twist distribution", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    path = os.path.join(HERE, "twist_distribution.png")
    fig.savefig(path, dpi=130)
    print(path)

    for name in names:
        design = CONVERGED.get(name)
        _, y_b, twist_b, _, twist_o = states(name, design)
        print(f"\n{name}: baseline root {twist_b[0]:+.3f} -> tip {twist_b[-1]:+.3f} deg")
        if twist_o is not None:
            print(f"{' ' * len(name)}  optimized root {twist_o[0]:+.3f} -> tip {twist_o[-1]:+.3f} deg")
            print(f"{' ' * len(name)}  max change {np.abs(twist_o - twist_b).max():+.3f} deg")


if __name__ == "__main__":
    main()
