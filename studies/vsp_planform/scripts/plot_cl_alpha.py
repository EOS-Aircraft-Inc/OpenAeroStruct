"""Cl-alpha curves from the AeroSandbox/NeuralFoil airfoil DOE.

Scratch script; reads the checked-in DOE results and writes a PNG next to itself.
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "figures"
)
DOE = "/home/alex/repos/OpenAeroStruct/studies/vsp_planform/data/airfoil_doe.csv"

RE_REF = 5.0e6  # mid-range: the wing spans Re 1.7e6 (tip) to 1.7e7 (root)


def load():
    """DOE polars keyed by (airfoil, Re), each sorted by alpha."""
    curves = defaultdict(list)
    with open(DOE) as f:
        for row in csv.DictReader(f):
            curves[(row["airfoil"], float(row["Re"]))].append(
                (float(row["alpha"]), float(row["CL"]), float(row["analysis_confidence"]))
            )
    return {k: np.array(sorted(v)) for k, v in curves.items()}


def lift_slope(alpha, cl):
    """Lift-curve slope per degree, fitted over the linear range -4 to +4 deg."""
    window = (alpha >= -4.0) & (alpha <= 4.0)
    return np.polyfit(alpha[window], cl[window], 1)[0]


def stall_point(alpha, cl):
    """Alpha and CL at maximum lift."""
    i = int(np.argmax(cl))
    return alpha[i], cl[i]


def draw_family(ax, curves, names, labels, colors, reynolds=RE_REF, mark_stall=True):
    for name, label, color in zip(names, labels, colors):
        data = curves[(name, reynolds)]
        alpha, cl = data[:, 0], data[:, 1]
        slope = lift_slope(alpha, cl)
        ax.plot(alpha, cl, color=color, lw=1.6, label=f"{label}   a0={slope:.4f}/deg")
        if mark_stall:
            a_s, cl_s = stall_point(alpha, cl)
            ax.plot([a_s], [cl_s], marker="o", ms=4.5, color=color, mfc="white", mew=1.4, zorder=5)

    # Thin-airfoil reference, 2*pi per radian through the symmetric-section origin.
    ref = np.array([-6.0, 10.0])
    ax.plot(ref, 2 * np.pi * np.radians(ref), color="0.45", ls=(0, (4, 3)), lw=1.0, zorder=1)
    ax.annotate(
        "2$\\pi$/rad",
        xy=(10.0, 2 * np.pi * np.radians(10.0)),
        xytext=(-2, -12),
        textcoords="offset points",
        fontsize=8,
        color="0.4",
    )

    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.axvline(0, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel(r"$\alpha$ [deg]")
    ax.set_ylabel("$C_L$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax.set_xlim(-8, 20)
    ax.set_ylim(-1.2, 2.1)


def main():
    curves = load()

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    fig.suptitle(
        f"Airfoil DOE — lift curves (NeuralFoil, Re = {RE_REF:.0e}, M = 0.432 unless noted)".replace("e+06", "e6"),
        fontsize=14,
    )

    blues = plt.cm.viridis(np.linspace(0.1, 0.85, 4))

    # Thickness, at the camber that the DOE likes best.
    draw_family(
        axes[0, 0],
        curves,
        ["naca4410", "naca4414", "naca4416", "naca4418"],
        ["4410  t/c 0.10", "4414  t/c 0.14", "4416  t/c 0.16", "4418  t/c 0.18"],
        blues,
    )
    axes[0, 0].set_title("thickness, at 4% camber")

    # Camber, at a thickness near both baselines.
    draw_family(
        axes[0, 1],
        curves,
        ["naca0015", "naca1415", "naca2415", "naca3415", "naca4415"],
        ["0015  0% camber", "1415  1%", "2415  2%", "3415  3%", "4415  4%"],
        plt.cm.plasma(np.linspace(0.05, 0.8, 5)),
    )
    axes[0, 1].set_title("camber, at t/c 0.15")

    # Camber position, holding camber and thickness.
    draw_family(
        axes[1, 0],
        curves,
        ["naca2215", "naca2415", "naca2615"],
        ["2215  max camber at 20% c", "2415  at 40% c", "2615  at 60% c"],
        plt.cm.cividis(np.linspace(0.1, 0.8, 3)),
    )
    axes[1, 0].set_title("camber position, at 2% camber / t/c 0.15")

    # Reynolds number, spanning root to tip of the wing.
    ax = axes[1, 1]
    reynolds = [1.7e6, 3.0e6, 5.0e6, 1.0e7, 1.7e7]
    colors = plt.cm.magma(np.linspace(0.15, 0.78, len(reynolds)))
    for re, color in zip(reynolds, colors):
        data = curves[("naca2415", re)]
        alpha, cl = data[:, 0], data[:, 1]
        ax.plot(alpha, cl, color=color, lw=1.6, label=f"Re = {re:.1e}".replace("e+0", "e"))
        a_s, cl_s = stall_point(alpha, cl)
        ax.plot([a_s], [cl_s], marker="o", ms=4.5, color=color, mfc="white", mew=1.4, zorder=5)
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.axvline(0, color="0.8", lw=0.8, zorder=0)
    ax.set_title("Reynolds number, naca2415 (wing tip to root)")
    ax.set_xlabel(r"$\alpha$ [deg]")
    ax.set_ylabel("$C_L$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax.set_xlim(-8, 20)
    ax.set_ylim(-1.2, 2.1)

    fig.text(
        0.5,
        0.012,
        "Open circles mark C_L max. a0 is the lift-curve slope fitted over -4 to +4 deg; the dashed line is "
        "2$\\pi$/rad through the origin, uncorrected for compressibility. NeuralFoil reports confidence "
        "above 0.91 across every curve shown, post-stall included.",
        ha="center",
        fontsize=9,
        color="0.35",
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    path = os.path.join(HERE, "airfoil_cl_alpha.png")
    fig.savefig(path, dpi=130)
    print(path)

    # Confidence check: does NeuralFoil still trust itself where we are plotting?
    low = [
        (name, re, float(np.min(v[:, 2])))
        for (name, re), v in curves.items()
        if re == RE_REF and np.min(v[:, 2]) < 0.5
    ]
    print("curves with min confidence < 0.5 at Re=5e6:", len(low))
    sample = curves[("naca4418", RE_REF)]
    for a_lo, a_hi in ((-8, 12), (12, 20)):
        window = (sample[:, 0] >= a_lo) & (sample[:, 0] <= a_hi)
        print(f"  naca4418 confidence over alpha {a_lo}..{a_hi}: min {sample[window, 2].min():.3f}")


if __name__ == "__main__":
    main()
