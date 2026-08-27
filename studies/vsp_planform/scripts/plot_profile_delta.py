"""Geometric delta, baseline vs optimized, for the ConstChord baseline.

ConstChord is the case that actually converged (31 iterations), so it is the one
whose optimized geometry is worth reading. Design vector from opt_sized.log,
fixed_lift at MTOW with the area pinned by the cruise-CL limit.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from studies.vsp_planform import config
from studies.vsp_planform.run_opt import POINT, build_problem, load_baseline, trim_alpha

HERE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "figures"
)
NAME = "const_chord"
WEIGHT = 382547.0

OPT = {
    "alpha": 4.15828236,
    "taper_B": 0.16313306,
    "twist_cp": np.array([-2.65077686, -2.5275155, -1.32748637, 3.89971517, 3.6938989]),
    "wingbox_pct": 0.75,
}

BASE_COLOR = "#4C72B0"
OPT_COLOR = "#C44E52"


def geometry(prob):
    mesh = prob.get_val("wing.mesh", units="m")
    return {
        "mesh": mesh.copy(),
        "y": mesh[0, :, 1].copy(),
        "le": mesh[0, :, 0].copy(),
        "te": mesh[-1, :, 0].copy(),
        "chord": (mesh[-1, :, 0] - mesh[0, :, 0]).copy(),
        "twist": prob.get_val("twist_abs", units="deg").copy(),
        "S_ref": float(prob.get_val(f"{POINT}.wing.S_ref")[0]),
    }


def main():
    mesh, stick, regions, planform0, _, _ = load_baseline(NAME, config.N_SPANWISE_HALF, 9)
    prob, _ = build_problem(NAME, mesh, stick, regions, planform0)

    prob.run_model()
    q = 0.5 * config.RHO * config.V_MS**2
    trim_alpha(prob, WEIGHT / (q * float(prob.get_val(f"{POINT}.wing.S_ref")[0])))
    base = geometry(prob)

    prob.set_val("alpha", OPT["alpha"], units="deg")
    prob.set_val("wing.taper_B", OPT["taper_B"])
    prob.set_val("wing.twist_cp", OPT["twist_cp"], units="deg")
    prob.set_val("wing.wingbox_pct", OPT["wingbox_pct"])
    prob.run_model()
    opt = geometry(prob)

    y = base["y"]
    semi = y[-1]
    eta = y / semi
    keep = y >= 0

    fig = plt.figure(figsize=(14, 13))
    fig.suptitle(
        f"ConstChord — geometric delta, baseline vs optimized\n"
        f"min drag at MTOW 382.5 kN, area pinned at {opt['S_ref']:.2f} m$^2$ by the CL $\\leq$ 1.05 limit",
        fontsize=14,
    )
    grid = fig.add_gridspec(3, 2, hspace=0.34, wspace=0.24, top=0.90, bottom=0.06, left=0.08, right=0.97)

    # Planform overlay.
    ax = fig.add_subplot(grid[0, :])
    for label, g, color in (("baseline", base, BASE_COLOR), ("optimized", opt, OPT_COLOR)):
        ax.plot(y[keep], g["le"][keep], "-", color=color, lw=1.7, label=label)
        ax.plot(y[keep], g["te"][keep], "-", color=color, lw=1.7)
        ax.fill_between(y[keep], g["le"][keep], g["te"][keep], color=color, alpha=0.10)
    ax.set_title("planform, half span (chord shown in true position)", fontsize=11)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("x [m]")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    # Chord.
    ax = fig.add_subplot(grid[1, 0])
    ax.plot(eta[keep], base["chord"][keep], "-", color=BASE_COLOR, lw=1.7, label="baseline")
    ax.plot(eta[keep], opt["chord"][keep], "-", color=OPT_COLOR, lw=1.7, label="optimized")
    ax.set_title("chord", fontsize=11)
    ax.set_xlabel(r"$\eta$ = y / semi-span")
    ax.set_ylabel("chord [m]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    # Chord delta.
    ax = fig.add_subplot(grid[1, 1])
    dchord = opt["chord"] - base["chord"]
    ax.plot(eta[keep], dchord[keep], "-", color="#7d4b9a", lw=1.7)
    ax.fill_between(eta[keep], 0, dchord[keep], color="#7d4b9a", alpha=0.18)
    ax.axhline(0, color="0.6", lw=0.9)
    ax.set_title("chord delta (optimized - baseline)", fontsize=11)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\Delta$ chord [m]")
    ax.grid(alpha=0.25)

    # Twist.
    ax = fig.add_subplot(grid[2, 0])
    ax.plot(eta[keep], base["twist"][keep], "-", color=BASE_COLOR, lw=1.7, label="baseline")
    ax.plot(eta[keep], opt["twist"][keep], "-", color=OPT_COLOR, lw=1.7, label="optimized")
    ax.axhline(0, color="0.85", lw=0.8)
    ax.set_title("twist", fontsize=11)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("twist [deg]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    # Edge movement.
    ax = fig.add_subplot(grid[2, 1])
    ax.plot(eta[keep], (opt["le"] - base["le"])[keep], "-", color="#2a9d8f", lw=1.7, label="leading edge")
    ax.plot(eta[keep], (opt["te"] - base["te"])[keep], "-", color="#e07a5f", lw=1.7, label="trailing edge")
    ax.axhline(0, color="0.6", lw=0.9)
    ax.set_title("edge movement in x (aft positive)", fontsize=11)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\Delta x$ [m]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    for axis in fig.axes[1:]:
        for frac, tag in (
            (regions.y_a_end * config.SCALE / semi, "A|B"),
            (regions.y_c_start * config.SCALE / semi, "B|C"),
        ):
            axis.axvline(frac, color="0.78", ls="--", lw=1, zorder=0)

    path = os.path.join(HERE, "const_chord_profile_delta.png")
    fig.savefig(path, dpi=130)
    print(path)

    root = int(np.argmax(keep))
    print(f"\n{'eta':>6} {'chord base':>11} {'chord opt':>10} {'delta':>8} {'twist base':>11} {'twist opt':>10}")
    for i in range(root, len(y), 4):
        print(
            f"{eta[i]:6.3f} {base['chord'][i]:11.3f} {opt['chord'][i]:10.3f} "
            f"{dchord[i]:8.3f} {base['twist'][i]:11.3f} {opt['twist'][i]:10.3f}"
        )
    print(f"\nS_ref {base['S_ref']:.3f} -> {opt['S_ref']:.3f} m^2 ({opt['S_ref'] / base['S_ref'] - 1:+.2%})")
    print(f"root chord {base['chord'][root]:.3f} -> {opt['chord'][root]:.3f} m")
    print(f"tip chord  {base['chord'][-1]:.3f} -> {opt['chord'][-1]:.3f} m")


if __name__ == "__main__":
    main()
