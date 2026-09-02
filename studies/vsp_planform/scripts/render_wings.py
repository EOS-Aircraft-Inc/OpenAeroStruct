"""Render the two VSP baselines from the DegenGeom camber meshes.

Scratch script, not part of the study. Writes PNGs next to itself.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from studies.vsp_planform import config
import studies.vsp_planform.mesh as M
from studies.vsp_planform.regions import detect_regions

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "figures"
)

TITLES = {"plan_l": "Plan_L  (Wing W4b2_w3)", "const_chord": "Plan_L ConstChord  (S12_t2)"}
REGION_COLORS = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}

# Vertical exaggeration for the front views. The span is ~36 m and the whole
# dihedral rise is ~1.8 m, so at true scale the front view is a 5%-tall sliver;
# letting matplotlib autoscale instead draws 4.4 deg of dihedral as if it were
# 30, which is worse than useless. A fixed, stated factor is honest and lets the
# two baselines be compared against each other.
Z_EXAGGERATION = 5.0

# Same idea for the camber lines, which are only a few percent of chord deep.
CAMBER_EXAGGERATION = 4.0


def region_bounds(name):
    """Full-mesh spanwise index bounds of regions A/B/C, per half."""
    _, stick = M.half_mesh(config.BASELINES[name])
    regions = detect_regions(stick)
    return regions


def draw_lines(ax, mesh, axes, color, lw=0.5, alpha=0.9):
    """Draw the chordwise and spanwise mesh lines in a 2D projection."""
    i, j = axes
    for k in range(mesh.shape[1]):
        ax.plot(mesh[:, k, i], mesh[:, k, j], color=color, lw=lw, alpha=alpha)
    for k in range(mesh.shape[0]):
        ax.plot(mesh[k, :, i], mesh[k, :, j], color=color, lw=lw, alpha=alpha)


def region_meshes(mesh, regions, num_secs):
    """Split a full mesh into A/B/C blocks (both halves), for coloring."""
    a, c = regions.idx_a_end, regions.idx_c_start
    root = num_secs - 1  # index of the root section in the full mesh
    blocks = {
        "C": [mesh[:, : root - c + 1, :], mesh[:, root + c :, :]],
        "B": [mesh[:, root - c : root - a + 1, :], mesh[:, root + a : root + c + 1, :]],
        "A": [mesh[:, root - a : root + a + 1, :]],
    }
    return blocks


def planform_figure(name):
    mesh = M.full_mesh(config.BASELINES[name])
    _, stick = M.half_mesh(config.BASELINES[name])
    regions = detect_regions(stick)
    num_secs = stick.num_secs
    blocks = region_meshes(mesh, regions, num_secs)

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"{TITLES[name]} — DegenGeom camber surface, {mesh.shape[0]}x{mesh.shape[1]} nodes", fontsize=14)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.85, 1.5], hspace=0.38, wspace=0.22)

    # Top view, spanning the full width: the span is 10x the chord, so it needs it.
    ax = fig.add_subplot(grid[0, :])
    for label, parts in blocks.items():
        for part in parts:
            draw_lines(ax, part, (1, 0), REGION_COLORS[label])
    ax.set_title("top view (planform)")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("x [m]")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    handles = [plt.Line2D([], [], color=c, lw=2, label=f"region {k}") for k, c in REGION_COLORS.items()]
    ax.legend(handles=handles, loc="upper center", ncol=3, fontsize=9, framealpha=0.9)

    # Front view. Not equal-aspect: at true scale the dihedral is a hairline.
    ax = fig.add_subplot(grid[1, :])
    for label, parts in blocks.items():
        for part in parts:
            draw_lines(ax, part, (1, 2), REGION_COLORS[label])
    ax.set_title(f"front view (dihedral and winglet), z exaggerated {Z_EXAGGERATION:g}x")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect(Z_EXAGGERATION)
    ax.grid(alpha=0.3)

    # 3D.
    ax = fig.add_subplot(grid[2, 0], projection="3d")
    for label, parts in blocks.items():
        for part in parts:
            ax.plot_wireframe(
                part[:, :, 0], part[:, :, 1], part[:, :, 2], color=REGION_COLORS[label], lw=0.4, rstride=1, cstride=1
            )
    # True scale: the box aspect follows the actual data ranges, so this panel is
    # the undistorted reference against which the exaggerated views are read.
    spans = [float(np.ptp(mesh[:, :, i])) for i in range(3)]
    ax.set_title("camber surface, true scale", y=1.06, loc="left", fontsize=11)
    ax.set_xlabel("x [m]", labelpad=-4)
    ax.set_ylabel("y [m]", labelpad=6)
    # Seen from above at true scale, z spans a few percent of the frame; its tick
    # labels only collide with the title, and the front view above carries z.
    ax.set_zlabel("")
    ax.set_zticklabels([])
    # A true-scale wing is a thin ribbon, so view it from well above: near the
    # horizon it degenerates into a line and shows nothing.
    ax.set_box_aspect([s / max(spans) for s in spans], zoom=1.3)
    ax.view_init(elev=52, azim=-70)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(plt.MaxNLocator(4))
    ax.tick_params(labelsize=7, pad=-1)

    # Sections, root / mid / winglet root / tip, in the local chord frame.
    ax = fig.add_subplot(grid[2, 1])
    half = mesh[:, num_secs - 1 :, :]
    picks = [
        (0, "root"),
        (regions.idx_a_end, "end of region A"),
        ((regions.idx_a_end + regions.idx_c_start) // 2, "mid region B"),
        (regions.idx_c_start, "winglet root"),
        (-1, "tip"),
    ]
    for idx, label in picks:
        section = half[:, idx, :]
        chord = np.linalg.norm(section[-1] - section[0])
        ax.plot(
            (section[:, 0] - section[0, 0]) / chord,
            (section[:, 2] - section[0, 2]) / chord,
            marker="o",
            ms=2.5,
            lw=1.2,
            label=f"{label} (c={chord:.2f} m)",
        )
    ax.set_title(f"camber lines, normalized by local chord (z exaggerated {CAMBER_EXAGGERATION:g}x)")
    ax.set_xlabel("(x - x_LE) / c")
    ax.set_ylabel("(z - z_LE) / c")
    ax.set_aspect(CAMBER_EXAGGERATION)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.07, right=0.97)
    path = os.path.join(OUT, f"{name}_geometry.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def resample_figure(name):
    """Native mesh against the resampled VLM mesh."""
    mesh, stick = M.half_mesh(config.BASELINES[name])
    regions = detect_regions(stick)
    y = M.spanwise_stations(mesh[0, :, 1], config.N_SPANWISE_HALF, regions.y_c_start * config.SCALE)
    new_mesh, residual = M.resample(mesh, y, M.N_CHORDWISE)

    fig = plt.figure(figsize=(13, 11))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.9, 1.3], hspace=0.35, wspace=0.2)
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, :]), fig.add_subplot(grid[2, 0])]
    native_panels = (mesh.shape[0] - 1) * (mesh.shape[1] - 1)
    new_panels = (new_mesh.shape[0] - 1) * (new_mesh.shape[1] - 1)
    fig.suptitle(
        f"{TITLES[name]} — resampling, {mesh.shape[0]}x{mesh.shape[1]} -> {new_mesh.shape[0]}x{new_mesh.shape[1]} "
        f"({native_panels} -> {new_panels} panels per half)",
        fontsize=13,
    )

    for ax, (i, j), title, xlabel, ylabel, equal in [
        (axes[0], (1, 0), "top view", "y [m]", "x [m]", True),
        (axes[1], (1, 2), f"front view (z exaggerated {Z_EXAGGERATION:g}x)", "y [m]", "z [m]", False),
    ]:
        draw_lines(ax, mesh, (i, j), "0.75", lw=0.4)
        draw_lines(ax, new_mesh, (i, j), "#C44E52", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if equal:
            ax.set_aspect("equal")
        else:
            ax.set_aspect(Z_EXAGGERATION)
            ax.grid(alpha=0.3)
    axes[0].invert_yaxis()
    axes[0].legend(
        handles=[
            plt.Line2D([], [], color="0.75", lw=1.5, label=f"native {mesh.shape[0]}x{mesh.shape[1]}"),
            plt.Line2D([], [], color="#C44E52", lw=1.5, label=f"resampled {new_mesh.shape[0]}x{new_mesh.shape[1]}"),
        ],
        loc="lower center",
        fontsize=9,
    )

    # Winglet close-up, where the resampling has to work hardest.
    ax = axes[2]
    draw_lines(ax, mesh, (1, 2), "0.75", lw=0.5)
    draw_lines(ax, new_mesh, (1, 2), "#C44E52", lw=1.0)
    y_c = regions.y_c_start * config.SCALE
    ax.set_xlim(y_c - 0.6, mesh[0, -1, 1] + 0.15)
    z_c = np.interp(y_c, mesh[0, :, 1], mesh[0, :, 2])
    ax.set_ylim(z_c - 0.35, mesh[0, :, 2].max() + 0.25)
    ax.axvline(y_c, color="#55A868", ls="--", lw=1, label=f"winglet start, y={y_c:.2f} m")
    ax.set_title("winglet close-up (front view)")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")

    # Chordwise convergence, so the choice of N_CHORDWISE is visible.
    ax = fig.add_subplot(grid[2, 1])
    counts = [5, 7, 9, 13, 17, 21]
    span_mesh, _ = M.resample_spanwise(mesh, y)
    errors = [M.resample_chordwise(span_mesh, n)[1] for n in counts]
    ax.plot(counts, [100 * e["max_relative"] for e in errors], "o-", label="max")
    ax.plot(counts, [100 * e["rms_relative"] for e in errors], "s-", label="rms")
    ax.axvline(M.N_CHORDWISE, color="0.4", ls="--", lw=1)
    ax.annotate(
        f"N_CHORDWISE = {M.N_CHORDWISE}",
        xy=(M.N_CHORDWISE, ax.get_ylim()[1]),
        xytext=(4, -12),
        textcoords="offset points",
        fontsize=9,
        color="0.3",
    )
    ax.set_yscale("log")
    ax.set_title("chordwise camber error vs node count")
    ax.set_xlabel("chordwise nodes")
    ax.set_ylabel("error [% of local chord]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    text = (
        f"spanwise residual   max {residual['spanwise']['max'] * 1e3:.1f} mm   "
        f"rms {residual['spanwise']['rms'] * 1e3:.1f} mm\n"
        f"chordwise residual  max {residual['chordwise']['max'] * 1e3:.1f} mm "
        f"({residual['chordwise']['max_relative'] * 100:.2f}% c)   "
        f"rms {residual['chordwise']['rms'] * 1e3:.1f} mm"
    )
    fig.text(0.5, 0.012, text, ha="center", fontsize=9, family="monospace")

    fig.subplots_adjust(top=0.93, bottom=0.09, left=0.07, right=0.97)
    path = os.path.join(OUT, f"{name}_resampling.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for baseline in config.BASELINES:
        print(planform_figure(baseline))
        print(resample_figure(baseline))
