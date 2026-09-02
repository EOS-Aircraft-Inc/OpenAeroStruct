"""Draw the wing 2 best case (case C of wing2_oas.py) in 2D and 3D.

Rebuilds the geometry from the design vector in ``out/logs/wing2_oas.json``
rather than re-running the optimization, so this is cheap and always shows
exactly the wing that produced the reported drag.

Two figures:

``wing2_planform.png``  the 2D set -- planform with the wingbox and its
                        constraint stations drawn on it, chord and box width
                        against span, front view, root/tip sections, twist.
``wing2_3d.png``        isometric wireframes, true scale.

The wingbox is drawn from the same schedule the optimizer was constrained with,
so what is shaded on the planform IS the constraint, not an illustration of it.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

from studies.vsp_planform import config  # noqa: E402
from studies.vsp_planform.param import rear_spar_fraction  # noqa: E402
from studies.vsp_planform.run_opt import POINT  # noqa: E402

import wing2_oas as W  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "figures")
LOG = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs", "wing2_oas.json")

BASE_C = "#4C72B0"
BEST_C = "#C44E52"
BOX_C = "#55A868"

# The span is ~18 m per half and the whole dihedral rise is under 2 m, so the
# front view is drawn with a stated exaggeration rather than autoscaled. Same
# convention as render_wings.py.
Z_EXAGGERATION = 5.0


def rebuild(case, y_a_new_in):
    """Rebuild one case's geometry from its stored design vector."""
    prob, mesh0, stick, regions, planform0 = W.build(W.BASELINE, y_a_new_in)
    prob.set_val("wing.taper_B", case["taper_B"])
    prob.set_val("wing.wingbox_pct", case["wingbox_pct"])
    prob.set_val("wing.twist_cp", np.array(case["twist_cp"]), units="deg")
    prob.set_val("alpha", case["alpha"], units="deg")
    prob.run_model()

    mesh = prob.get_val("wing.mesh", units="m")
    drag = float(prob.get_val("drag")[0])
    # Confirm we rebuilt the wing that was actually reported, rather than
    # quietly drawing something else.
    if abs(drag - case["drag_N"]) > 1.0:
        raise RuntimeError(f"rebuilt drag {drag:.1f} N does not match the logged {case['drag_N']:.1f} N")

    return {
        "mesh": mesh,
        "twist": prob.get_val("twist_abs", units="deg").copy(),
        "S_ref": float(prob.get_val(f"{POINT}.wing.S_ref")[0]),
        "drag": drag,
        "regions": regions,
        "log": case,
    }


def planform(mesh):
    """Leading edge, trailing edge and span stations of a half mesh, inches."""
    y = np.abs(mesh[0, :, 1]) / config.SCALE
    return y, mesh[0, :, 0] / config.SCALE, mesh[-1, :, 0] / config.SCALE


def box_edges(mesh):
    """Front and rear spar x stations along the span, inches."""
    y, le, te = planform(mesh)
    chord = te - le
    front = le + W.FRONT_PCT * chord
    rear = le + rear_spar_fraction(y, W.REAR_SCHEDULE) * chord
    return y, front, rear


def spar_line(mesh, fractions):
    """3D coordinates of a chord-fraction line along the span.

    Interpolated along the actual chordwise mesh rather than straight between
    the leading and trailing edge, so the line sits on the camber surface.
    """
    ny = mesh.shape[1]
    out = np.zeros((ny, 3))
    frac = np.broadcast_to(np.asarray(fractions, dtype=float), (ny,))
    for j in range(ny):
        x = mesh[:, j, 0]
        xi = (x - x[0]) / (x[-1] - x[0])
        for k in range(3):
            out[j, k] = np.interp(frac[j], xi, mesh[:, j, k])
    return out


def mirror(mesh):
    """Mirror a half mesh about y = 0 into a full wing."""
    left = mesh[:, ::-1, :].copy()
    left[:, :, 1] *= -1.0
    return np.hstack((left[:, :-1, :], mesh))


def draw_mesh(ax, mesh, axes, color, lw=0.4, alpha=0.85, z_scale=1.0):
    i, j = axes
    for k in range(mesh.shape[1]):
        ax.plot(mesh[:, k, i], mesh[:, k, j] * z_scale, color=color, lw=lw, alpha=alpha)
    for k in range(mesh.shape[0]):
        ax.plot(mesh[k, :, i], mesh[k, :, j] * z_scale, color=color, lw=lw, alpha=alpha)


def figure_2d(base, best):
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(
        "Wing 2 best case — ConstChord, region A re-lofted to the inboard nacelle, "
        "kinked rear spar\n"
        f"drag {base['drag']:.0f} -> {best['drag']:.0f} N "
        f"({best['drag'] / base['drag'] - 1:+.2%}) at MTOW, span pinned at 118 ft",
        fontsize=13,
    )
    gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.22, top=0.90, bottom=0.06)

    # --- planform with the wingbox on it
    ax = fig.add_subplot(gs[0, :])
    y_b, le_b, te_b = planform(base["mesh"])
    ax.plot(y_b, le_b, color=BASE_C, lw=1.2, ls="--", label="as-built")
    ax.plot(y_b, te_b, color=BASE_C, lw=1.2, ls="--")

    y, le, te = planform(best["mesh"])
    ax.plot(y, le, color=BEST_C, lw=1.8, label="best case")
    ax.plot(y, te, color=BEST_C, lw=1.8)

    y_box, front, rear = box_edges(best["mesh"])
    inboard = y_box <= abs(best["regions"].y_c_start)
    ax.fill_between(
        y_box[inboard], front[inboard], rear[inboard], color=BOX_C, alpha=0.30, lw=0, label="wingbox"
    )
    ax.plot(y_box[inboard], front[inboard], color=BOX_C, lw=1.0)
    ax.plot(y_box[inboard], rear[inboard], color=BOX_C, lw=1.0)

    # The constraint stations, with what they got against what they needed.
    for (y_st, req), width, margin in zip(
        W.WIDTH_STATIONS, best["log"]["box_width_in"], best["log"]["box_margin_in"]
    ):
        f = np.interp(y_st, y_box, front)
        r = np.interp(y_st, y_box, rear)
        ax.plot([y_st, y_st], [f, r], color="0.15", lw=2.2, solid_capstyle="butt")
        tight = "\n(binding)" if abs(margin) < 0.05 else ""
        # The axis is inverted, so a positive point offset is forward of the
        # front spar -- clear of the shaded box rather than on top of it.
        ax.annotate(
            f"y={y_st:.0f}\"\n{width:.1f}\" / {req:.0f}\"{tight}",
            xy=(y_st, f),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="0.15",
        )

    for x, lb in ((abs(best["regions"].y_a_end), "A|B"), (abs(best["regions"].y_c_start), "B|C")):
        ax.axvline(x, color="0.75", ls="--", lw=1)
        ax.annotate(lb, xy=(x, ax.get_ylim()[1]), fontsize=8, color="0.5", ha="center", va="top")

    ax.set_xlim(-40, 780)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set(xlabel="y [in]", ylabel="x [in]")
    # Padded clear of the station callouts, which sit above the box.
    ax.set_title("planform, half span — wingbox shaded, constraint stations marked (got / required)", pad=30)
    ax.legend(fontsize=9, loc="lower left")

    # --- chord and box width against span
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(y_b, te_b - le_b, color=BASE_C, lw=1.4, ls="--", label="as-built chord")
    ax.plot(y, te - le, color=BEST_C, lw=1.8, label="best case chord")
    ax.plot(y_box[inboard], (rear - front)[inboard], color=BOX_C, lw=1.8, label="box width")
    ax.plot(
        [s[0] for s in W.WIDTH_STATIONS],
        [s[1] for s in W.WIDTH_STATIONS],
        "v",
        color="0.15",
        ms=7,
        label="box width required",
    )
    for x in (abs(best["regions"].y_a_end), abs(best["regions"].y_c_start)):
        ax.axvline(x, color="0.75", ls="--", lw=1)
    ax.grid(alpha=0.25)
    ax.set(title="chord and wingbox width", xlabel="y [in]", ylabel="[in]")
    ax.legend(fontsize=8)

    # --- rear spar schedule, the thing that kinks
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(y_box[inboard], rear_spar_fraction(y_box, W.REAR_SCHEDULE)[inboard], color=BOX_C, lw=2.0)
    ax.axhline(W.FRONT_PCT, color=BOX_C, lw=1.2, ls=":")
    # Inverted axis: a negative point offset drops the label below the line.
    ax.annotate(f"front spar {W.FRONT_PCT:.2f}c", xy=(300, W.FRONT_PCT), xytext=(0, -12),
                textcoords="offset points", fontsize=8, color=BOX_C, ha="center")
    ax.set_ylim(0.06, 0.80)
    for y_k, pct in W.REAR_SCHEDULE:
        ax.plot(y_k, pct, "o", color=BOX_C, ms=6)
        ax.annotate(f"{pct:.3f}c at y={y_k:.0f}\"", xy=(y_k, pct), xytext=(-6, 8),
                    textcoords="offset points", fontsize=8, ha="right", color="0.25")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.set(title="rear spar schedule (kinks forward outboard)", xlabel="y [in]", ylabel="x/c")

    # --- front view
    ax = fig.add_subplot(gs[2, 0])
    draw_mesh(ax, mirror(base["mesh"]), (1, 2), BASE_C, z_scale=Z_EXAGGERATION)
    draw_mesh(ax, mirror(best["mesh"]), (1, 2), BEST_C, z_scale=Z_EXAGGERATION)
    ax.set_aspect(Z_EXAGGERATION)
    ax.grid(alpha=0.25)
    ax.set(title=f"front view (z exaggerated {Z_EXAGGERATION:g}x)", xlabel="y [m]", ylabel="z [m]")

    # --- twist
    ax = fig.add_subplot(gs[2, 1])
    for lb, color, case in (("as-built", BASE_C, base), ("best case", BEST_C, best)):
        yy = np.abs(case["mesh"][0, :, 1])
        ax.plot(yy / yy.max(), case["twist"], color=color, lw=1.7, marker="o", ms=3, label=lb)
    for x in (abs(best["regions"].y_a_end), abs(best["regions"].y_c_start)):
        ax.axvline(x * config.SCALE / np.abs(best["mesh"][0, :, 1]).max(), color="0.75", ls="--", lw=1)
    ax.grid(alpha=0.25)
    ax.set(title="twist distribution", xlabel=r"$\eta$", ylabel="twist [deg]")
    ax.legend(fontsize=9)

    path = os.path.join(OUT_DIR, "wing2_planform.png")
    fig.savefig(path, dpi=130)
    return path


def _draw_3d(ax, cases, best=None, y_lim_in=None, stride=1):
    """One 3D panel: wireframes, optionally the spar lines, optionally cropped."""
    for _, color, case, lw in cases:
        m = case["mesh"]
        if y_lim_in is not None:
            keep = np.abs(m[0, :, 1]) / config.SCALE >= y_lim_in
            m = m[:, keep, :]
        ax.plot_wireframe(
            m[:, :, 0], m[:, :, 1], m[:, :, 2], color=color, lw=lw, rstride=stride, cstride=stride, alpha=0.9
        )

    if best is not None:
        m = best["mesh"]
        y_in = np.abs(m[0, :, 1]) / config.SCALE
        inboard = y_in <= abs(best["regions"].y_c_start)
        if y_lim_in is not None:
            inboard &= y_in >= y_lim_in
        for frac, style in ((np.full(y_in.size, W.FRONT_PCT), ":"), (rear_spar_fraction(y_in, W.REAR_SCHEDULE), "-")):
            line = spar_line(m, frac)[inboard]
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color=BOX_C, lw=2.2, ls=style, zorder=5)

    # True proportions, but built from the panel's own extent so a cropped view
    # is not squashed by the full span.
    ref = np.concatenate([c[2]["mesh"].reshape(-1, 3) for c in cases])
    if y_lim_in is not None:
        ref = ref[np.abs(ref[:, 1]) / config.SCALE >= y_lim_in]
    span = [float(np.ptp(ref[:, i])) for i in range(3)]
    ax.set_box_aspect([s / max(span) for s in span], zoom=1.15)

    ax.set_xlabel("x [m]", labelpad=8, fontsize=8)
    ax.set_ylabel("y [m]", labelpad=8, fontsize=8)
    ax.set_zticklabels([])
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_major_locator(plt.MaxNLocator(4))
        a.pane.set_alpha(0.04)
    ax.tick_params(labelsize=7, pad=0)
    ax.grid(alpha=0.2)


def figure_3d(base, best):
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(
        "Wing 2 best case in 3D — half wing, true scale, front (dotted) and rear (solid) spar drawn\n"
        f"S_ref {base['S_ref']:.2f} -> {best['S_ref']:.2f} m$^2$, "
        f"drag {base['drag']:.0f} -> {best['drag']:.0f} N",
        fontsize=13,
    )
    gs = fig.add_gridspec(2, 2, hspace=0.05, wspace=0.05, top=0.90, bottom=0.03, left=0.02, right=0.98)

    both = [("as-built", BASE_C, base, 0.3), ("best", BEST_C, best, 0.45)]

    # Planform-ish view from above, where the re-lofted taper reads.
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    _draw_3d(ax, both, best)
    ax.view_init(elev=62, azim=-72)
    ax.set_title("from above — as-built (blue) vs best (red)", y=0.96, loc="left", fontsize=10)

    # Low angle, where the winglet and the dihedral read.
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    _draw_3d(ax, both, best)
    ax.view_init(elev=14, azim=-62)
    ax.set_title("low angle — winglet and dihedral", y=0.96, loc="left", fontsize=10)

    # Best case alone, the headline view.
    ax = fig.add_subplot(gs[1, 0], projection="3d")
    _draw_3d(ax, [("best", BEST_C, best, 0.45)], best)
    ax.view_init(elev=34, azim=-66)
    ax.set_title("best case, isometric", y=0.96, loc="left", fontsize=10)

    # Outboard crop: this is where the rear spar kinks forward and where the
    # 25 in junction box constraint binds, so it is worth its own panel.
    ax = fig.add_subplot(gs[1, 1], projection="3d")
    _draw_3d(ax, [("best", BEST_C, best, 0.8)], best, y_lim_in=330.0)
    ax.view_init(elev=44, azim=-70)
    ax.set_title("outboard half — the rear spar kinking forward to 0.499c", y=0.96, loc="left", fontsize=10)

    path = os.path.join(OUT_DIR, "wing2_3d.png")
    fig.savefig(path, dpi=130)
    return path


if __name__ == "__main__":
    W.apply_wing2_box()

    with open(LOG) as fh:
        log = json.load(fh)
    best_case = log["C_optimized"]

    base = rebuild(log["A_baseline"], None)
    best = rebuild(best_case, W.REGION_A_END_IN)

    print(f"  as-built  drag {base['drag']:9.1f} N   S_ref {base['S_ref']:.3f} m^2")
    print(f"  best case drag {best['drag']:9.1f} N   S_ref {best['S_ref']:.3f} m^2"
          f"   ({best['drag'] / base['drag'] - 1:+.2%})")

    for path in (figure_2d(base, best), figure_3d(base, best)):
        print(f"  wrote {path}")
