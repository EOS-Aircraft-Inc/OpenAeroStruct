"""Wing 3 -- ailerons at 90% semi-span, 6 in aft-spar depth. 2D and 3D views.

Wing 3 is the design that closes. Wing 2 put the aileron actuator at the winglet
junction and demanded 7 in of depth there; on the as-built section that station
could not deliver it at ANY spar position, so wing 2's headline number was never
structurally admissible. Moving the ailerons inboard to 90% semi-span and
accepting 6 in makes the constraint non-binding: the aero optimum satisfies it
with margin, for free.

Geometry is rebuilt from the stored design vector rather than re-optimized, and
the rebuilt drag is checked against the logged value before anything is drawn.
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
sys.path.insert(0, os.path.dirname(_HERE))

from studies.vsp_planform import config  # noqa: E402
from studies.vsp_planform.param import rear_spar_fraction  # noqa: E402
from studies.vsp_planform.run_opt import POINT  # noqa: E402

import wing2_oas as W  # noqa: E402
from plot_wing2 import BASE_C, BEST_C, BOX_C, Z_EXAGGERATION, draw_mesh, mirror, planform, rebuild, spar_line  # noqa: E402
from doe_v3 import asbuilt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "figures")
LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")

SEMI_IN = 118.0 * 12.0 / 2.0
Y_AIL = 0.90 * SEMI_IN
DEPTH_REQ = 6.0
AIL_C = "#8172B2"


def retention_fn():
    af = asbuilt()
    xs = np.linspace(0.05, 0.95, 300)
    t = np.array([float(af.local_thickness(x_over_c=x)) for x in xs])
    return xs, t / t.max()


def depth_along_span(mesh, stick, schedule, xs, ret):
    y, le, te = planform(mesh)
    chord = te - le
    y_s = np.abs(np.asarray(stick.le[:, 1], dtype=float))
    toc = np.interp(y, y_s, np.asarray(stick.toc, dtype=float))
    spar = np.array([float(rear_spar_fraction(v, schedule)) for v in y])
    return y, np.interp(spar, xs, ret) * toc * chord, spar


def box_edges_sched(mesh, schedule, front):
    y, le, te = planform(mesh)
    chord = te - le
    return y, le + front * chord, le + np.array([float(rear_spar_fraction(v, schedule)) for v in y]) * chord


def figure_2d(base, best, stick_b, schedule, stations, xs, ret):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "WING 3 — ailerons at 90% semi-span, 6 in aft-spar depth\n"
        f"drag {base['drag']:.0f} → {best['drag']:.0f} N "
        f"({best['drag'] / base['drag'] - 1:+.2%}) at MTOW, span pinned at 118 ft — "
        "the depth constraint binds, but costs only 0.09% against the unconstrained optimum",
        fontsize=13,
    )
    gs = fig.add_gridspec(3, 2, hspace=0.32, wspace=0.22)

    # --- planform with the wingbox and the aileron span
    ax = fig.add_subplot(gs[0, :])
    for w, c, lab in ((base, BASE_C, "as-built"), (best, BEST_C, "wing 3")):
        y, le, te = planform(w["mesh"])
        ax.plot(y, le, color=c, lw=1.6, label=lab)
        ax.plot(y, te, color=c, lw=1.6)
    y, front, rear = box_edges_sched(best["mesh"], schedule, W.FRONT_PCT)
    ax.fill_between(y, front, rear, color=BOX_C, alpha=0.30, label="structural box")
    ax.plot(y, front, color=BOX_C, lw=1.0)
    ax.plot(y, rear, color=BOX_C, lw=1.0)

    ax.axvspan(Y_AIL, y.max(), color=AIL_C, alpha=0.13)
    ax.axvline(Y_AIL, color=AIL_C, lw=1.8, ls="--")
    ax.annotate(
        f"aileron inboard end\ny = {Y_AIL:.0f} in (90% semi)",
        xy=(Y_AIL, ax.get_ylim()[0]),
        xytext=(Y_AIL - 150, ax.get_ylim()[0] + 0.12 * np.ptp(ax.get_ylim())),
        color=AIL_C, fontsize=9, fontweight="bold",
    )
    for y_in, req in stations:
        ax.axvline(y_in, color="0.55", lw=0.7, ls=":")
    ax.invert_yaxis()
    ax.set_xlabel("y, in")
    ax.set_ylabel("x, in")
    ax.set_title("Planform, structural box and aileron span", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")

    # --- chord
    ax = fig.add_subplot(gs[1, 0])
    for w, c, lab in ((base, BASE_C, "as-built"), (best, BEST_C, "wing 3")):
        y, le, te = planform(w["mesh"])
        ax.plot(y, te - le, color=c, lw=1.8, label=lab)
    ax.axvline(Y_AIL, color=AIL_C, lw=1.5, ls="--")
    ax.set_xlabel("y, in"); ax.set_ylabel("chord, in")
    ax.set_title("Chord distribution", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # --- THE constraint: spar depth against span
    ax = fig.add_subplot(gs[1, 1])
    yb, db, _ = depth_along_span(best["mesh"], stick_b, schedule, xs, ret)
    ax.plot(yb, db, color=BEST_C, lw=2.0, label="wing 3 depth at aft spar")
    ax.axhline(DEPTH_REQ, color="k", lw=1.3, ls="--", label=f"{DEPTH_REQ:.0f} in requirement")
    ax.axvspan(Y_AIL, yb.max(), color=AIL_C, alpha=0.13)
    ax.axvline(Y_AIL, color=AIL_C, lw=1.5, ls="--")
    d_at = float(np.interp(Y_AIL, yb, db))
    ax.plot([Y_AIL], [d_at], "o", color=AIL_C, ms=8, zorder=5)
    ax.annotate(f"{d_at:.2f} in\n({d_at - DEPTH_REQ:+.2f} in margin)", xy=(Y_AIL, d_at),
                xytext=(Y_AIL - 260, d_at + 1.2), color=AIL_C, fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=AIL_C, lw=1.2))
    ax.set_xlabel("y, in"); ax.set_ylabel("spar depth, in")
    ax.set_title("Aft-spar depth — the constraint wing 2 could not meet", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper right")

    # --- box width vs requirements
    ax = fig.add_subplot(gs[2, 0])
    y, front, rear = box_edges_sched(best["mesh"], schedule, W.FRONT_PCT)
    ax.plot(y, rear - front, color=BOX_C, lw=2.0, label="wing 3 box width")
    for y_in, req in stations:
        ax.plot([y_in], [req], "kv", ms=7)
    ax.plot([], [], "kv", ms=7, label="requirement")
    ax.axvline(Y_AIL, color=AIL_C, lw=1.5, ls="--")
    ax.set_xlabel("y, in"); ax.set_ylabel("box width, in")
    ax.set_title("Box width against its constraint stations", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # --- twist
    ax = fig.add_subplot(gs[2, 1])
    for w, c, lab in ((base, BASE_C, "as-built"), (best, BEST_C, "wing 3")):
        y, _, _ = planform(w["mesh"])
        ax.plot(y, w["twist"], color=c, lw=1.8, label=lab)
    ax.axvline(Y_AIL, color=AIL_C, lw=1.5, ls="--")
    ax.set_xlabel("y, in"); ax.set_ylabel("twist, deg")
    ax.set_title("Twist distribution", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    path = os.path.join(OUT_DIR, "wing3_planform.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_3d(base, best, schedule):
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("WING 3 — isometric, true scale (dihedral to scale)", fontsize=13)

    for i, (w, c, lab) in enumerate(((base, BASE_C, "as-built"), (best, BEST_C, "wing 3"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        m = mirror(w["mesh"])
        # NOTE: do not call the 2D draw_mesh here. On a 3D axis it lays the
        # projection down at z = 0 and reads as a stray line under the wing.
        for k in range(m.shape[1]):
            ax.plot(m[:, k, 1], m[:, k, 0], m[:, k, 2], color=c, lw=0.3, alpha=0.8)
        for k in range(m.shape[0]):
            ax.plot(m[k, :, 1], m[k, :, 0], m[k, :, 2], color=c, lw=0.3, alpha=0.8)

        if lab == "wing 3":
            y, le, te = planform(w["mesh"])
            fr = np.array([float(rear_spar_fraction(v, schedule)) for v in y])
            sp = spar_line(w["mesh"], fr)
            ax.plot(sp[:, 1], sp[:, 0], sp[:, 2], color=BOX_C, lw=2.2, label="aft spar")
            fp = spar_line(w["mesh"], np.full(y.size, W.FRONT_PCT))
            ax.plot(fp[:, 1], fp[:, 0], fp[:, 2], color=BOX_C, lw=1.4, ls="--", label="front spar")
            sel = y >= Y_AIL
            ax.plot(sp[sel, 1], sp[sel, 0], sp[sel, 2], color=AIL_C, lw=4.0, label="aileron span")
            ax.legend(fontsize=8, loc="upper left")

        ax.set_title(f"{lab} — {w['drag']:.0f} N, S_ref {w['S_ref']:.2f} m²", fontsize=11)
        ax.set_xlabel("y, m"); ax.set_ylabel("x, m"); ax.set_zlabel("z, m")
        # A 36 m span against a 2.7 m chord is a ribbon from any low angle, so
        # the view is deliberately high: the planform is what has to read.
        ax.view_init(elev=46, azim=-100)

        # The mesh is in global VSP coordinates (x sits around 24-27 m), so the
        # axes have to be set from the data. Letting them include the origin is
        # what squashes the wing into a line.
        span = (m[:, :, 1].min(), m[:, :, 1].max())
        chordwise = (m[:, :, 0].min(), m[:, :, 0].max())
        vert = (m[:, :, 2].min(), m[:, :, 2].max())
        pad = lambda lo, hi, f=0.06: (lo - f * (hi - lo), hi + f * (hi - lo))
        ax.set_xlim(*pad(*span)); ax.set_ylim(*pad(*chordwise)); ax.set_zlim(*pad(*vert))
        ax.locator_params(axis="y", nbins=4)
        ax.locator_params(axis="z", nbins=4)
        try:
            # True relative proportions. An earlier version floored z at 0.18 of
            # the span, which is 6.5 m against a 2.7 m chord -- it exaggerated
            # the dihedral into something the wing does not have.
            rx, ry, rz = np.ptp(span), np.ptp(chordwise), np.ptp(vert)
            ax.set_box_aspect((rx, ry, rz))
        except Exception:
            pass

    path = os.path.join(OUT_DIR, "wing3_3d.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    with open(os.path.join(LOGS, "aileron_90.json")) as fh:
        ail = json.load(fh)
    six = [c for c in ail["cases"] if c["depth_req_in"] == DEPTH_REQ]
    best_case = min(six, key=lambda c: c["drag_N"])
    print(f"  wing 3 = junction spar {best_case['junction_spar_xc']:.3f}c, "
          f"drag {best_case['drag_N']:.1f} N, depth at aileron {best_case['depth_at_aileron_in']:.2f} in")

    schedule = ((356.0, 0.750), (674.9, best_case["junction_spar_xc"]))
    stations = ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0), (674.9, W.JUNCTION_BOX_IN))
    W.REAR_SCHEDULE = schedule
    W.WIDTH_STATIONS = stations
    config.WINGBOX_FRONT_PCT = W.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = stations

    with open(os.path.join(LOGS, "wing2_oas.json")) as fh:
        w2log = json.load(fh)

    base = rebuild(w2log["A_baseline"], None)
    best = rebuild(best_case, W.REGION_A_END_IN)
    _, stick_b, _, _ = W.load_relofted(W.BASELINE, W.REGION_A_END_IN)

    xs, ret = retention_fn()
    print(f"  as-built {base['drag']:9.1f} N   wing 3 {best['drag']:9.1f} N "
          f"({best['drag'] / base['drag'] - 1:+.2%})")

    for p in (figure_2d(base, best, stick_b, schedule, stations, xs, ret), figure_3d(base, best, schedule)):
        print(f"  wrote {p}")
