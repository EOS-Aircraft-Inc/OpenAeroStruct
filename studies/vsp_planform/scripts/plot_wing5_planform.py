"""Wing 5 planform rendering -- the wing 3 2D/3D set, redrawn for wing 5.

The reference here is the ConstChord as-built wing, not wing 3. Wing 5 IS wing
3's planform (same taper_B, wingbox_pct and twist, S_ref to five decimals), so
drawing wing 3 puts a second line exactly under wing 5 and says nothing; against
the as-built reference the figure shows what the planform actually changed. The
wing 5 vs wing 3 delta -- which is a t/c and weight story, not a planform one --
lives in ``wing5.png``.

Wing 5 is rebuilt from the stored design vector in
``out/logs/wing5_design_point.json`` and the as-built reference from
``out/logs/wing2_oas.json``; each rebuilt drag is checked against the logged
value before anything is drawn.
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
import coupled_loop as cl  # noqa: E402
from plot_wing2 import planform, spar_line, mirror  # noqa: E402
from doe_v3 import asbuilt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "figures")
LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")

Y_AIL = cl.Y_AIL
Y_CROSS = 447.0
DEPTH_REQ = cl.DEPTH_REQ_IN
SCHEDULE = ((356.0, 0.750), (674.9, cl.JUNCTION_SPAR))

BASE_C, C5 = "#4C72B0", "#c0392b"
BOX_C = "#55A868"
AIL_C = "#8172B2"
CROSS_C = "#2e8b57"


def retention_fn():
    """Thickness retention t(x/c)/t_max of the as-built section."""
    af = asbuilt()
    xs = np.linspace(0.05, 0.95, 300)
    t = np.array([float(af.local_thickness(x_over_c=x)) for x in xs])
    return xs, t / t.max()


def rebuild(case, y_a_new_in):
    """Rebuild one logged design (planform + t/c) and verify its drag."""
    prob, mesh0, stick, regions, planform0 = W.build(W.BASELINE, y_a_new_in)
    prob.set_val("wing.taper_B", case["taper_B"])
    prob.set_val("wing.wingbox_pct", case["wingbox_pct"])
    prob.set_val("wing.twist_cp", np.array(case["twist_cp"]), units="deg")
    if "t_over_c_cp" in case:  # the as-built reference keeps the loft's own t/c
        prob.set_val("wing.t_over_c_cp", np.array(case["t_over_c_cp"]))
    prob.set_val("alpha", case["alpha"], units="deg")
    prob.run_model()

    drag = float(prob.get_val("drag")[0])
    if abs(drag - case["drag_N"]) > 1.0:
        raise RuntimeError(f"rebuilt drag {drag:.1f} N != logged {case['drag_N']:.1f} N")

    mesh = prob.get_val("wing.mesh", units="m")
    y = np.abs(mesh[0, :, 1]) / config.SCALE
    yp = 0.5 * (y[:-1] + y[1:])
    toc = np.asarray(prob.get_val("wing.t_over_c")).ravel()
    o = np.argsort(yp)

    return {
        "mesh": mesh,
        "twist": prob.get_val("twist_abs", units="deg").copy(),
        "S_ref": float(prob.get_val(f"{POINT}.wing.S_ref")[0]),
        "drag": drag,
        "y_toc": yp[o],
        "toc": toc[o],
        "regions": regions,
        "log": case,
    }


def asbuilt_toc(stick):
    """The as-built spanwise t/c: the loft's own, on the loft's own stations."""
    ys = np.abs(np.asarray(stick.le[:, 1], dtype=float))
    o = np.argsort(ys)
    return ys[o], np.asarray(stick.toc, dtype=float)[o]


def ref_spline_toc():
    """The OAS t/c spline with its default control points, i.e. as-built.

    Wing 5's t/c is a multiplier on this, so it is what the design's spline has
    to be divided by to recover the factor applied to the physical loft.
    """
    prob, _, _, _, _ = W.build(W.BASELINE, W.REGION_A_END_IN)
    prob.run_model()
    y = np.abs(np.asarray(prob.get_val("wing.mesh", units="m"))[0, :, 1]) / config.SCALE
    yp = 0.5 * (y[:-1] + y[1:])
    o = np.argsort(yp)
    return yp[o], np.asarray(prob.get_val("wing.t_over_c")).ravel()[o]


def box_edges(mesh):
    y, le, te = planform(mesh)
    chord = te - le
    rear = np.array([float(rear_spar_fraction(v, SCHEDULE)) for v in y])
    return y, le + W.FRONT_PCT * chord, le + rear * chord


def depth_along_span(w, xs, ret, stick, ref, factor=True):
    """Structural depth at the aft spar, inches: retention * t/c * chord.

    t/c is the PHYSICAL loft thickness scaled by the factor the design applied to
    the OAS spline -- not the spline value itself. The spline smooths the loft and
    outboard reads ~0.125 where the section is really 0.132, which is enough to
    make wing 5's aileron station look 0.30 in short of a requirement it meets.
    """
    y, le, te = planform(w["mesh"])
    chord = te - le
    y_s, toc_s = stick
    toc = np.interp(y, y_s, toc_s)
    if factor:
        y_r, toc_r = ref
        toc = toc * np.interp(y, w["y_toc"], w["toc"]) / np.interp(y, y_r, toc_r)
    spar = np.array([float(rear_spar_fraction(v, SCHEDULE)) for v in y])
    return y, np.interp(spar, xs, ret) * toc * chord


def figure_2d(base, w5, xs, ret, stick, ref, stick_base):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "WING 5 — planform against the ConstChord as-built reference\n"
        f"drag {base['drag']:.0f} → {w5['drag']:.0f} N ({w5['drag'] / base['drag'] - 1:+.2%}) at MTOW, "
        f"S_ref {base['S_ref']:.2f} → {w5['S_ref']:.2f} m², span pinned at 118 ft",
        fontsize=13,
    )
    gs = fig.add_gridspec(3, 2, hspace=0.34, wspace=0.22)

    # --- planform, box, aileron span
    ax = fig.add_subplot(gs[0, :])
    y, le, te = planform(base["mesh"])
    ax.plot(y, le, color=BASE_C, lw=1.3, ls="--", label="ConstChord as-built")
    ax.plot(y, te, color=BASE_C, lw=1.3, ls="--")
    y, le, te = planform(w5["mesh"])
    ax.plot(y, le, color=C5, lw=1.6, label="wing 5")
    ax.plot(y, te, color=C5, lw=1.6)

    y, front, rear = box_edges(w5["mesh"])
    ax.fill_between(y, front, rear, color=BOX_C, alpha=0.30, lw=0, label="structural box")
    ax.plot(y, front, color=BOX_C, lw=1.0)
    ax.plot(y, rear, color=BOX_C, lw=1.0)

    ax.axvline(Y_CROSS, color=CROSS_C, lw=1.8, ls=":")
    ax.axvspan(Y_AIL, y.max(), color=AIL_C, alpha=0.13)
    ax.axvline(Y_AIL, color=AIL_C, lw=1.8, ls="--")
    # Room below the trailing edge for the callouts. The axis is inverted and
    # equal-aspect, so writing them into the outline is the alternative -- an
    # earlier version had the crossover label struck through by the leading edge.
    x_lo, x_hi = ax.get_ylim()
    ax.set_ylim(x_lo - 8.0, x_hi + 42.0)
    y_txt = x_hi + 16.0
    ax.annotate(f"t/c blends back to as-built\nby y = {Y_CROSS:.0f} in (63% semi)",
                xy=(Y_CROSS, y_txt), color=CROSS_C, fontsize=9,
                fontweight="bold", ha="center", va="top")
    ax.annotate(f"aileron inboard end\ny = {Y_AIL:.0f} in (90% semi)",
                xy=(Y_AIL, y_txt), color=AIL_C, fontsize=9,
                fontweight="bold", ha="center", va="top")
    ax.invert_yaxis()
    ax.set_xlabel("y, in"); ax.set_ylabel("x, in")
    ax.set_title("Planform, structural box, thickening region and aileron span", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")

    # --- t/c: the one thing that changed
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ref[0], ref[1], color=BASE_C, lw=1.8, ls="--", label="as-built t/c")
    ax.plot(w5["y_toc"], w5["toc"], color=C5, lw=2.2,
            label=f"wing 5 (root {w5['log']['toc_root']:.3f})")
    t5_on_r = np.interp(ref[0], w5["y_toc"], w5["toc"])
    ax.fill_between(ref[0], ref[1], t5_on_r, where=t5_on_r > ref[1], color=C5, alpha=0.15)
    ax.axvline(Y_CROSS, color=CROSS_C, lw=1.5, ls=":")
    ax.axvline(Y_AIL, color=AIL_C, lw=1.5, ls="--")
    ax.set_xlabel("y, in"); ax.set_ylabel("t/c")
    ax.set_title("Thickness — added inboard, untouched outboard", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # --- aft-spar depth, both wings, against the requirement
    ax = fig.add_subplot(gs[1, 1])
    yb, db = depth_along_span(base, xs, ret, stick_base, ref, factor=False)
    y5, d5 = depth_along_span(w5, xs, ret, stick, ref)
    ax.plot(yb, db, color=BASE_C, lw=1.8, ls="--", label="as-built depth at aft spar")
    ax.plot(y5, d5, color=C5, lw=2.2, label="wing 5 depth at aft spar")
    ax.axhline(DEPTH_REQ, color="k", lw=1.3, ls="--", label=f"{DEPTH_REQ:.0f} in requirement")
    ax.axvline(Y_CROSS, color=CROSS_C, lw=1.5, ls=":")
    ax.axvspan(Y_AIL, y5.max(), color=AIL_C, alpha=0.13)
    ax.axvline(Y_AIL, color=AIL_C, lw=1.5, ls="--")
    a5 = float(np.interp(Y_AIL, y5, d5))
    ax.plot([Y_AIL], [a5], "o", color=AIL_C, ms=8, zorder=5)
    ax.annotate(f"{a5:.2f} in at the aileron\n({a5 - DEPTH_REQ:+.2f} in against\nthe 6 in requirement)",
                xy=(Y_AIL, a5), xytext=(Y_AIL - 330, a5 - 2.6),
                color=AIL_C, fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=AIL_C, lw=1.2))
    ax.set_xlabel("y, in"); ax.set_ylabel("spar depth, in")
    ax.set_title("Aft-spar depth — the extra depth is all inboard", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper right")

    # --- chord and box width, showing the planform really is unchanged
    ax = fig.add_subplot(gs[2, 0])
    for w, c, lab, ls in ((base, BASE_C, "as-built", "--"), (w5, C5, "wing 5", "-")):
        y, le, te = planform(w["mesh"])
        ax.plot(y, te - le, color=c, lw=1.8, ls=ls, label=f"{lab} chord")
    y, front, rear = box_edges(w5["mesh"])
    ax.plot(y, rear - front, color=BOX_C, lw=1.8, label="box width")
    for y_in, req in ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0), (674.9, W.JUNCTION_BOX_IN)):
        ax.plot([y_in], [req], "kv", ms=7)
    ax.plot([], [], "kv", ms=7, label="box width required")
    ax.axvline(Y_AIL, color=AIL_C, lw=1.5, ls="--")
    ax.set_xlabel("y, in"); ax.set_ylabel("chord / width, in")
    ax.set_title("Chord and box width against the as-built chord", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # --- twist
    ax = fig.add_subplot(gs[2, 1])
    for w, c, lab, ls in ((base, BASE_C, "as-built", "--"), (w5, C5, "wing 5", "-")):
        y, _, _ = planform(w["mesh"])
        ax.plot(y, w["twist"], color=c, lw=1.8, ls=ls, label=lab)
    ax.axvline(Y_AIL, color=AIL_C, lw=1.5, ls="--")
    ax.set_xlabel("y, in"); ax.set_ylabel("twist, deg")
    ax.set_title("Twist distribution", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    path = os.path.join(OUT_DIR, "wing5_planform.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_3d(base, w5):
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("WING 5 — isometric, true scale (dihedral to scale)", fontsize=13)

    for i, (w, c, lab) in enumerate(((base, BASE_C, "ConstChord as-built"), (w5, C5, "wing 5"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        m = mirror(w["mesh"])
        for k in range(m.shape[1]):
            ax.plot(m[:, k, 1], m[:, k, 0], m[:, k, 2], color=c, lw=0.3, alpha=0.8)
        for k in range(m.shape[0]):
            ax.plot(m[k, :, 1], m[k, :, 0], m[k, :, 2], color=c, lw=0.3, alpha=0.8)

        y, _, _ = planform(w["mesh"])
        fr = np.array([float(rear_spar_fraction(v, SCHEDULE)) for v in y])
        sp = spar_line(w["mesh"], fr)
        ax.plot(sp[:, 1], sp[:, 0], sp[:, 2], color=BOX_C, lw=2.2, label="aft spar")
        fp = spar_line(w["mesh"], np.full(y.size, W.FRONT_PCT))
        ax.plot(fp[:, 1], fp[:, 0], fp[:, 2], color=BOX_C, lw=1.4, ls="--", label="front spar")
        sel = y >= Y_AIL
        ax.plot(sp[sel, 1], sp[sel, 0], sp[sel, 2], color=AIL_C, lw=4.0, label="aileron span")
        thick = y <= Y_CROSS
        ax.plot(fp[thick, 1], fp[thick, 0], fp[thick, 2], color=CROSS_C, lw=3.0,
                label="thickened region" if lab == "wing 5" else "as-built t/c")
        ax.legend(fontsize=8, loc="upper left")

        ax.set_title(f"{lab} — {w['drag']:.0f} N, S_ref {w['S_ref']:.2f} m²", fontsize=11)
        ax.set_xlabel("y, m"); ax.set_ylabel("x, m"); ax.set_zlabel("z, m")
        ax.view_init(elev=46, azim=-100)

        span = (m[:, :, 1].min(), m[:, :, 1].max())
        chordwise = (m[:, :, 0].min(), m[:, :, 0].max())
        vert = (m[:, :, 2].min(), m[:, :, 2].max())
        pad = lambda lo, hi, f=0.06: (lo - f * (hi - lo), hi + f * (hi - lo))
        ax.set_xlim(*pad(*span)); ax.set_ylim(*pad(*chordwise)); ax.set_zlim(*pad(*vert))
        ax.locator_params(axis="y", nbins=4)
        ax.locator_params(axis="z", nbins=4)
        try:
            ax.set_box_aspect((np.ptp(span), np.ptp(chordwise), np.ptp(vert)))
        except Exception:
            pass

    path = os.path.join(OUT_DIR, "wing5_3d.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    W.apply_wing2_box()
    stations = ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0), (674.9, W.JUNCTION_BOX_IN))
    W.REAR_SCHEDULE = SCHEDULE
    W.WIDTH_STATIONS = stations
    config.WINGBOX_FRONT_PCT = W.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = SCHEDULE
    config.WINGBOX_WIDTH_STATIONS = stations

    with open(os.path.join(LOGS, "wing5_design_point.json")) as fh:
        dp = json.load(fh)

    with open(os.path.join(LOGS, "wing2_oas.json")) as fh:
        w2log = json.load(fh)

    # The as-built reference is drawn without the region-A re-loft, which is a
    # wing 2/3/5 design change and not part of the baseline.
    base = rebuild(w2log["A_baseline"], None)
    w5 = rebuild(dp["wing5_mtow"], W.REGION_A_END_IN)
    xs, ret = retention_fn()
    print(f"  as-built {base['drag']:9.1f} N   wing 5 {w5['drag']:9.1f} N "
          f"({w5['drag'] / base['drag'] - 1:+.2%})")
    print(f"  S_ref    {base['S_ref']:.3f} -> {w5['S_ref']:.3f} m²")

    _, stick5, _, _ = W.load_relofted(W.BASELINE, W.REGION_A_END_IN)
    _, stick_ab = W.half_mesh(config.BASELINES[W.BASELINE])  # no re-loft: the reference
    stick, stick_base, ref = asbuilt_toc(stick5), asbuilt_toc(stick_ab), ref_spline_toc()
    print(f"  depth at the aileron: {np.interp(Y_AIL, *depth_along_span(w5, xs, ret, stick, ref)):.2f} in "
          f"against {DEPTH_REQ:.0f} in required")

    for p in (figure_2d(base, w5, xs, ret, stick, ref, stick_base), figure_3d(base, w5)):
        print(f"  wrote {p}")
