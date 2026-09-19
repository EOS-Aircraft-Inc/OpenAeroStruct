"""Requested t/c against what OpenAeroStruct actually delivers, for each arc.

The profile is asked for as a LINEAR ramp on 5 spline control points, root to
root*ratio (arc_optimal_toc.PROFILES). What reaches the drag model is whatever
``om.SplineComp(method="bsplines", order=4)`` makes of those control points,
sampled at the true mid-panel stations. The two are not the same curve, and the
difference is not cosmetic: depth is ``retention * t/c * chord``, so the delivered
t/c at the aileron is what sets the required chord, the area and the weight.

Nothing is optimized here -- each arc is built and run once, which takes a minute.

    python studies/vsp_planform/scripts/plot_toc_request.py
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "..", "..")))

import arc_optimal_toc as A  # noqa: E402

FIGS = os.path.normpath(os.path.join(_HERE, "..", "out", "figures"))
PROFILE = "optimal"
AIRFOIL = "e694"


def probe(arc):
    """Build the arc, apply its t/c profile, run once, take the distribution."""
    y_a, rule, pin_p, sched = A.ARCHS[arc]
    fix_pct = A.STRAIGHT_AFT_PCT.get(arc)
    if fix_pct is not None:
        sched = ((356.0, fix_pct), (674.9, fix_pct))
    blend = A.SECTION_BLEND.get(arc)
    if blend is None:
        A.RET_AT = A.CMT_AT = None
    else:
        A.RET_AT, A.CMT_AT, _ = A.blended_section(*blend)
    return A.optimize(y_a, rule, pin_p, A.PROFILES[PROFILE], 50.0, sched,
                      fix_pct=fix_pct, probe=True)


def main():
    A.RET, A.C_MAX_T = A.section(AIRFOIL)
    cp0, ratio = A.PROFILES[PROFILE]
    os.makedirs(FIGS, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    rows = []
    for ax, arc in zip(axes, ("A", "B", "C")):
        pr = probe(arc)
        y = np.asarray(pr["y_panel_in"])
        toc = np.asarray(pr["toc_panel"])
        cp = np.asarray(pr["toc_cp"])
        node = np.asarray(pr["y_node_in"])

        # the request: linear in NORMALISED SPAN, which is what the spline's own
        # abscissa is (openaerostruct/utils/interpolation.py). Control points sit
        # at equal spacing on that same coordinate.
        s = (y - node[0]) / (node[-1] - node[0])
        want = cp0 + (cp0 * ratio - cp0) * s
        s_cp = np.linspace(0.0, 1.0, len(cp))
        y_cp = node[0] + s_cp * (node[-1] - node[0])

        err = toc - want
        rows.append((arc, want[0], toc[0], want[-1], toc[-1],
                     float(np.max(np.abs(err))), float(100 * np.max(np.abs(err / want)))))

        ax.plot(y, want, "--", color="#666", lw=1.8, label="requested: linear ramp")
        ax.plot(y, toc, "o-", color="#33527a", ms=3.2, lw=1.8,
                label="delivered by the SplineComp")
        ax.plot(y_cp, cp, "s", color="#a8331f", ms=8, mfc="none", mew=2,
                label="the 5 control points sent")
        ax.axvline(A.Y_AIL, color="#a8331f", ls="--", lw=1.0)
        ax.annotate("aileron", xy=(A.Y_AIL, 0.98), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=8.5, color="#a8331f")
        ax.set_ylabel("t/c")
        ax.set_title("Arc {}  —  max error {:.4f} t/c ({:.1f}% of the request)".format(
            arc, rows[-1][5], rows[-1][6]), fontsize=10.5)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8.5, loc="upper right")

    axes[-1].set_xlabel("spanwise station  y  [in]")
    fig.suptitle("t/c requested vs delivered — a linear ramp on the control points "
                 "is not a linear distribution", fontsize=12)
    out = os.path.join(FIGS, "toc_request_vs_delivered.png")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print("\n{:>4} {:>10} {:>10} {:>10} {:>10} {:>11} {:>8}".format(
        "arc", "root req", "root got", "tip req", "tip got", "max |err|", "max %"))
    for r in rows:
        print("{:>4} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>11.4f} {:>8.1f}".format(*r))
    print("\nwrote " + out)


if __name__ == "__main__":
    main()
