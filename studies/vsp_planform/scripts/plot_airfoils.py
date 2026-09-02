"""Top-ranked sections from DOE v3: shapes and polars, against the as-built.

The v3 screen (spar position scanned rather than pinned at max thickness) put the
as-built section 321st of 2048 feasible, overturning the study's earlier "nothing
beats the as-built" conclusion. This draws what that ranking is actually claiming.

All polars are run at ONE common Reynolds number so the comparison is like for
like. The ranking in doe_v3.py used each section's own junction chord to set Re,
which is the right thing for sizing but makes the curves incomparable, so the
numbers here will differ slightly from the ranking table.

READ THE CAVEATS ON THE FIGURE. These are NeuralFoil predictions, the leaders are
sailplane sections leaning on extensive predicted laminar flow, and the OAS model
used everywhere else in this study cannot see any of this -- it reduces a section
to t/c and c_max_t through the Raymer form factor.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))
sys.path.insert(0, os.path.dirname(_HERE))

import aerosandbox as asb  # noqa: E402

from studies.vsp_planform import config  # noqa: E402
from doe_v3 import asbuilt, fp  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "figures")

# Top of the v3 ranking AT 6 IN DEPTH -- the adopted requirement -- plus the two
# that hold Cl_max closest to the as-built (e694, nlf1015) and the
# highest-confidence one (hq17).
#
# Three high-L/D entries are deliberately absent. stf86361 (rank 9, L/D 229.5)
# has a FIRST-peak Cl_max of 0.992 against a cruise CL of 0.93 -- a 6% margin,
# which is the fx2 failure mode: a section that stalls at the cruise point while
# reporting a flattering post-reattachment maximum. fx67k150 (1.289) and
# fx67k170 (1.303) are thin on margin and poorly predicted (conf 0.59, 0.84).
NAMES = ["fx78k140a20", "fx74130wp2", "fx78k140", "fx78k150", "ah88k130",
         "ah81k144", "e694", "hq17", "nlf1015"]

# The aft spar at the aileron station, from the rear-spar schedule. This is where
# depth is required, so it is where thickness retention is read.
SPAR_XC = 0.5736
DEPTH_REQ_IN = 6.0

AL = np.arange(-6, 20.05, 0.25)
CL_OP = 0.93
CHORD_IN = 60.0  # representative outboard chord for wing 3
RE = config.RE_PER_M * CHORD_IN * 0.0254

COLORS = plt.cm.viridis(np.linspace(0.05, 0.85, len(NAMES)))
AS_C = "#C44E52"


def polar(af):
    a = af.get_aero_from_neuralfoil(alpha=AL, Re=RE, mach=config.MACH, model_size="medium")
    g = lambda n: np.atleast_1d(np.asarray(a[n], float)).ravel()
    cl, cd, cf = g("CL"), g("CD"), g("analysis_confidence")
    i = fp(cl)
    out = {"cl": cl, "cd": cd, "conf": float(cf.min()), "ipk": i,
           "clmax": float(cl[i]), "a_clmax": float(AL[i])}
    if cl[: i + 1].max() >= CL_OP:
        out["cd_op"] = float(np.interp(CL_OP, cl[: i + 1], cd[: i + 1]))
        out["ld_op"] = CL_OP / out["cd_op"]
    else:
        out["cd_op"] = np.nan
        out["ld_op"] = np.nan
    return out


if __name__ == "__main__":
    sections = [("as-built", asbuilt(), AS_C)]
    for n, c in zip(NAMES, COLORS):
        try:
            sections.append((n, asb.Airfoil(n), c))
        except Exception as e:
            print(f"  skipped {n}: {e}")

    data = {}
    for name, af, _ in sections:
        data[name] = polar(af)
        d = data[name]
        print(f"  {name:>14}  L/D@0.93 {d['ld_op']:6.1f}  Cd {d['cd_op']:.5f}  "
              f"Cl_max(1st) {d['clmax']:.3f} @ {d['a_clmax']:+.1f} deg  conf {d['conf']:.3f}")

    fig = plt.figure(figsize=(21, 11))
    fig.suptitle(
        f"DOE v3 top-ranked sections vs the as-built, screened at {DEPTH_REQ_IN:.0f} in aft-spar depth — "
        f"NeuralFoil, Re = {RE:.2e}, M = {config.MACH:.3f}\n"
        "common Re for comparability (the ranking used each section's own junction chord). "
        "Cl_max is ALWAYS the first peak.",
        fontsize=13,
    )
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.26)

    # --- shapes
    ax = fig.add_subplot(gs[0, 0])
    for name, af, c in sections:
        xy = af.coordinates
        lw = 2.2 if name == "as-built" else 1.2
        ax.plot(xy[:, 0], xy[:, 1], color=c, lw=lw, label=name)
    ax.set_aspect("equal")
    ax.set_xlabel("x/c"); ax.set_ylabel("z/c")
    ax.set_title("Section shapes", fontsize=11, pad=8)
    ax.grid(alpha=0.3)
    # The legend sat on top of the title in an equal-aspect axes; the polar and
    # bar panels already name every section, so this one goes underneath.
    ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              frameon=False)

    # --- lift curve
    ax = fig.add_subplot(gs[0, 1])
    for name, af, c in sections:
        d = data[name]
        lw = 2.2 if name == "as-built" else 1.2
        ax.plot(AL, d["cl"], color=c, lw=lw)
        ax.plot([d["a_clmax"]], [d["clmax"]], "o", color=c, ms=5)
    ax.axhline(CL_OP, color="k", ls="--", lw=1.2)
    ax.annotate(f"cruise Cl = {CL_OP}", xy=(AL[0], CL_OP), xytext=(AL[0] + 0.4, CL_OP + 0.08), fontsize=9)
    ax.set_xlabel("alpha, deg"); ax.set_ylabel("Cl")
    ax.set_title("Lift curve — markers are the FIRST peak, not the global max", fontsize=11)
    ax.grid(alpha=0.3)

    # --- drag polar
    ax = fig.add_subplot(gs[1, 0])
    for name, af, c in sections:
        d = data[name]
        i = d["ipk"]
        lw = 2.2 if name == "as-built" else 1.2
        ax.plot(d["cd"][: i + 1], d["cl"][: i + 1], color=c, lw=lw, label=name)
        if np.isfinite(d["cd_op"]):
            ax.plot([d["cd_op"]], [CL_OP], "o", color=c, ms=6)
    ax.axhline(CL_OP, color="k", ls="--", lw=1.0, alpha=0.6)
    ax.set_xlim(0, 0.02)
    ax.set_xlabel("Cd"); ax.set_ylabel("Cl")
    ax.set_title("Drag polar (to first peak) — dots at cruise Cl", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # --- L/D
    ax = fig.add_subplot(gs[1, 1])
    order = sorted(data, key=lambda n: -data[n]["ld_op"])
    vals = [data[n]["ld_op"] for n in order]
    cols = [dict((nm, c) for nm, _, c in sections)[n] for n in order]
    ax.barh(range(len(order)), vals, color=cols)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.invert_yaxis()
    for i, (n, v) in enumerate(zip(order, vals)):
        ax.text(v + 2, i, f"{v:.0f}  (conf {data[n]['conf']:.2f}, Cl_max {data[n]['clmax']:.2f})",
                va="center", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.45)
    ax.set_xlabel(f"L/D at Cl = {CL_OP}")
    ax.set_title("L/D at cruise Cl", fontsize=11)
    ax.grid(alpha=0.3, axis="x")

    fig.text(
        0.5, 0.008,
        "CAVEATS  •  NeuralFoil predictions, not XFoil or test data  •  leaders are sailplane sections whose L/D "
        "relies on extensive predicted laminar flow, which will not hold on a transport wing at this Re  •  "
        "OAS CANNOT SEE THIS: it reduces a section to t/c and c_max_t via the Raymer form factor, so none of these "
        "gains appear in the wing-level drag numbers elsewhere in this study",
        ha="center", fontsize=8.5, style="italic", wrap=True,
    )

    path = os.path.join(OUT_DIR, "airfoil_candidates.png")
    # --- thickness distribution, and what it retains at the spar. This is the
    # structural half of the section choice: depth = t(spar) * chord, so a
    # section that keeps its thickness aft needs less chord for the same depth.
    ax = fig.add_subplot(gs[0, 2])
    xs = np.linspace(0.02, 0.98, 300)
    ret = {}
    for name, af, c in sections:
        t = np.array([float(af.local_thickness(x_over_c=x)) for x in xs])
        lw = 2.2 if name == "as-built" else 1.2
        ax.plot(xs, t, color=c, lw=lw)
        ret[name] = float(np.interp(SPAR_XC, xs, t))
        ax.plot([SPAR_XC], [ret[name]], "o", color=c, ms=5, mec="k", mew=0.5, zorder=5)
    ax.axvline(SPAR_XC, color="0.4", ls="--", lw=1.2)
    ax.annotate(f"aft spar {SPAR_XC:.3f}c", xy=(SPAR_XC, ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", fontsize=8, color="0.3", va="top")
    ax.set_xlabel("x/c"); ax.set_ylabel("t/c")
    ax.set_title("Thickness distribution — dots are t at the spar", fontsize=11)
    ax.grid(alpha=0.3)

    # --- the chord each section needs to deliver the required depth there.
    # c_req = depth / t(spar): the number that sets area, and therefore weight.
    ax = fig.add_subplot(gs[1, 2])
    names = [n for n, _, _ in sections]
    creq = [DEPTH_REQ_IN / ret[n] if ret[n] > 1e-6 else np.nan for n in names]
    order = np.argsort(creq)
    cols = [c for _, _, c in sections]
    ypos = np.arange(len(names))
    ax.barh(ypos, [creq[i] for i in order], color=[cols[i] for i in order])
    ax.set_yticks(ypos); ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.invert_yaxis()
    ab = DEPTH_REQ_IN / ret["as-built"]
    ax.axvline(ab, color=AS_C, ls="--", lw=1.2)
    for k, i in enumerate(order):
        ax.text(creq[i] + 0.4, k, f"{creq[i]:.1f} in ({creq[i]-ab:+.1f})", va="center", fontsize=8)
    ax.set_xlabel(f"chord required for {DEPTH_REQ_IN:.0f} in of depth at {SPAR_XC:.3f}c, in")
    ax.set_title("Chord the section forces on the wing\n(less is better: it sets S_ref, and S_ref sets weight)",
                 fontsize=11)
    ax.set_xlim(0, max(np.nanmax(creq) * 1.25, ab * 1.25))
    ax.grid(alpha=0.3, axis="x")

    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {path}")
