"""Planform and thickness figures for an arc, drawn from the deck it was SIZED on.

Everything here is read back out of ``<deck>/OAS_Inputs`` -- the station metadata,
the Selig contours and the ``*_spar.csv`` companion. Nothing is recomputed from a
design point, so what the figures show is what WingCalc was actually given. That is
the reason they are worth drawing: the export shipped the wrong section for weeks
and every reported number still looked right.

    python studies/vsp_planform/scripts/plot_arc.py \
        --deck studies/vsp_planform/out/logs/deck_arcB_optimal_e694 --name "Arc B"
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "..", "..")))

FIGS = os.path.normpath(os.path.join(_HERE, "..", "out", "figures"))
Y_AIL = 0.90 * 708.0
SEP = "#" * 40


def read_export(deck):
    """Stations, contours and spar fractions exactly as the deck carries them."""
    oas = os.path.join(deck, "OAS_Inputs")
    meta = [f for f in os.listdir(oas)
            if f.endswith(".csv") and "_spar" not in f and "_lift" not in f]
    if not meta:
        raise SystemExit("no station metadata CSV in " + oas)
    text = open(os.path.join(oas, meta[0]), encoding="utf-8-sig").read()

    st = []
    for blk in text.split(SEP):
        f = re.search(r"Airfoil File Name,\s*(\S+)", blk)
        le = re.search(r"Leading Edge Point,\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)", blk)
        te = re.search(r"Trailing Edge Point,\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)", blk)
        c = re.search(r"Chord,\s*([-\d.]+)", blk)
        if f and le and te and c:
            st.append(dict(dat=f.group(1),
                           le=[float(v) for v in le.groups()],
                           te=[float(v) for v in te.groups()],
                           chord=float(c.group(1))))
    st.sort(key=lambda r: r["le"][1])

    # The spar companion, INTERPOLATED onto the stations rather than looked up by
    # key. The two files round y differently -- the metadata writes 6 decimals and
    # the spar file 4 -- so an exact-key match silently dropped every station whose
    # 4th decimal was a 5, and the misses came back as NaN rather than as an error.
    # The spar distribution is piecewise linear in y, so interpolating is exact.
    spar = [f for f in os.listdir(oas) if f.endswith("_spar.csv")]
    sy = sf = sr = None
    if spar:
        rows = [ln for ln in open(os.path.join(oas, spar[0]),
                                  encoding="utf-8-sig").read().splitlines()
                if ln and not ln.startswith("#")]
        head = rows[0].split(",")
        vals = [dict(zip(head, ln.split(","))) for ln in rows[1:]]
        sy = np.array([float(v["y_in"]) for v in vals])
        sf = np.array([float(v["front_pct"]) for v in vals])
        sr = np.array([float(v["rear_pct"]) for v in vals])
        order = np.argsort(sy)
        sy, sf, sr = sy[order], sf[order], sr[order]

    for r in st:
        xy = np.array([[float(a) for a in ln.split()]
                       for ln in open(os.path.join(oas, r["dat"])).read().splitlines()
                       if len(ln.split()) == 2 and not ln.strip()[0].isalpha()])
        x, y = xy[:, 0], xy[:, 1]
        i = int(np.argmin(x))
        g = np.linspace(0.002, 0.998, 400)
        up = np.interp(g, x[:i + 1][::-1], y[:i + 1][::-1])
        lo = np.interp(g, x[i:], y[i:])
        r["xc"], r["t"] = g, up - lo
        r["toc"] = float(r["t"].max())
        r["x_tmax"] = float(g[r["t"].argmax()])
        if sy is None:
            r["front"] = r["rear"] = np.nan
        else:
            r["front"] = float(np.interp(r["le"][1], sy, sf))
            r["rear"] = float(np.interp(r["le"][1], sy, sr))
        # depth is measured ON THE CONTOUR at the spar, not from t/c times a
        # retention factor, so it is the depth the shipped geometry really has
        r["t_at_spar"] = (float(np.interp(r["rear"], g, r["t"]))
                          if sy is not None else np.nan)
        r["depth_in"] = r["t_at_spar"] * r["chord"]
    return st


def region_breaks(st):
    """(A|B, B|C) from the chord distribution itself.

    The break is where the taper line meets the constant root chord, NOT the last
    station that still happens to be flat -- the resampled stations do not land on
    the breakpoint, so picking the last flat one reported 337.5 in for a wing whose
    region A ends at 361.7. The tapered part is a straight line in y, so fitting it
    and solving for c = c_root recovers the breakpoint whatever the station spacing.
    """
    y = np.array([r["le"][1] for r in st])
    c = np.array([r["chord"] for r in st])
    taper = np.flatnonzero(c < 0.98 * c[0])
    if taper.size < 2:
        return float(y[0]), float(y[-1])
    m, b = np.polyfit(y[taper], c[taper], 1)
    return float((c[0] - b) / m), float(y[-1])


def figures(st, name, tag):
    y = np.array([r["le"][1] for r in st])
    xle = np.array([r["le"][0] for r in st])
    xte = np.array([r["te"][0] for r in st])
    ch = np.array([r["chord"] for r in st])
    toc = np.array([r["toc"] for r in st])
    dep = np.array([r["depth_in"] for r in st])
    fwd = np.array([r["front"] for r in st])
    aft = np.array([r["rear"] for r in st])
    y_ab, y_bc = region_breaks(st)
    i_ail = int(np.argmin(np.abs(y - Y_AIL)))
    os.makedirs(FIGS, exist_ok=True)

    # ------------------------------------------------ planform
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.fill(np.r_[y, y[::-1]], np.r_[xle, xte[::-1]],
            color="#dfe7f2", ec="#33527a", lw=1.6, zorder=2)
    if np.isfinite(aft).all():
        x_f, x_a = xle + fwd * ch, xle + aft * ch
        lbl = "wingbox  {:.3f}c - {:.3f}c".format(fwd[0], aft[0])
        if abs(aft[-1] - aft[0]) > 1e-4:
            lbl = "wingbox  {:.3f}c - {:.3f}c kinking to {:.3f}c".format(
                fwd[0], aft[0], aft[-1])
        ax.fill(np.r_[y, y[::-1]], np.r_[x_f, x_a[::-1]], color="#f6d9b0",
                ec="#b4761f", lw=1.2, alpha=.95, zorder=3, label=lbl)
        ax.plot(y, x_a, color="#a8331f", lw=2.4, zorder=4, label="aft spar")
    # Labels go in AXES fractions, not data coordinates: x is inverted here and the
    # chord range differs per arc, so anything placed in data units either collided
    # with the title or fell off the bottom depending on the wing being drawn.
    dat_ax = ("data", "axes fraction")
    for yy, lab, col in [(y_ab, "A|B\n{:.0f}".format(y_ab), "#111"),
                         (Y_AIL, "aileron\n{:.0f}".format(Y_AIL), "#a8331f"),
                         (y_bc, "B|C\n{:.0f}".format(y_bc), "#111")]:
        ax.axvline(yy, color=col, ls="--", lw=1.1, zorder=5)
        ax.annotate(lab, xy=(yy, 0.985), xycoords=dat_ax, ha="center", va="top",
                    fontsize=8.5, color=col, zorder=6)
    j = int(np.argmin(np.abs(y - y_ab)))
    for x0, x1, nm, sub in [
            (y[0], y_ab, "REGION A", "constant chord {:.1f} in".format(ch[0])),
            (y_ab, y_bc, "REGION B",
             "{:.1f} -> {:.1f} in  (tapered)".format(ch[j], ch[-1]))]:
        ax.annotate("", xy=(x0, 0.12), xytext=(x1, 0.12), xycoords=dat_ax,
                    textcoords=dat_ax,
                    arrowprops=dict(arrowstyle="<->", color="#444", lw=1.1))
        ax.annotate(nm + "\n" + sub, xy=((x0 + x1) / 2, 0.105), xycoords=dat_ax,
                    ha="center", va="top", fontsize=8.5, zorder=6)
    ax.set_xlabel("spanwise station  y  [in]")
    ax.set_ylabel("x, aft ->  [in]")
    ax.set_title(name + " planform - regions and wingbox, as exported to WingCalc",
             pad=14)
    ax.invert_yaxis()
    ax.legend(loc="lower left", fontsize=9, framealpha=.95)
    ax.grid(alpha=.25)
    f1 = os.path.join(FIGS, tag + "_regions.png")
    fig.tight_layout()
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    # ------------------------------------------------ t/c and delivered depth
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    a1.plot(y, toc, "o-", color="#33527a", ms=3.5, lw=1.8)
    a1.set_ylabel("t/c")
    a1.set_title(name + " - thickness, and the spar depth it delivers")
    a1.annotate("root {:.4f}".format(toc[0]), xy=(y[0], toc[0]),
                xytext=(14, -12), textcoords="offset points", fontsize=8.5)
    a1.annotate("tip {:.4f}".format(toc[-1]), xy=(y[-1], toc[-1]),
                xytext=(-58, 10), textcoords="offset points", fontsize=8.5)

    a2.plot(y, dep, "o-", color="#a8331f", ms=3.5, lw=1.8,
            label="depth at the aft spar, measured on the contour")
    a2.axhline(7.0, color="#666", ls="--", lw=1.2,
           label="7 in requirement (root to aileron only)")
    a2.axvspan(Y_AIL, y[-1], color="#eee", zorder=0)
    a2.plot([y[i_ail]], [dep[i_ail]], "o", ms=9, mfc="none", mec="#a8331f", mew=2)
    a2.annotate("{:.2f} in at the aileron".format(dep[i_ail]),
                xy=(y[i_ail], dep[i_ail]), xytext=(-165, 24),
                textcoords="offset points", fontsize=9, color="#a8331f",
                arrowprops=dict(arrowstyle="->", color="#a8331f"))
    a2.set_ylabel("aft-spar depth [in]")
    a2.set_xlabel("spanwise station  y  [in]")
    a2.legend(fontsize=9, loc="upper right")
    for ax_ in (a1, a2):
        for yy in (y_ab, Y_AIL, y_bc):
            ax_.axvline(yy, color="#999", ls="--", lw=1.0)
        ax_.grid(alpha=.25)
    f2 = os.path.join(FIGS, tag + "_toc.png")
    fig.tight_layout()
    fig.savefig(f2, dpi=150)
    plt.close(fig)

    print("  regions  A|B {:.1f} in, B|C {:.1f} in, {} stations".format(
        y_ab, y_bc, len(st)))
    print("  spar     {:.4f}c -> {:.4f}c, front {:.3f}c".format(
        aft[0], aft[-1], fwd[0]))
    print("  t/c      {:.4f} root -> {:.4f} tip, max-thickness x/c {:.3f} at the root"
          .format(toc[0], toc[-1], st[0]["x_tmax"]))
    print("  depth    {:.2f} in at the aileron, {:.2f} in minimum over the box".format(
        dep[i_ail], dep.min()))
    print("  wrote " + f1)
    print("  wrote " + f2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True, help="a deck directory holding OAS_Inputs/")
    ap.add_argument("--name", required=True, help="title, e.g. 'Arc B'")
    ap.add_argument("--tag", default=None, help="file stem; default derived from --name")
    a = ap.parse_args()
    figures(read_export(a.deck), a.name,
            a.tag or a.name.lower().replace(" ", ""))
