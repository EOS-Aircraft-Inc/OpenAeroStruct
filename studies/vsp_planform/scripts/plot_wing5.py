"""Wing 5 vs wing 3: thickness, where the weight went, and what it bought."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

LOGS = Path(__file__).resolve().parent.parent / "out" / "logs"
FIGS = Path(__file__).resolve().parent.parent / "out" / "figures"
Y_CROSS, Y_AIL = 447.0, 637.2
C3, C5 = "#444444", "#c0392b"


def bays(tag):
    p = LOGS / tag / "04.Weights" / "wingWeightSummary.csv"
    rows = [r for r in csv.reader(p.open()) if r and not r[0].startswith("#")]
    hdr = [h.strip() for h in rows[0]]
    iWS, iW = hdr.index("WS"), hdr.index("W_bay")
    out = []
    for r in rows[1:]:
        try:
            out.append((float(r[iWS]), float(r[iW])))
        except (ValueError, IndexError):
            pass
    return np.array(out)


def toc_of(cp):
    from studies.vsp_planform import run_opt, config
    import wing2_oas as w2
    w2.apply_wing2_box()
    mesh, stick, regions, pf0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    prob, _ = run_opt.build_problem(w2.BASELINE, mesh, stick, regions, pf0)
    if cp is not None:
        prob.set_val("wing.t_over_c_cp", np.asarray(cp))
    prob.run_model()
    toc = np.asarray(prob.get_val("wing.t_over_c")).ravel()
    y = np.abs(np.asarray(prob.get_val("wing.mesh", units="m"))[0, :, 1]) / config.SCALE
    yp = 0.5 * (y[:-1] + y[1:])
    o = np.argsort(yp)
    return yp[o], toc[o]


def main():
    d = json.loads((LOGS / "wing5.json").read_text())
    w3, w5 = d["wing3"][-1], d["wing5"][-1]
    y3, t3 = toc_of(None)
    y5, t5 = toc_of(w5["cp"])
    b3, b5 = bays("wc_wing3_p4"), bays("wc_wing5_p4")

    fig = plt.figure(figsize=(14.5, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.32, wspace=0.28)

    # --- t/c profile ---
    a = fig.add_subplot(gs[0, :2])
    a.plot(y3, t3, lw=2.4, color=C3, label=f"wing 3  (root {w3['toc_root']:.3f})")
    a.plot(y5, t5, lw=2.4, color=C5, label=f"wing 5  (root {w5['toc_root']:.3f})")
    a.fill_between(y3, t3, np.interp(y3, y5, t5), where=np.interp(y3, y5, t5) > t3,
                   color=C5, alpha=0.13)
    a.axvline(Y_CROSS, color="#2e8b57", ls=":", lw=1.8)
    a.text(Y_CROSS - 10, 0.225, f"crossover WS {Y_CROSS:.0f} in\n(63% semi)",
           ha="right", va="top", fontsize=9, color="#2e8b57")
    a.axvline(Y_AIL, color="#999", ls="-.", lw=1.2)
    a.text(Y_AIL - 10, 0.105, "aileron\n6 in depth", ha="right", fontsize=8.5, color="#777")
    a.axhline(0.20, color="#bbb", lw=0.9, ls=(0, (1, 3)))
    a.text(6, 0.2035, "0.20 conventional", fontsize=8, color="#aaa")
    a.set_xlabel("span station WS (in)"); a.set_ylabel("t/c")
    a.set_title("Thickness added only inboard of the crossover", fontsize=11)
    a.legend(fontsize=9.5); a.grid(alpha=0.25); a.set_xlim(0, 700)

    # --- headline numbers ---
    a = fig.add_subplot(gs[0, 2]); a.axis("off")
    lines = [
        ("wing weight", f"{w3['w_wing_lb']:,.0f} lb", f"{w5['w_wing_lb']:,.0f} lb",
         f"{w5['w_wing_lb']-w3['w_wing_lb']:+,.0f}"),
        ("battery", f"{w3['m_batt_lb']:,.0f} lb", f"{w5['m_batt_lb']:,.0f} lb",
         f"{w5['m_batt_lb']-w3['m_batt_lb']:+,.0f}"),
        ("drag", f"{w3['drag_N']:,.0f} N", f"{w5['drag_N']:,.0f} N",
         f"{w5['drag_N']-w3['drag_N']:+,.0f}"),
        ("S_ref", f"{w3['S_ref']:.2f} m²", f"{w5['S_ref']:.2f} m²", "±0.00"),
        ("elec range", f"{w3['R_nmi']:.1f} nmi", f"{w5['R_nmi']:.1f} nmi",
         f"{w5['R_nmi']-w3['R_nmi']:+.1f}"),
    ]
    a.text(0.0, 0.98, "wing 5 vs wing 3", fontsize=13, weight="bold", va="top")
    a.text(0.52, 0.88, "wing 3", fontsize=9, color=C3, ha="center", weight="bold")
    a.text(0.78, 0.88, "wing 5", fontsize=9, color=C5, ha="center", weight="bold")
    for i, (k, v3, v5, dd) in enumerate(lines):
        yy = 0.76 - i * 0.145
        a.text(0.0, yy, k, fontsize=10)
        a.text(0.52, yy, v3, fontsize=9.5, ha="center", color=C3)
        a.text(0.78, yy, v5, fontsize=9.5, ha="center", color=C5)
        a.text(1.0, yy, dd, fontsize=9.5, ha="right", weight="bold",
               color="#1a7a3a" if dd.startswith("-") and k in ("wing weight", "drag")
               else ("#1a7a3a" if dd.startswith("+") and k in ("battery", "elec range") else "#999"))
    a.text(0.0, 0.03, f"+2.57% electric range at 300 Wh/kg", fontsize=10.5,
           weight="bold", color="#1a7a3a")

    # --- per-bay weight ---
    a = fig.add_subplot(gs[1, 0:2])
    w = 14
    a.bar(b3[:, 0] - w / 2, b3[:, 1], w, color=C3, label="wing 3")
    a.bar(b5[:, 0] + w / 2, b5[:, 1], w, color=C5, label="wing 5")
    a.axvline(Y_CROSS, color="#2e8b57", ls=":", lw=1.8)
    a.set_xlabel("span station WS (in)"); a.set_ylabel("bay weight (lb)")
    a.set_title("The saving is inboard, where the bending moment is", fontsize=11)
    a.legend(fontsize=9.5); a.grid(alpha=0.25, axis="y")

    # --- cumulative saving ---
    a = fig.add_subplot(gs[1, 2])
    dW = b5[:, 1] - b3[:, 1]
    a.plot(b3[:, 0], np.cumsum(dW), lw=2.4, color=C5)
    a.axvline(Y_CROSS, color="#2e8b57", ls=":", lw=1.8)
    a.axhline(0, color="#bbb", lw=0.8)
    frac = np.cumsum(dW)[np.argmin(np.abs(b3[:, 0] - Y_CROSS))] / dW.sum()
    a.text(Y_CROSS + 12, np.cumsum(dW)[-1] * 0.45,
           f"{100*frac:.0f}% of the saving\nis inboard of here", fontsize=9, color="#2e8b57")
    a.set_xlabel("span station WS (in)"); a.set_ylabel("cumulative Δ bay weight (lb)")
    a.set_title("Cumulative structural saving", fontsize=11)
    a.grid(alpha=0.25)

    fig.suptitle("Wing 5 — wing 3 thickened inboard: 581 lb lighter, same area, +2.6% electric range",
                 fontsize=13.5, y=0.975)
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "wing5.png"
    fig.savefig(out, dpi=155, bbox_inches="tight")
    print("wrote", out)
    print(f"bay weight total: wing3 {b3[:,1].sum():.1f} -> wing5 {b5[:,1].sum():.1f} lb "
          f"({b5[:,1].sum()-b3[:,1].sum():+.1f})")


if __name__ == "__main__":
    main()
