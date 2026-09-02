"""t/c profile: as-built vs the swept optimum vs the crossover-informed proposal."""

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
SEMI_IN = 708.0
Y_CROSS = 447.0        # spanwise break-even from the per-bay decomposition
Y_AIL = 637.2


def toc_for(cp_root, ratio, n_cp=5):
    """Spanwise t/c the model actually produces for a linear cp ramp."""
    from studies.vsp_planform import run_opt, config
    import wing2_oas as w2
    w2.apply_wing2_box()
    mesh, stick, regions, pf0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    prob, _ = run_opt.build_problem(w2.BASELINE, mesh, stick, regions, pf0)
    if cp_root is not None:
        prob.set_val("wing.t_over_c_cp", np.linspace(cp_root, cp_root * ratio, n_cp))
    prob.run_model()
    toc = np.asarray(prob.get_val("wing.t_over_c")).ravel()
    y = np.abs(np.asarray(prob.get_val("wing.mesh", units="m"))[0, :, 1]) / config.SCALE
    yp = 0.5 * (y[:-1] + y[1:])
    o = np.argsort(yp)
    return yp[o], toc[o]


def main():
    y_ab, toc_ab = toc_for(None, None)              # as-built loft
    y_op, toc_op = toc_for(0.220, 0.58)             # swept optimum at the 22% cap

    # Crossover-informed proposal: thicken inboard of the break-even station only,
    # blend back to as-built by Y_CROSS, leave the outer wing alone.
    blend = np.clip((Y_CROSS - y_ab) / Y_CROSS, 0.0, 1.0)
    toc_pr = toc_ab + blend * (np.interp(y_ab, y_op, toc_op) - toc_ab)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

    a = ax[0]
    a.plot(y_ab, toc_ab, lw=2.2, color="#444", label="as-built (root 0.177)")
    a.plot(y_op, toc_op, lw=2.2, color="#c0392b", label="swept optimum (root 0.220)")
    a.plot(y_ab, toc_pr, lw=2.2, ls="--", color="#1f77b4",
           label="proposal: thicken inboard of crossover")
    a.axvline(Y_CROSS, color="#2e8b57", ls=":", lw=1.8)
    a.text(Y_CROSS - 8, 0.235, f"crossover\nWS {Y_CROSS:.0f} in ({100*Y_CROSS/SEMI_IN:.0f}%)",
           ha="right", va="top", fontsize=9, color="#2e8b57")
    a.axvline(Y_AIL, color="#888", ls="-.", lw=1.3)
    a.text(Y_AIL - 8, 0.10, "aileron\n6 in depth", ha="right", fontsize=9, color="#666")
    a.axhline(0.20, color="#999", lw=0.8, ls=(0, (1, 3)))
    a.text(5, 0.203, "0.20 conventional limit", fontsize=8, color="#999")
    a.set_xlabel("span station WS (in)")
    a.set_ylabel("t/c")
    a.set_title("Thickness distribution")
    a.legend(fontsize=9, loc="upper right")
    a.grid(alpha=0.25)
    a.set_xlim(0, 700)

    b = ax[1]
    rows = json.loads((LOGS / "coupled_toc_profile.json").read_text())
    root = [r["root_toc"] for r in rows]
    rng = [r["R_nmi"] for r in rows]
    wgt = [r["w_wing_lb"] for r in rows]
    b.plot(root, rng, "o-", lw=2.2, color="#c0392b", label="electric range")
    b.set_xlabel("root t/c")
    b.set_ylabel("electric cruise range (nmi)", color="#c0392b")
    b.tick_params(axis="y", labelcolor="#c0392b")
    b.axvline(0.22, color="#1f77b4", ls="--", lw=1.5)
    b.text(0.2205, min(rng) + 0.5, "0.22 cap", fontsize=9, color="#1f77b4")
    b.grid(alpha=0.25)
    b2 = b.twinx()
    b2.plot(root, wgt, "s--", lw=1.8, color="#555", label="wing weight")
    b2.set_ylabel("wing weight (lb)", color="#555")
    b2.tick_params(axis="y", labelcolor="#555")
    b.set_title("Range plateaus at root t/c ~ 0.23")
    lines = b.get_lines()[:1] + b2.get_lines()[:1]
    b.legend(lines, [l.get_label() for l in lines], fontsize=9, loc="lower right")

    fig.suptitle("Wing 3: t/c is worth ~800 lb of structure, and only inboard of 63% semi-span",
                 fontsize=12)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "toc_profile_vs_asbuilt.png"
    fig.savefig(out, dpi=160)
    print("wrote", out)

    print(f"\n{'WS in':>8} {'as-built':>9} {'optimum':>9} {'proposal':>9}")
    for yq in (0, 100, 176, 356, 447, 524, 637):
        print(f"{yq:>8.0f} {np.interp(yq, y_ab, toc_ab):>9.4f} "
              f"{np.interp(yq, y_op, toc_op):>9.4f} {np.interp(yq, y_ab, toc_pr):>9.4f}")


if __name__ == "__main__":
    main()
