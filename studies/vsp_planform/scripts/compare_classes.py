"""The planform trade reduced to three constraint classes, best drag in each.

The full comparison sheet ranks seven wings that differ in several ways at once.
This one asks a narrower question: given a class of constraint on the planform,
what is the least drag available inside it?

  free                 no constraint on chord shape beyond the parameterization:
                       region A|B re-lofted to 176 in, taper_B and the straight
                       line's chord fraction both free. The aft spar kinks
                       0.750 -> 0.550. This is wing 3.
  straight front spar   the straight line pinned at the front spar's 0.12c, so
                       the leading edge is x_spar_fwd - 0.12c and the aft spar is
                       free to curve. Region A runs under the `preserved` rule,
                       since under `root_le_fixed` the LE is frozen by
                       construction and the question cannot be posed. Wing 7.
  constant chord        the ConstChord loft's own A|B breakpoint kept at 361.7 in,
                       so the constant bay survives 51% of the semi-span instead
                       of 25%. Wing 8, as-built t/c.

All three are at MTOW 382,547 N, span pinned at 118 ft, trimmed to the same lift,
under the same box-width stations, 6 in aileron depth at 90% semi-span and rear
spar schedule. Each is replayed from its stored design vector rather than
re-optimized, and the replayed drag is checked against the logged value.

DRAG IS NOT THE MERIT FUNCTION. The study ranks designs on electric range at
fixed MTOW, m_batt/D, where wing weight trades against battery. Break-even is
1.486 lb of wing weight per newton of drag, so the drag spread here converts into
the weight each class must save to be worth having; the figure says so rather
than leaving a drag ranking to be read as a verdict. Those weights need WingCalc
and are not in this figure.

Writes out/figures/class_comparison.png.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

from studies.vsp_planform import config, param                 # noqa: E402
from studies.vsp_planform.run_opt import POINT                 # noqa: E402
import wing2_oas as w2                                         # noqa: E402
from wing5_mtow import stations_and_schedule                   # noqa: E402
from wing8_constchord_toc import REGION_A_AS_BUILT_IN          # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")
FIGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "figures")
q = 0.5 * config.RHO * config.V_MS**2

# Electric range at fixed MTOW: break-even wing weight per newton of drag.
# m_batt at wing 3's 8613.9 lb, over wing 3's drag. See coupling/mission.py.
BREAK_EVEN_LB_PER_N = 15551.7 / 10463.9341

CLASSES = [
    ("free\n(kinking aft spar)", "#C44E52", "wing 3"),
    ("straight\nfront spar", "#4C72B0", "wing 7"),
    ("constant\nchord", "#DD8452", "wing 8"),
]


def replay(case, y_a_in, rule):
    """Rebuild a stored case and confirm it reproduces its logged drag."""
    schedule, stations, _ = stations_and_schedule()
    w2.REAR_SCHEDULE, w2.WIDTH_STATIONS = schedule, stations
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = stations

    saved = param.REGION_A_RULE[w2.BASELINE]
    param.REGION_A_RULE[w2.BASELINE] = rule
    try:
        prob, _, _, _, _ = w2.build(w2.BASELINE, y_a_in)
        prob.set_val("wing.taper_B", case["taper_B"])
        prob.set_val("wing.wingbox_pct", case["wingbox_pct"])
        prob.set_val("wing.twist_cp", np.array(case["twist_cp"]), units="deg")
        prob.set_val("alpha", case["alpha"], units="deg")
        if case.get("t_over_c_cp") is not None:
            prob.set_val("wing.t_over_c_cp", np.array(case["t_over_c_cp"]))
        prob.run_model()
    finally:
        param.REGION_A_RULE[w2.BASELINE] = saved

    s_ref = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    cd = lambda k: float(prob.get_val(f"{POINT}.wing_perf.{k}")[0])
    drag = q * s_ref * (cd("CDi") + cd("CDv") + cd("CDw"))
    if abs(drag - case["drag_N"]) > 2.0:
        raise RuntimeError(f"replayed {drag:.1f} N != logged {case['drag_N']:.1f} N")
    return {"S_ref": s_ref, "drag_N": drag,
            "induced_N": q * s_ref * cd("CDi"), "viscous_N": q * s_ref * cd("CDv"),
            "mesh": np.asarray(prob.get_val("wing.mesh", units="m"))}


if __name__ == "__main__":
    w7log = json.load(open(os.path.join(LOGS, "wing7_design_point.json")))
    w8log = json.load(open(os.path.join(LOGS, "wing8_design_point.json")))

    print("  replaying free (wing 3) ...")
    r_free = replay(w7log["wing3_mtow"], w2.REGION_A_END_IN, "root_le_fixed")
    print("  replaying straight front spar (wing 7) ...")
    r_fwd = replay(w7log["wing7_mtow"], w2.REGION_A_END_IN, "preserved")
    print("  replaying constant chord (wing 8) ...")
    r_cc = replay(w8log["constchord_asbuilt"], REGION_A_AS_BUILT_IN, "root_le_fixed")
    res = [r_free, r_fwd, r_cc]
    ref = r_free["drag_N"]

    fig = plt.figure(figsize=(15.5, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.32, wspace=0.28)
    names = [c[0] for c in CLASSES]
    cols = [c[1] for c in CLASSES]
    xs = np.arange(3)

    # --- total drag, with the break-even weight each class must buy back
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(xs, [r["drag_N"] for r in res], color=cols, width=0.62)
    for i, r in enumerate(res):
        d = r["drag_N"] - ref
        lbl = f"{r['drag_N']:.0f}\n{100*(r['drag_N']/ref-1):+.2f}%"
        if i:
            lbl += f"\nmust save\n{BREAK_EVEN_LB_PER_N*d:.0f} lb"
        ax.text(i, r["drag_N"] + 6, lbl, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("drag, N"); ax.set_title("Total drag at MTOW")
    ax.set_ylim(min(r["drag_N"] for r in res) - 60, max(r["drag_N"] for r in res) + 110)
    ax.grid(alpha=0.25, axis="y")

    # --- where the penalty is spent
    ax = fig.add_subplot(gs[0, 1])
    ind = [r["induced_N"] for r in res]; vis = [r["viscous_N"] for r in res]
    ax.bar(xs, ind, color="#4C72B0", width=0.62, label="induced")
    ax.bar(xs, vis, bottom=ind, color="#DD8452", width=0.62, label="viscous")
    for i in range(3):
        ax.text(i, ind[i] / 2, f"{ind[i]:.0f}", ha="center", va="center", color="w", fontsize=9)
        ax.text(i, ind[i] + vis[i] / 2, f"{vis[i]:.0f}", ha="center", va="center", color="w", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("drag, N"); ax.set_title("Induced vs viscous (wave = 0)")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.25, axis="y")

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(xs, [r["S_ref"] for r in res], color=cols, width=0.62)
    for i, r in enumerate(res):
        ax.text(i, r["S_ref"] + 0.05, f"{r['S_ref']:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("S_ref, m²"); ax.set_title("Wing area")
    ax.set_ylim(min(r["S_ref"] for r in res) - 1.2, max(r["S_ref"] for r in res) + 1.2)
    ax.grid(alpha=0.25, axis="y")

    # --- planforms. x down, as in the study's other planform panels.
    ax = fig.add_subplot(gs[1, :2])
    for (nm, c, tag), r in zip(CLASSES, res):
        m = r["mesh"] / config.SCALE
        y = np.abs(m[0, :, 1])
        ax.plot(y, m[0, :, 0], color=c, lw=1.6, label=f"{nm.replace(chr(10), ' ')} — {tag}")
        ax.plot(y, m[-1, :, 0], color=c, lw=1.6)
    ax.axvline(0.90 * 708.0, color="#8172B2", ls="--", lw=1.1)
    ax.text(0.90 * 708.0, ax.get_ylim()[0], " aileron", color="#8172B2", fontsize=8, va="bottom")
    ax.invert_yaxis(); ax.set_xlabel("y, in"); ax.set_ylabel("x, in")
    ax.set_title("Planforms — leading and trailing edges")
    ax.legend(fontsize=8.5, loc="center left"); ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[1, 2])
    for (nm, c, tag), r in zip(CLASSES, res):
        m = r["mesh"] / config.SCALE
        y = np.abs(m[0, :, 1])
        ax.plot(y, m[-1, :, 0] - m[0, :, 0], color=c, lw=1.6)
    ax.set_xlabel("y, in"); ax.set_ylabel("chord, in")
    ax.set_title("Chord distribution"); ax.grid(alpha=0.25)

    fig.suptitle("Best drag available inside each planform constraint class — full OAS at MTOW 382 547 N, "
                 "span pinned at 118 ft, all trimmed to the same lift", fontsize=12)
    fig.text(0.5, 0.012,
             "Drag is NOT the merit function: the study ranks on electric range at fixed MTOW (m_batt/D), where break-even is "
             f"{BREAK_EVEN_LB_PER_N:.3f} lb of wing weight per newton.\nThe 'must save' figures are what each class has to buy back "
             "in structure to be worth having. Those weights need WingCalc and are not in this figure. Wing-only drag throughout.",
             ha="center", fontsize=8.5, style="italic")

    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, "class_comparison.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\n  wrote {path}")

    print("\n" + "=" * 84)
    print(f"{'class':26} {'drag N':>10} {'vs free':>9} {'S_ref':>8} {'induced':>9} {'viscous':>9} {'must save lb':>13}")
    for (nm, _, tag), r in zip(CLASSES, res):
        d = r["drag_N"] - ref
        print(f"{nm.replace(chr(10), ' ') + ' [' + tag + ']':26} {r['drag_N']:>10.1f} "
              f"{100*(r['drag_N']/ref-1):>+8.2f}% {r['S_ref']:>8.3f} {r['induced_N']:>9.1f} "
              f"{r['viscous_N']:>9.1f} {BREAK_EVEN_LB_PER_N*d:>13.1f}")
    print("=" * 84)
