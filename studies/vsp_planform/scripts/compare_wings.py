"""Plan L vs ConstChord vs wings 2-6 — one comparable set, OAS.

Every case here is run the same way: MTOW 382547 N, span pinned at 118 ft, each
trimmed to the SAME lift, full OAS (CDi + CDv + CDw). That is the only way these
numbers can sit in one table; the simplified-model figures elsewhere in the study
cannot join them.

**Plan L is the reference.** It is the production baseline, so every percentage
on these charts is quoted against it rather than against ConstChord.

The cases:

  Plan L as-built       THE REFERENCE. The other VSP baseline; first run at MTOW
                        in this session.
  ConstChord as-built   the second as-built geometry, and the parent that wings
                        2, 3 and 4 are all derived from
  wing 2                the ADMISSIBLE version: 7 in depth at the winglet
                        junction forces the spar to 0.400c. The widely-quoted
                        -2.52% wing 2 is NOT admissible -- at its 52.77 in
                        junction chord the section's max thickness is 6.25 in,
                        so 7 in is unreachable at any spar station.
  wing 3                ailerons at 90% semi-span, 6 in depth, spar 0.550c
  wing 4                wing 3 plus monotonic (non-increasing outboard) twist
                        through region B -- the manufacturable twist
  wing 5                wing 3's planform exactly (S_ref and wingbox_pct match to
                        five decimals) with t/c raised inboard only, root 0.177 ->
                        0.214, blended back to the as-built loft by WS 447 in.
                        That station is where thickness stops paying: structural
                        benefit falls ~40x root to tip with bending moment, drag
                        cost only ~2.5x with chord. Costs drag, buys 581 lb of
                        structure -- the first case here whose merit is NOT drag,
                        so read its weight column, not its bar.
  wing 6                wing 5 plus the same monotonic-twist constraint wing 4
                        put on wing 3, root to junction on ``twist_abs``. The cost
                        of monotonicity is measured on wing 5's own optimum
                        (+56.7 N, +0.536%) rather than carried over from wing 4's
                        +59.3 N on wing 3. Its root twist sits on the +5 deg
                        bound, so that cost is an upper bound.

Produces `wing_comparison.png`: drag and its breakdown, wing weight, area,
planforms, chord, twist, and the front/aft spar chord ratios along the span.
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

from pathlib import Path  # noqa: E402

from studies.vsp_planform import config  # noqa: E402
from studies.vsp_planform.degen_csv import lifting_surfaces, read_degen_csv  # noqa: E402
from studies.vsp_planform.param import rear_spar_fraction  # noqa: E402
import studies.vsp_planform.run_opt as ro  # noqa: E402
from studies.vsp_planform.run_opt import POINT, load_baseline, trim_alpha  # noqa: E402

import wing2_oas as w2  # noqa: E402
from plot_wing2 import planform  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "figures")
LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")

q = 0.5 * config.RHO * config.V_MS**2

# Plan L is the reference: it is the production baseline, so every percentage on
# these charts is quoted against it. ConstChord is shown as the other as-built
# geometry, and is the parent that wings 2 and 3 are derived from.
REFERENCE = "Plan L\nas-built"

COLORS = {
    "Plan L\nas-built": "#937860",
    "ConstChord\nas-built": "#4C72B0",
    "wing 2\n(7 in @ junction)": "#DD8452",
    "wing 3\n(6 in @ 90%)": "#C44E52",
    "wing 4\n(+ monotonic twist)": "#8172B2",
    "wing 5\n(+ inboard t/c)": "#55A868",
    "wing 6\n(t/c + monotonic)": "#0B7A75",
}


def eval_baseline(name):
    """As-built baseline at MTOW, full OAS, its own detected regions."""
    w2.apply_wing2_box()
    mesh, stick, regions, planform0, _, _ = load_baseline(name)
    prob, _ = ro.build_problem(name, mesh, stick, regions, planform0)
    prob.run_model()
    s_ref = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    alpha = trim_alpha(prob, w2.W / (q * s_ref))
    s_ref = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    cd = lambda k: float(prob.get_val(f"{POINT}.wing_perf.{k}")[0])
    m = prob.get_val("wing.mesh", units="m")
    return {
        "S_ref": s_ref,
        "CL": cd("CL"),
        "alpha": alpha,
        "toc": np.asarray(prob.get_val("wing.t_over_c")).ravel().copy(),
        "geom_name": name,
        # No scheduled box: an as-built baseline's spar is whatever the loft has,
        # so it is left out of the spar-ratio panel rather than drawn with a
        # schedule it was never designed to.
        "schedule": None,
        "induced_N": q * s_ref * cd("CDi"),
        "viscous_N": q * s_ref * cd("CDv"),
        "wave_N": q * s_ref * cd("CDw"),
        "drag_N": q * s_ref * (cd("CDi") + cd("CDv") + cd("CDw")),
        "mesh": m,
        "twist": prob.get_val("twist_abs", units="deg").copy(),
    }


def rebuild_from(case, schedule, stations):
    """Rebuild a stored optimized case so its planform can be drawn.

    A case carrying ``t_over_c_cp`` (wing 5) has its thickness restored too --
    without it the rebuild would draw wing 5 with the as-built loft and the drag
    cross-check below would fail, which is what it is there for.
    """
    w2.REAR_SCHEDULE = schedule
    w2.WIDTH_STATIONS = stations
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = stations

    prob, _, _, _, _ = w2.build(w2.BASELINE, w2.REGION_A_END_IN)
    prob.set_val("wing.taper_B", case["taper_B"])
    prob.set_val("wing.wingbox_pct", case["wingbox_pct"])
    prob.set_val("wing.twist_cp", np.array(case["twist_cp"]), units="deg")
    prob.set_val("alpha", case["alpha"], units="deg")
    if case.get("t_over_c_cp") is not None:
        prob.set_val("wing.t_over_c_cp", np.array(case["t_over_c_cp"]))
    prob.run_model()

    s_ref = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    cd = lambda k: float(prob.get_val(f"{POINT}.wing_perf.{k}")[0])
    drag = q * s_ref * (cd("CDi") + cd("CDv") + cd("CDw"))
    if abs(drag - case["drag_N"]) > 2.0:
        raise RuntimeError(f"rebuilt {drag:.1f} N != logged {case['drag_N']:.1f} N")
    return {
        "S_ref": s_ref,
        "CL": cd("CL"),
        "induced_N": q * s_ref * cd("CDi"),
        "viscous_N": q * s_ref * cd("CDv"),
        "wave_N": q * s_ref * cd("CDw"),
        "drag_N": drag,
        "mesh": prob.get_val("wing.mesh", units="m"),
        "toc": np.asarray(prob.get_val("wing.t_over_c")).ravel().copy(),
        "twist": prob.get_val("twist_abs", units="deg").copy(),
        "schedule": schedule,
    }


if __name__ == "__main__":
    cases = {}

    print("  running Plan L as-built (the reference) ...")
    cases["Plan L\nas-built"] = eval_baseline("plan_l")
    print("  running ConstChord as-built ...")
    cases["ConstChord\nas-built"] = eval_baseline("const_chord")

    stations_w2 = ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0), (674.9, w2.JUNCTION_BOX_IN))

    spar = json.load(open(os.path.join(LOGS, "spar_sweep_oas.json")))["cases"]
    w2case = [c for c in spar if abs(c["junction_spar_xc"] - 0.400) < 1e-9][0]
    print("  rebuilding wing 2 (admissible, spar 0.400c) ...")
    cases["wing 2\n(7 in @ junction)"] = rebuild_from(w2case, ((356.0, 0.750), (674.9, 0.400)), stations_w2)

    # wing 3 is rebuilt from aileron_90.json below. (There was a load of
    # "wing3_design_point.json" here whose result was never used, and which no
    # script in the repo writes -- it crashed the comparison on a clean tree.)
    ail = json.load(open(os.path.join(LOGS, "aileron_90.json")))
    w3case = [c for c in ail["cases"] if c["depth_req_in"] == 6.0 and abs(c["junction_spar_xc"] - 0.550) < 1e-9][0]
    y_ail = ail["meta"]["y_aileron_in"]
    stations_w3 = tuple(tuple(s) for s in [[100.0, 65.0], [176.0, 65.0], [356.0, 55.0],
                                           [y_ail, w3case["chord_req_at_aileron_in"] * (0.574 - 0.12)],
                                           [674.9, w2.JUNCTION_BOX_IN]])
    print("  rebuilding wing 3 (spar 0.550c) ...")
    cases["wing 3\n(6 in @ 90%)"] = rebuild_from(w3case, ((356.0, 0.750), (674.9, 0.550)), stations_w3)

    # wing 4 = wing 3 plus the monotonic-twist constraint. Same schedule and the
    # same constraint set; only the twist differs, so it reuses wing 3's setup.
    w4path = os.path.join(LOGS, "monotonic_twist_wing3.json")
    if os.path.exists(w4path):
        w4case = json.load(open(w4path))["monotonic"]
        print("  rebuilding wing 4 (wing 3 + monotonic twist) ...")
        cases["wing 4\n(+ monotonic twist)"] = rebuild_from(w4case, ((356.0, 0.750), (674.9, 0.550)), stations_w3)
    else:
        print("  NOTE: monotonic_twist_wing3.json not present -- wing 4 omitted")

    # wing 5 = wing 3's planform with t/c raised inboard only. It is run at MTOW
    # by wing5_mtow.py precisely so it can sit in this table; the coupled runs
    # that produced it were trimmed to cruise weight and cannot join.
    w5path = os.path.join(LOGS, "wing5_design_point.json")
    if os.path.exists(w5path):
        w5case = json.load(open(w5path))["wing5_mtow"]
        print("  rebuilding wing 5 (wing 3 + inboard t/c) ...")
        cases["wing 5\n(+ inboard t/c)"] = rebuild_from(
            w5case, ((356.0, 0.750), (674.9, 0.550)), stations_w3)
        cases["wing 5\n(+ inboard t/c)"]["w_wing_lb"] = 8032.5
        cases["wing 3\n(6 in @ 90%)"]["w_wing_lb"] = 8613.9
    else:
        print("  NOTE: wing5_design_point.json not present -- wing 5 omitted")

    # wing 6 = wing 5 with the monotonic-twist constraint. Same schedule and the
    # same station set as wing 3 and wing 5, so it reuses their setup; it carries
    # wing 5's t/c control points, and only the twist differs.
    w6path = os.path.join(LOGS, "wing6_design_point.json")
    if os.path.exists(w6path):
        w6case = json.load(open(w6path))["wing6_mtow"]
        print("  rebuilding wing 6 (wing 5 + monotonic twist) ...")
        cases["wing 6\n(t/c + monotonic)"] = rebuild_from(
            w6case, ((356.0, 0.750), (674.9, 0.550)), stations_w3)
    else:
        print("  NOTE: wing6_design_point.json not present -- wing 6 omitted")

    # ---- wing weight for every case, from the structural tool ----
    # A drag-only chart cannot rank wing 5, whose whole merit is weight. So each
    # case's rebuilt geometry is exported and sized. Cached: sizing is ~180 s a
    # case and the geometry only changes when a design does.
    from studies.vsp_planform.coupling import deck as wcdeck
    from studies.vsp_planform.coupling import geometry as wgeo

    wpath = os.path.join(LOGS, "wing_weights.json")
    weights = json.load(open(wpath)) if os.path.exists(wpath) else {}

    # Can the sizer actually run from here? WingCalc's bay loop forces a "spawn"
    # pool, and every worker re-imports the pickled callable in a fresh
    # interpreter -- where ``WingCalc_Tool`` does not exist unless it is on a
    # path by that name (the clone is Structures-WingCalc_Tool, and deck.py binds
    # the name through importlib, which a child does not inherit).
    # deck._alias_dir_for_workers() exports a correctly-named symlink on
    # PYTHONPATH so the workers can start; this confirms it before committing to
    # a 20-minute sizing run.
    import subprocess
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(wcdeck._alias_dir_for_workers())] +
        ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    can_size = subprocess.run([sys.executable, "-c", "import WingCalc_Tool"],
                              capture_output=True).returncode == 0
    if not can_size:
        print("  NOTE: WingCalc_Tool is not importable in a fresh interpreter, so its "
              "spawn workers cannot start.\n        Sizing skipped; cases without a "
              "recorded weight are drawn as 'not sized'.")
    comp0 = list(lifting_surfaces(read_degen_csv(config.BASELINES[w2.BASELINE])).values())[0][0]
    for n in list(cases):
        if n in weights:
            cases[n]["w_wing_lb"] = weights[n]
            continue
        if not can_size:
            continue
        if "w_wing_lb" in cases[n]:
            # wing 3 and wing 5 carry their converged coupled-loop weights, which
            # are a better number than a one-shot sizing at a fixed 8400 lb guess.
            continue
        c = cases[n]
        if "mesh" not in c or "toc" not in c:
            print(f"  {n.replace(chr(10),' ')}: no t/c recorded, weight skipped")
            continue
        tag = "cmp_" + n.split(chr(10))[0].replace(" ", "").replace("(", "").replace(")", "")
        print(f"  sizing {n.replace(chr(10), ' ')} ...", flush=True)
        # Section SHAPE comes from the case's OWN baseline loft: Plan L and
        # ConstChord are different aerofoils, and borrowing one for the other
        # would put the wrong thickness distribution into the box.
        gname = c.get("geom_name", w2.BASELINE)
        cmp_ = list(lifting_surfaces(read_degen_csv(config.BASELINES[gname])).values())[0][0]
        yj = 674.9 if gname == w2.BASELINE else float(np.abs(cmp_.stick.le[:, 1]).max())
        oas = {"mesh": np.asarray(c["mesh"]), "toc": np.asarray(c["toc"]),
               "plate": cmp_.plate, "stick": cmp_.stick, "y_junction": yj}
        wcdeck.write_deck(wcdeck.WC_DECK, Path(LOGS) / f"deck_{tag}",
                          86000.0, 8400.0, oas=oas)
        try:
            weights[n] = wcdeck.run_wingcalc(Path(LOGS) / f"deck_{tag}",
                                             Path(LOGS) / f"wc_{tag}")
        except Exception as exc:  # a sizer failure must not cost the whole figure
            print(f"    sizing FAILED for {n.replace(chr(10), ' ')}: {exc}", flush=True)
            continue
        cases[n]["w_wing_lb"] = weights[n]
        json.dump(weights, open(wpath, "w"), indent=2)

    base = cases[REFERENCE]["drag_N"]
    names = list(cases)
    for n in names:
        c = cases[n]
        wl = f"  W_wing {c['w_wing_lb']:7.1f} lb" if "w_wing_lb" in c else ""
        print(f"  {n.replace(chr(10), ' '):>28}  S_ref {c['S_ref']:7.3f}  drag {c['drag_N']:9.1f} N  "
              f"{c['drag_N'] / base - 1:+7.2%}   CL {c['CL']:.4f}{wl}")

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "Plan L (reference) vs ConstChord vs wings 2–6 — drag from full OAS at MTOW 382 547 N, span pinned at "
        "118 ft, all trimmed to the same lift;\nwing weight from WingCalc sizing the same geometry. "
        "All percentages are against PLAN L.",
        fontsize=13,
    )
    gs = fig.add_gridspec(3, 3, hspace=0.34, wspace=0.28)
    cols = [COLORS[n] for n in names]
    # Bar charts get the short name only; the full label overlaps its neighbours.
    short = [n.split("\n")[0] for n in names]

    # total drag
    ax = fig.add_subplot(gs[0, 0])
    vals = [cases[n]["drag_N"] for n in names]
    ax.bar(range(len(names)), vals, color=cols)
    ax.set_ylim(min(vals) * 0.97, max(vals) * 1.012)
    for i, v in enumerate(vals):
        ax.text(i, v + 8, f"{v:.0f}\n{v / base - 1:+.2%}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(short, fontsize=8.5, rotation=20, ha="right")
    ax.set_ylabel("drag, N"); ax.set_title("Total drag at MTOW", fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    # wing weight -- the axis a drag-only chart was missing
    ax = fig.add_subplot(gs[0, 1])
    wv = [cases[n].get("w_wing_lb") for n in names]
    have = [i for i, v in enumerate(wv) if v is not None]
    wref = cases[REFERENCE].get("w_wing_lb")
    if have:
        ax.bar([i for i in have], [wv[i] for i in have], color=[cols[i] for i in have])
        lo = min(wv[i] for i in have); hi = max(wv[i] for i in have)
        ax.set_ylim(lo * 0.94, hi * 1.03)
        for i in have:
            lbl = f"{wv[i]:.0f}"
            if wref:
                lbl += f"\n{wv[i] / wref - 1:+.2%}"
            ax.text(i, wv[i] + (hi - lo) * 0.03, lbl, ha="center", fontsize=9,
                    fontweight="bold")
    for i in range(len(names)):
        if wv[i] is None:
            ax.text(i, 0.5, "not sized", ha="center", va="center", fontsize=8,
                    color="#999", rotation=90, transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(names))); ax.set_xticklabels(short, fontsize=8.5, rotation=20, ha="right")
    ax.set_ylabel("wing weight, lb")
    ax.set_title("Wing weight (WingCalc, 20 bays closed)", fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    # area
    ax = fig.add_subplot(gs[0, 2])
    ar = [cases[n]["S_ref"] for n in names]
    ax.bar(range(len(names)), ar, color=cols)
    ax.set_ylim(min(ar) * 0.95, max(ar) * 1.03)
    for i, v in enumerate(ar):
        ax.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(short, fontsize=8.5, rotation=20, ha="right")
    ax.set_ylabel("S_ref, m²"); ax.set_title("Wing area", fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    # planforms
    ax = fig.add_subplot(gs[1, :2])
    for n in names:
        y, le, te = planform(cases[n]["mesh"])
        ax.plot(y, le, color=COLORS[n], lw=1.7, label=n.replace("\n", " "))
        ax.plot(y, te, color=COLORS[n], lw=1.7)
    ax.axvline(0.90 * 708.0, color="#8172B2", lw=1.6, ls="--")
    ax.text(0.90 * 708.0 - 8, ax.get_ylim()[1], " aileron (wing 3)", color="#8172B2",
            fontsize=8, rotation=90, va="top", ha="right")
    ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_xlabel("y, in"); ax.set_ylabel("x, in")
    ax.set_title("Planforms", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")

    # chord
    ax = fig.add_subplot(gs[1, 2])
    for n in names:
        y, le, te = planform(cases[n]["mesh"])
        ax.plot(y, te - le, color=COLORS[n], lw=1.8, label=n.replace("\n", " "))
    ax.axvline(0.90 * 708.0, color="#8172B2", lw=1.5, ls="--")
    ax.set_xlabel("y, in"); ax.set_ylabel("chord, in")
    ax.set_title("Chord distribution", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # twist
    ax = fig.add_subplot(gs[2, :2])
    for n in names:
        y, _, _ = planform(cases[n]["mesh"])
        # wing 5 carries wing 3's twist and wing 6 carries wing 4's to four
        # decimals, so the later pair is dashed -- equal solid lines would draw
        # four curves and show two.
        dashed = n.startswith(("wing 5", "wing 6"))
        ax.plot(y, cases[n]["twist"], color=COLORS[n], lw=1.9,
                ls=(0, (5, 2.5)) if dashed else "-", label=n.replace("\n", " "))
    ax.axvline(0.90 * 708.0, color="#8172B2", lw=1.5, ls="--")
    ax.text(0.90 * 708.0, ax.get_ylim()[1], " aileron (wing 3)", color="#8172B2", fontsize=8, va="top")
    ax.set_xlabel("y, in"); ax.set_ylabel("twist, deg")
    ax.set_title("Twist — wings 4 and 6 constrained monotonic ROOT TO JUNCTION\n(both sit on the +5 deg root bound)", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)

    # spar chord fractions -- the box that every one of these designs is really
    # arguing about. The front spar is fixed at 0.12c for all of them; the rear
    # spar is the scheduled, kinking one, and it is the difference between wing 2
    # and wing 3.
    ax = fig.add_subplot(gs[2, 2])
    # Wings 3-6 share the 0.550c junction schedule exactly, so they are one curve,
    # not four. Identical schedules are collapsed into a single labelled line
    # rather than stacked -- four lines under one another read as four designs
    # making four different choices, which is the opposite of the truth here.
    groups = {}
    for n in (n for n in names if cases[n].get("schedule") is not None):
        groups.setdefault(tuple(map(tuple, cases[n]["schedule"])), []).append(n)
    for sched, members in groups.items():
        y, _, _ = planform(cases[members[0]]["mesh"])
        label = ", ".join(m.split(chr(10))[0].replace("wing ", "") for m in members)
        label = f"wing{'s' if len(members) > 1 else ''} {label} aft ({sched[-1][1]:.3f}c junction)"
        ax.plot(y, rear_spar_fraction(y, sched), color=COLORS[members[0]], lw=2.2,
                label=label)
    ax.axhline(w2.FRONT_PCT, color="0.25", lw=1.6, ls="--",
               label=f"front spar {w2.FRONT_PCT:.2f}c (all)")
    for y_kink in (356.0, 674.9):
        ax.axvline(y_kink, color="0.8", lw=0.9, ls=":")
    ax.annotate("schedule breakpoints\n356 in / 674.9 in", xy=(674.9, 0.19),
                xytext=(-10, 0), textcoords="offset points", ha="right",
                fontsize=7.5, color="0.45")
    ax.axvline(0.90 * 708.0, color="#8172B2", lw=1.5, ls="--")
    ax.set_ylim(0.0, 0.85)
    ax.set_xlabel("y, in"); ax.set_ylabel("spar station, x/c")
    ax.set_title("Front and aft spar chord ratios\n(as-built baselines have no scheduled box)", fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="center left", framealpha=0.9)

    fig.text(0.5, 0.012,
             "wing 2 shown is the ADMISSIBLE 7 in design (spar 0.400c). The often-quoted -2.52% wing 2 cannot meet its own "
             "depth requirement: at its 52.77 in junction chord the section's MAX thickness is 6.25 in.",
             ha="center", fontsize=8.5, style="italic")

    path = os.path.join(OUT_DIR, "wing_comparison.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {path}")

    def jsonable(v):
        return v.tolist() if isinstance(v, np.ndarray) else v

    json.dump({n.replace("\n", " "): {k: jsonable(v) for k, v in c.items() if k not in ("mesh", "schedule")}
               for n, c in cases.items()},
              open(os.path.join(LOGS, "wing_comparison.json"), "w"), indent=2)
