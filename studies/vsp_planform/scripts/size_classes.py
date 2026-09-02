"""Size the three architectures with WingCalc, so the trade has a weight axis.

Drag ranks nothing on its own here: the study's merit function is electric range
at fixed MTOW, m_batt/D, and break-even is ~1.49 lb of wing weight per newton of
drag. The three classes sit within 122 N of each other, so which one wins is
decided almost entirely by structure -- and none of them had a weight.

Bi-level, for the reason coupled_loop.py records: one WingCalc evaluation is
0.3 s but bay sizing is ~180 s over ~143k trials of integer ply counts, so the
sizer cannot sit inside a gradient loop. Wing weight enters OAS only as the load
it must carry, and the measured loop gain is -0.03 -- a heavier wing relieves its
own bending -- so the fixed point converges in 2-4 passes.

Each pass: replay the architecture's stored design vector (NOT a re-optimization,
the planform is fixed), export the OAS geometry as the OpenVSP station file
WingCalc's provider reads, size every bay, and feed the resulting wing weight
back as the next pass's structural design weight.

Writes out/logs/class_weights.json.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

from studies.vsp_planform import config                        # noqa: E402
from studies.vsp_planform.degen_csv import read_degen_csv, lifting_surfaces  # noqa: E402
from studies.vsp_planform.coupling import deck as wcdeck       # noqa: E402
from studies.vsp_planform.coupling import mission              # noqa: E402
import wing2_oas as w2                                         # noqa: E402
from wing8_constchord_toc import REGION_A_AS_BUILT_IN          # noqa: E402
from compare_classes import replay                             # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")
MTOW_LB = mission.MTOW_LB
W_SEED_LB, PASSES, TOL_LB = 8400.0, 4, 25.0


def size(case, y_a_in, rule, label):
    """Weight fixed point for one architecture. Returns the converged history."""
    r = replay(case, y_a_in, rule)
    comp = list(lifting_surfaces(read_degen_csv(config.BASELINES[w2.BASELINE])).values())[0][0]
    oas = {"mesh": r["mesh"], "toc": r["toc"], "plate": comp.plate,
           "stick": comp.stick, "y_junction": 674.9}

    w, hist = W_SEED_LB, []
    tag = label.replace(" ", "_")
    for p in range(1, PASSES + 1):
        print(f"\n{'#'*72}\n# {label} pass {p}: W_wing in {w:.1f} lb\n{'#'*72}", flush=True)
        wcdeck.write_deck(wcdeck.WC_DECK, Path(LOGS) / f"deck_{tag}", MTOW_LB, w, oas=oas)
        w_new = wcdeck.run_wingcalc(Path(LOGS) / f"deck_{tag}", Path(LOGS) / f"wc_{tag}")
        rng = mission.electric_range_nmi(w_new, r["drag_N"])
        hist.append({"pass": p, "w_in_lb": w, "w_wing_lb": w_new,
                     "residual_lb": w_new - w, "batt_lb": mission.battery_lb(w_new),
                     "R_nmi": rng})
        print(f">>> {label} p{p}: {w:.1f} -> {w_new:.1f} lb ({w_new - w:+.1f}) | "
              f"drag {r['drag_N']:.1f} N | batt {mission.battery_lb(w_new):.1f} lb | "
              f"R {rng:.2f} nmi", flush=True)
        if abs(w_new - w) < TOL_LB:
            break
        w += 0.5 * (w_new - w)          # damped: the loop gain is small but negative
    return r, hist


if __name__ == "__main__":
    w7 = json.load(open(os.path.join(LOGS, "wing7_design_point.json")))
    w8 = json.load(open(os.path.join(LOGS, "wing8_design_point.json")))
    ARCHS = [
        ("free (wing 3)", w7["wing3_mtow"], w2.REGION_A_END_IN, "root_le_fixed"),
        ("straight fwd spar (wing 7)", w7["wing7_mtow"], w2.REGION_A_END_IN, "preserved"),
        ("constant chord (wing 8)", w8["constchord_asbuilt"], REGION_A_AS_BUILT_IN, "root_le_fixed"),
    ]

    out = {}
    for label, case, y_a, rule in ARCHS:
        r, hist = size(case, y_a, rule, label)
        out[label] = {"drag_N": r["drag_N"], "S_ref": r["S_ref"],
                      "w_wing_lb": hist[-1]["w_wing_lb"],
                      "batt_lb": hist[-1]["batt_lb"], "R_nmi": hist[-1]["R_nmi"],
                      "converged": abs(hist[-1]["residual_lb"]) < TOL_LB,
                      "history": hist}
        with open(os.path.join(LOGS, "class_weights.json"), "w") as f:
            json.dump(out, f, indent=2)

    print("\n" + "=" * 92)
    print(f"{'architecture':28} {'drag N':>10} {'W_wing lb':>11} {'batt lb':>10} {'range nmi':>11} {'vs free':>10}")
    ref = out["free (wing 3)"]["R_nmi"]
    for k, v in out.items():
        flag = "" if v["converged"] else "  (NOT converged)"
        print(f"{k:28} {v['drag_N']:>10.1f} {v['w_wing_lb']:>11.1f} {v['batt_lb']:>10.1f} "
              f"{v['R_nmi']:>11.2f} {100*(v['R_nmi']/ref-1):>+9.2f}%{flag}")
    print("=" * 92)
    print("\nWeight is what ranks these: break-even is "
          f"{mission.battery_lb(out['free (wing 3)']['w_wing_lb'])/out['free (wing 3)']['drag_N']:.3f} lb per N of drag.")
