"""Spanwise t/c PROFILE sweep: how thick, and how the thickness is distributed.

The uniform-scale sweep said thicker always wins over 0.150-0.200 root t/c, and
was still climbing at the top of the band (+2.90% electric range at 0.200). Two
questions that leaves open, and this answers both:

  LEVEL  how far does it keep paying?          root t/c 0.20 -> 0.25
  SHAPE  is uniform thickening even right?     tip/root ratio 1.00 (constant t/c
         root to tip) -> 0.58 (the as-built taper, thick root and thin tip)

Shape matters because the two costs live at different stations. Drag is paid over
the whole wetted area, so thickness anywhere costs. Structure is paid where the
bending moment is, which is inboard. And the 6 in depth requirement sits at 90%
semi-span, which is neither -- so the constraint pulls thickness outboard while
the structure wants it inboard.

t/c is set as a linear ramp on the 5 spline control points, root -> root*ratio.
A pre-pass reads the achieved t/c at the aileron station before the depth
requirement is built, so c_req = 6 in / (retention * t/c) tracks the profile
rather than the baseline loft.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coupled_loop as cl

# Finding the CROSSOVER: where extra thickness stops paying. The uniform sweep
# was still climbing at root 0.200, but its marginal return was decaying fast
# (+277 / +282 / +155 / +108 % of range per unit t/c across successive bands),
# which extrapolates to zero near root t/c ~ 0.22. These points bracket that.
# Shape is held at the as-built taper so only the LEVEL varies.
# LEVEL x SHAPE, both swept. The earlier run held the shape at the as-built-like
# 0.58 and only moved the level, which cannot answer what the profile should be.
# 0.45 reaches a 0.10 tip from a 0.22 root; 1.00 is constant t/c to the tip.
ROOT_TOCS = [0.180, 0.200, 0.220, 0.250]
TIP_RATIOS = [0.45, 0.58, 0.75, 1.00]
W_WING_SEED = 8440.1
MTOW_LB, K_LB, PAYLOAD_LB, FUEL_LB = 86_000.0, 56_000.0, 17_100.0, 5_400.0
BATT_LB_BOOK = 16_665.6
K_EX_BATT_LB = K_LB - BATT_LB_BOOK
ETA, E_STAR, LB_KG, NMI_M = 0.80, 300.0, 0.45359237, 1852.0
OUT = Path(__file__).resolve().parent.parent / "out" / "logs" / "coupled_toc_profile.json"


def _cp_ramp(n, root, ratio):
    return np.linspace(root, root * ratio, n)


def run(root_toc, ratio):
    from studies.vsp_planform import run_opt, config
    from studies.vsp_planform.degen_csv import read_degen_csv, lifting_surfaces
    from studies.vsp_planform.param import rear_spar_fraction
    import wing2_oas as w2
    from doe_v3 import asbuilt

    cruise = MTOW_LB - 0.5 * FUEL_LB

    af = asbuilt()
    xs = np.linspace(0.05, 0.95, 300)
    t = np.array([float(af.local_thickness(x_over_c=x)) for x in xs])
    ret_of = lambda x: float(np.interp(x, xs, t / t.max()))

    w2.apply_wing2_box()
    schedule = ((356.0, 0.750), (674.9, cl.JUNCTION_SPAR))
    spar_ail = float(rear_spar_fraction(cl.Y_AIL, schedule))
    ret = ret_of(spar_ail)

    # --- pre-pass: what t/c does this profile actually put at the aileron? ---
    w2.REAR_SCHEDULE = schedule
    w2.WIDTH_STATIONS = ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0),
                         (cl.Y_AIL, 0.0), (674.9, w2.JUNCTION_BOX_IN))
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = w2.WIDTH_STATIONS
    mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    pre, _ = run_opt.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
    n_cp = int(np.asarray(pre.get_val("wing.t_over_c_cp")).size)
    pre.set_val("wing.t_over_c_cp", _cp_ramp(n_cp, root_toc, ratio))
    pre.run_model()
    toc_v = np.asarray(pre.get_val("wing.t_over_c")).ravel()
    y_nodes = np.abs(np.asarray(pre.get_val("wing.mesh", units="m"))[0, :, 1]) / config.SCALE
    y_panel = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    o = np.argsort(y_panel)
    toc_ail = float(np.interp(cl.Y_AIL, y_panel[o], toc_v[o]))

    # --- real run with the depth requirement keyed to that t/c ---
    c_req = cl.DEPTH_REQ_IN / (ret * toc_ail)
    w_equiv = (spar_ail - w2.FRONT_PCT) * c_req
    stations = ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0),
                (cl.Y_AIL, w_equiv), (674.9, w2.JUNCTION_BOX_IN))
    w2.WIDTH_STATIONS = stations
    config.WINGBOX_WIDTH_STATIONS = stations

    mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    prob, _ = run_opt.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
    prob.set_val("wing.t_over_c_cp", _cp_ramp(n_cp, root_toc, ratio))
    prob.run_model()
    q = 0.5 * config.RHO * config.V_MS**2
    s0 = float(prob.get_val(f"{run_opt.POINT}.wing.S_ref")[0])
    alpha0 = run_opt.trim_alpha(prob, cruise * cl.LB / (q * s0))
    run_opt.add_optimization(prob, "plan_l", mesh, planform0, s0,
                             mode="fixed_lift", weight=cruise * cl.LB)
    prob.set_val("wing.t_over_c_cp", _cp_ramp(n_cp, root_toc, ratio))   # setup() reset it
    prob.set_val("alpha", alpha0, units="deg")
    prob.run_model()
    prob.run_driver()

    st = run_opt._state(prob)
    toc_final = np.asarray(prob.get_val("wing.t_over_c")).ravel()
    chords = prob.get_val("station_chord", units="m") / config.SCALE

    comp = list(lifting_surfaces(read_degen_csv(config.BASELINES[w2.BASELINE])).values())[0][0]
    oas = {"mesh": np.asarray(prob.get_val("wing.mesh", units="m")), "toc": toc_final,
           "plate": comp.plate, "stick": comp.stick, "y_junction": 674.9}

    # the tag names this point's deck and work directory: unique per point, so
    # concurrent shards never write the same path
    tag = f"r{int(root_toc*1000)}_t{int(ratio*100)}"
    cl.write_deck(cl.WC_DECK, OUT.parent / f"deck_{tag}", MTOW_LB, W_WING_SEED, oas=oas)
    w_new = cl.run_wingcalc(OUT.parent / f"deck_{tag}", OUT.parent / f"wc_{tag}")

    m_batt = MTOW_LB - K_EX_BATT_LB - PAYLOAD_LB - FUEL_LB - w_new
    r_nmi = ETA * (m_batt * LB_KG * E_STAR * 3600.0) / st["drag_N"] / NMI_M

    return {"root_toc": root_toc, "ratio": ratio,
            "toc_root": float(toc_final[0]), "toc_tip": float(toc_final[-1]),
            "toc_ail": toc_ail, "chord_req_ail_in": c_req,
            "drag_N": st["drag_N"], "S_ref": st["S_ref"], "CL": st["CL"],
            "L/D": st["L/D"], "w_wing_lb": w_new, "m_batt_lb": m_batt,
            "R_nmi": r_nmi, "wingbox_pct": st["wingbox_pct"],
            "chord_ail_in": float(chords[3]), "success": bool(prob.driver.result.success)}


def main():
    # Sharding. Each point is an independent coupled evaluation -- OAS optimize,
    # export, size -- so the grid parallelises perfectly across processes. A shard
    # takes every nth point and writes its own file; merge_shards() combines them.
    # Cap WINGCALC_BAY_WORKERS per shard so the bay pools do not oversubscribe:
    # shards x workers should stay under the core count.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    # --roots / --ratios override the grid, so extra rows can be added to a run
    # already in flight without disturbing it; --tag names the output file.
    ap.add_argument("--roots", type=str, default=None, help="comma-separated root t/c")
    ap.add_argument("--ratios", type=str, default=None, help="comma-separated tip/root")
    ap.add_argument("--tag", type=str, default=None, help="output file suffix")
    args = ap.parse_args()

    roots = [float(v) for v in args.roots.split(",")] if args.roots else ROOT_TOCS
    ratios = [float(v) for v in args.ratios.split(",")] if args.ratios else TIP_RATIOS
    points = [(root, ratio) for root in roots for ratio in ratios]
    mine = [pt for i, pt in enumerate(points) if i % args.nshards == args.shard]
    suffix = args.tag if args.tag else f"shard{args.shard}"
    out = (OUT if (args.nshards == 1 and not args.tag)
           else OUT.with_name(f"{OUT.stem}_{suffix}{OUT.suffix}"))
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} of {len(points)} points "
          f"-> {out.name}", flush=True)

    res = []
    for root, ratio in mine:
        print(f"\n{'#'*78}\n# root t/c {root:.3f}, tip/root {ratio:.2f} "
              f"(tip {root*ratio:.3f})\n{'#'*78}", flush=True)
        t0 = time.perf_counter()
        r = run(root, ratio)
        r["seconds"] = time.perf_counter() - t0
        res.append(r)
        print(f"\n>>> root {root:.3f} ratio {ratio:.2f} | t/c ail {r['toc_ail']:.4f} "
              f"| S_ref {r['S_ref']:.3f} | drag {r['drag_N']:.1f} N "
              f"| W_wing {r['w_wing_lb']:.1f} lb | R {r['R_nmi']:.1f} nmi", flush=True)
        out.write_text(json.dumps(res, indent=2))

    if args.nshards > 1 or args.tag:   # the merge owns the whole grid
        return

    base = res[0]
    print("\n" + "=" * 122)
    print(f"{'root':>6} {'ratio':>6} {'t/c rt':>7} {'t/c tip':>8} {'t/c ail':>8} {'S_ref':>8} "
          f"{'drag N':>9} {'W_wing':>9} {'m_batt':>9} {'R nmi':>8} {'vs 1st':>8}")
    for r in res:
        d = 100.0 * (r["R_nmi"] / base["R_nmi"] - 1.0)
        print(f"{r['root_toc']:>6.3f} {r['ratio']:>6.2f} {r['toc_root']:>7.4f} "
              f"{r['toc_tip']:>8.4f} {r['toc_ail']:>8.4f} {r['S_ref']:>8.3f} "
              f"{r['drag_N']:>9.1f} {r['w_wing_lb']:>9.1f} {r['m_batt_lb']:>9.1f} "
              f"{r['R_nmi']:>8.1f} {d:>+7.2f}%")
    print("=" * 122)
    b = max(res, key=lambda r: r["R_nmi"])
    print(f"BEST: root t/c {b['root_toc']:.3f}, tip/root {b['ratio']:.2f} "
          f"-> {b['R_nmi']:.1f} nmi, drag {b['drag_N']:.1f} N, wing {b['w_wing_lb']:.1f} lb")


if __name__ == "__main__":
    main()
