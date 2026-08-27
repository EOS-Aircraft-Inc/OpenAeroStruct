"""Aero-structural weight fixed point: OAS planform <-> WingCalc sizing.

The loop the study has been missing. Wing weight is the exchanged scalar:

    MTOW    = K + W_wing + payload + fuel        K = OEW_nom - W_wing_nom
    cruise  = MTOW - 0.5 * fuel                  what OAS trims lift to
    W_wing  = WingCalc(optimize_bay) at MTOW     structural design weight

Two weights, deliberately different: OAS flies at *cruise* weight, WingCalc
sizes its 2.5 g cases against *MTOW*. Using one for both is a silent
few-percent error in either drag or margins.

COUPLED: weight AND geometry. Each pass rewrites the deck's OpenVSP station
export from the OAS optimized mesh (see wingcalc_geom.py), so the planform,
chords and t/c distribution OAS chose are what WingCalc sizes.
"""

import json
import sys
import time
from pathlib import Path

# The repo root is not on the path from a bare script run: `pip install -e .`
# installs `openaerostruct` only, and sys.path[0] is this directory. Every
# other script in the study inserts it; these never did, so they worked only
# from a shell that had already exported PYTHONPATH.
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3]))

from studies.vsp_planform.coupling.deck import (  # noqa: E402
    WC_DECK, WC_ROOT, FRONT_PCT, AFT_PCT_SCALAR, WINGBOX_SPAN_IN,
    run_wingcalc, write_deck,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

LB = 4.4482216  # N per lbf

# --- user's numbers -------------------------------------------------------
OEW_NOM = 63_500.0      # lbm
W_WING_NOM = 7_500.0    # lbm, the wing inside OEW_NOM
PAYLOAD = 17_100.0      # lbm
FUEL = 5_400.0          # lbm  -> MTOW 86,000 at W_wing 7,500
K = OEW_NOM - W_WING_NOM        # 56,000 lbm of non-wing aircraft

BASELINE = "const_chord"
MAX_PASSES = 5
TOL_LB = 25.0           # wing-weight residual to call it converged



WORK = Path(__file__).resolve().parent.parent / "out" / "logs"


def weights(w_wing):
    mtow = K + w_wing + PAYLOAD + FUEL
    return mtow, mtow - 0.5 * FUEL


# --- wing 3 structural configuration -------------------------------------
# The aft-spar depth requirement houses the aileron actuator. config.py carries
# only the defaults ((100,65) and no rear-spar kink), so the wing 3 constraint
# set has to be installed the same way aileron_90.py installs it -- otherwise
# the optimizer shrinks the chord with nothing outboard holding it.
AILERON_FRAC = 0.90
SEMI_IN = 118.0 * 12.0 / 2.0            # 708 in
Y_AIL = AILERON_FRAC * SEMI_IN          # 637.2 in
DEPTH_REQ_IN = 6.0                      # at Y_AIL -- wing 3's requirement
JUNCTION_SPAR = 0.550                   # aft spar x/c at the winglet junction


def _wing3_setup():
    """Install wing 3's rear-spar schedule and width stations. Returns helpers."""
    import numpy as np
    from studies.vsp_planform import config
    from studies.vsp_planform.param import rear_spar_fraction
    import wing2_oas as w2
    from doe_v3 import asbuilt

    af = asbuilt()
    xs = np.linspace(0.05, 0.95, 300)
    t = np.array([float(af.local_thickness(x_over_c=x)) for x in xs])
    ret_of = lambda x: float(np.interp(x, xs, t / t.max()))

    w2.apply_wing2_box()
    _, stick0, _, _ = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    y_s = np.abs(np.asarray(stick0.le[:, 1], dtype=float))
    toc_at = lambda y: float(np.interp(y, y_s, stick0.toc))

    schedule = ((356.0, 0.750), (674.9, JUNCTION_SPAR))
    spar_ail = float(rear_spar_fraction(Y_AIL, schedule))
    ret, toc = ret_of(spar_ail), toc_at(Y_AIL)

    # depth -> minimum chord -> equivalent minimum box width (exact re-encoding:
    # at a fixed spar fraction depth is strictly proportional to chord)
    c_req = DEPTH_REQ_IN / (ret * toc)
    w_equiv = (spar_ail - w2.FRONT_PCT) * c_req

    stations = ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0),
                (Y_AIL, w_equiv), (674.9, w2.JUNCTION_BOX_IN))

    w2.REAR_SCHEDULE = schedule
    w2.WIDTH_STATIONS = stations
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = stations
    print(f"  wing 3 box: spar@ail {spar_ail:.3f}, t/c {toc:.4f}, retention {ret:.4f}"
          f" -> chord req {c_req:.2f} in, width req {w_equiv:.2f} in", flush=True)
    return w2, ret, toc, c_req


def run_oas(cruise_lb):
    """Optimize wing 3's planform at this cruise weight; hand back the mesh."""
    import numpy as np
    from studies.vsp_planform import run_opt, config
    from studies.vsp_planform.degen_csv import read_degen_csv, lifting_surfaces

    w2, ret, toc, c_req = _wing3_setup()
    weight_N = cruise_lb * LB

    mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    prob, _ = run_opt.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
    prob.run_model()
    q = 0.5 * config.RHO * config.V_MS**2
    s0 = float(prob.get_val(f"{run_opt.POINT}.wing.S_ref")[0])
    alpha0 = run_opt.trim_alpha(prob, weight_N / (q * s0))
    run_opt.add_optimization(prob, "plan_l", mesh, planform0, s0,
                             mode="fixed_lift", weight=weight_N)
    prob.set_val("alpha", alpha0, units="deg")
    prob.run_model()
    prob.run_driver()

    # Capture the optimum BEFORE w2.evaluate(): that helper re-trims the model to
    # its own hardcoded MTOW (wing2_oas.W = 382547 N) on its first line, so a
    # _state() read after it reports MTOW aero no matter what weight was optimized.
    state = run_opt._state(prob)
    r = w2.evaluate(prob, regions.y_c_start)
    c_ail = float(r["station_chord_in"][3])
    comp = list(lifting_surfaces(read_degen_csv(config.BASELINES[w2.BASELINE])).values())[0][0]
    return {
        "optimized": state,
        "success": bool(prob.driver.result.success),
        "iterations": int(prob.driver.result.iter_count),
        "mesh": np.asarray(prob.get_val("wing.mesh", units="m")),
        "toc": np.asarray(prob.get_val("wing.t_over_c")).ravel(),
        "plate": comp.plate, "stick": comp.stick,
        "y_junction": 674.9,
        "fwd_spar_x_bl0": float(comp.stick.le[0, 0]) + FRONT_PCT * float(comp.stick.chord[0]),
        "fwd_spar_z_bl0": float(comp.stick.le[0, 2]),
        "chord_req_ail_in": c_req,
        "chord_ail_in": c_ail,
        "depth_ail_in": ret * toc * c_ail,
        "junction_chord_in": float(r["junction_chord_in"]),
    }


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    w_wing = W_WING_NOM
    history = []

    for p in range(1, MAX_PASSES + 1):
        mtow, cruise = weights(w_wing)
        print(f"\n{'#' * 78}\n# PASS {p}: W_wing {w_wing:.1f} lbm -> "
              f"MTOW {mtow:.1f} lbm, cruise {cruise:.1f} lbm\n{'#' * 78}", flush=True)

        t0 = time.perf_counter()
        oas = run_oas(cruise)
        t_oas = time.perf_counter() - t0

        deck = WORK / f"deck_pass{p}"
        write_deck(WC_DECK, deck, mtow, w_wing, oas=oas)
        t0 = time.perf_counter()
        w_new = run_wingcalc(deck, WORK / f"wc_pass{p}")
        t_wc = time.perf_counter() - t0

        resid = w_new - w_wing
        opt = oas["optimized"]
        history.append({
            "pass": p, "w_wing_in": w_wing, "mtow": mtow, "cruise": cruise,
            "w_wing_out": w_new, "residual_lb": resid,
            "drag_N": opt["drag_N"], "CL": opt["CL"], "S_ref": opt["S_ref"],
            "L/D": opt["L/D"], "oas_s": t_oas, "wc_s": t_wc,
            "oas_success": oas["success"], "oas_iters": oas["iterations"],
            "wingbox_pct": opt["wingbox_pct"], "taper_B": opt["taper_B"],
            "depth_ail_in": oas["depth_ail_in"],
            "chord_ail_in": oas["chord_ail_in"],
            "chord_req_ail_in": oas["chord_req_ail_in"],
            "junction_chord_in": oas["junction_chord_in"],
        })
        print(f"\n>>> PASS {p}: W_wing {w_wing:.1f} -> {w_new:.1f} lbm "
              f"(residual {resid:+.1f} lb) | drag {opt['drag_N']:.1f} N, "
              f"L/D {opt['L/D']:.2f}, S_ref {opt['S_ref']:.3f} m^2 "
              f"| depth@ail {oas['depth_ail_in']:.2f} in (req {DEPTH_REQ_IN:.1f}) "
              f"| wingbox_pct {opt['wingbox_pct']:.4f} "
              f"| OAS {t_oas:.0f}s WC {t_wc:.0f}s", flush=True)

        (WORK / "coupled_loop.json").write_text(json.dumps(history, indent=2))

        if abs(resid) < TOL_LB:
            print(f"\nCONVERGED after pass {p}: |residual| {abs(resid):.1f} < {TOL_LB} lb")
            break
        w_wing = w_wing + 0.5 * resid   # damped update

    print("\n" + "=" * 96)
    print(f"{'pass':>4} {'W_in':>9} {'MTOW':>10} {'cruise':>9} {'W_out':>9} "
          f"{'resid':>8} {'drag N':>9} {'L/D':>7} {'S_ref':>8} {'depth':>7} {'wb_pct':>7}")
    for h in history:
        print(f"{h['pass']:>4} {h['w_wing_in']:>9.1f} {h['mtow']:>10.1f} "
              f"{h['cruise']:>9.1f} {h['w_wing_out']:>9.1f} {h['residual_lb']:>+8.1f} "
              f"{h['drag_N']:>9.1f} {h['L/D']:>7.2f} {h['S_ref']:>8.3f} "
              f"{h['depth_ail_in']:>7.2f} {h['wingbox_pct']:>7.4f}")
    print("=" * 96)


if __name__ == "__main__":
    main()
