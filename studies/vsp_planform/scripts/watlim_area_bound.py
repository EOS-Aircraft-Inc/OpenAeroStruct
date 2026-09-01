"""The wing area at which the CLIMB GRADIENT requirements are just met.

WHAT THIS IS, AND WHAT IT IS NOT. field_length_bound.py bounds the wing area with
take-off and landing DISTANCE, read from interpolation tables. This bounds it with
the WAT climb GRADIENTS, and there is no table for those -- they come out of the
Atlas mission simulation, one run per wing. So this is a bisection over runs, not a
root find on a surface, and it is slow by construction.

THE REQUIREMENTS, from run_watlim_only.py:

    watlim_2nd   >= 3.0 %      second segment, one engine inoperative
    watlim_4th   >= 1.7 %      fourth segment
    approach     >= 2.7 %
    landing      >= 3.2 %
    aeo          >= 2.2 %      all engines operating

held at every altitude and every DISA the mission config sweeps.

POWER IS HELD, AT THE AIRCRAFT LEVEL. The nominal 1400 kW, applied through
``motor.rating`` because that is the field run_watlim_analysis actually reads. See
_set_power: writing nacelle.electric_mcp or hybrid_mcp changes nothing at all.

SPAN IS HELD, NOT ASPECT RATIO. This study pins the span at 118 ft, so a change of
wing area is a change of ASPECT RATIO: AR = span^2 / S_ref. Atlas's own sizing does
the opposite -- it holds AR and lets the span follow (size_aircraft.py:632) -- so
passing an area to Atlas without also setting the span would silently change the
span and answer a different question. ``--hold ar`` selects Atlas's convention if
that comparison is ever wanted.

EVERY REQUIREMENT GETS ITS OWN CROSSOVER. Reporting only the worst margin hides the
answer, because the phases do not even move the same way with area. Measured at
1400 kW: approach and landing get WORSE as the wing shrinks and each has a genuine
crossover, AEO gets BETTER as it shrinks and is met everywhere, and the two WATLIM
segments are not met at any area at all. A single "boundary area" would have been
one of those five and would have been silent about the other four.
"""

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")
M2_FT2 = 10.7639104
FT_TO_M = 0.3048
SPAN_FT = 118.0
MTOP_KW_NOMINAL = 1400.0

REQUIRED_PCT = {"watlim_2nd": 3.0, "watlim_4th": 1.7, "approach": 2.7,
                "landing": 3.2, "aeo": 2.2}


@contextlib.contextmanager
def _atlas_cwd():
    """Atlas reads ac_data.xlsx and its empirical CSVs by paths relative to its root."""
    import atlas
    root = os.path.dirname(os.path.dirname(os.path.abspath(atlas.__file__)))
    prev = os.getcwd()
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(prev)


def _set_wing(ac_data, s_ref_m2, hold, span_ft=SPAN_FT):
    """Put this wing into the Atlas aircraft dictionary, consistently.

    S_ref, S_plan and S_trap all move together, and so do the three spans. Setting
    the area alone leaves Atlas with a wing whose area and span disagree, which it
    does not check and which quietly changes the answer.
    """
    w = ac_data["ac"]["geom"]["wing"]
    if hold == "span":
        span_m = span_ft * FT_TO_M
        ar = span_m**2 / s_ref_m2
    else:                                   # Atlas's own convention
        ar = float(w["AR"]["value"])
        span_m = float(np.sqrt(s_ref_m2 * ar))
    for k in ("S_ref", "S_plan", "S_trap"):
        w[k]["value"] = s_ref_m2
        w[k]["units"] = "m**2"
    for k in ("span", "span_plan", "span_trap"):
        w[k]["value"] = span_m
        w[k]["units"] = "m"
    w["AR"]["value"] = ar
    if "AR_plan" in w and isinstance(w["AR_plan"], dict):
        w["AR_plan"]["value"] = ar
    return {"S_ref_m2": s_ref_m2, "span_m": span_m, "AR": ar}


def _set_power(ac_data, aircraft_kw):
    """Hold the installed power at an AIRCRAFT-level value, in kW.

    The knob is ``motor.rating``, not the nacelle MCP. run_watlim_analysis reads
    ``ac|propulsion|motor|rating`` and derives everything else from it through
    prepare_pow_ratings (size_aircraft_doe.py). Setting nacelle.electric_mcp or
    hybrid_mcp instead changes NOTHING -- measured: the gradients were identical
    from 1400 kW to 2800 kW, which is how this was found.

    A nacelle carries ``em_stks_per_nac * em_per_stk`` motor units, 6 by default, so
    the aircraft-level power is that many times the motor rating. The default rating
    of 264.283 kW is therefore about 1586 kW at the aircraft.
    """
    p = ac_data["ac"]["propulsion"]
    units_per_nac = (float(p["nacelle"]["em_stks_per_nac"]["value"])
                     * float(p["nacelle"]["em_per_stk"]["value"]))
    rating = float(aircraft_kw) / units_per_nac
    ratio = float(p["motor"]["nom_climb_rated_pow_ratio"]["value"])
    p["motor"]["rating"]["value"] = rating
    p["motor"]["rating"]["units"] = "kW"
    p["motor"]["nom_climb_e_pow_per_nac"]["value"] = rating * units_per_nac * ratio
    p["motor"]["nom_climb_e_pow_per_nac"]["units"] = "kW"
    p["nacelle"]["electric_mcp"]["value"] = rating * units_per_nac
    p["nacelle"]["electric_mcp"]["units"] = "kW"
    return {"aircraft_kw": float(aircraft_kw), "motor_rating_kw": rating,
            "units_per_nac": units_per_nac}


def gradients_at(s_ref_m2, mtop_kw, hold="span", payload_lbm=17100.0,
                 active_turbines=4, tag="watlim_area"):
    """One Atlas WATLIM run. Returns the five climb gradients in percent."""
    from atlas.scenarios.runs.full_ac_sizing.size_aircraft import prepare_ac_data_base
    from atlas.scenarios.runs.full_ac_sizing.size_aircraft_doe import run_watlim_analysis
    from atlas.scenarios.setup_mission.load_ac_data import load_ac_data_from_excel
    from atlas.aerodynamics.empirical_cd_scale import load_empirical_cd_scale_factors

    with _atlas_cwd():
        base = load_ac_data_from_excel(
            filename="atlas/scenarios/setup_mission/ac_data.xlsx",
            bat_filename=("atlas/propulsion/empirical_data/"
                          "MolicelP80X_module210s8p_4grp14_hiOCV_hiIR_xfeed_per_side_260105.xlsx"),
            cell_sheetname="BOL_cell_fct_CRate", config_sheetname="battery_config")
        ac = prepare_ac_data_base(base, payload_lbm)
        geom = _set_wing(ac, s_ref_m2, hold)
        pw = _set_power(ac, mtop_kw)
        k0, ki = load_empirical_cd_scale_factors(
            "atlas/aerodynamics/data/empirical_cd_scale_factors.csv")
        t0 = time.time()
        res = run_watlim_analysis(
            ac_data=ac, feeder_mode="full_crossfeed", plot_n2=False, turb_type="ACCE",
            case_results_dir=None, case_tag=tag, hybrid_all_phases=True,
            active_turbines=active_turbines, aircraft_go_around_kw=float(mtop_kw),
            return_motor_power_ratings=True, apply_emp_cd_scale_corr=True,
            emp_k_cd0=k0, emp_k_cdi=ki,
            emp_cd_scale_data_csv="atlas/aerodynamics/data/empirical_cd_scale_factors.csv")
    if not res:
        return None
    out = {k: float(res.get(k, float("nan"))) for k in REQUIRED_PCT}
    out.update(geom); out.update(pw)
    out["seconds"] = time.time() - t0
    out["margins"] = {k: out[k] - REQUIRED_PCT[k] for k in REQUIRED_PCT}
    worst = min(out["margins"], key=out["margins"].get)
    out["worst_phase"], out["worst_margin_pct"] = worst, out["margins"][worst]
    return out


def crossovers(rows):
    """Per requirement, the area where it crosses its own limit, by linear bracket.

    Returns one entry per requirement. A requirement met everywhere, or met nowhere,
    says so and reports its best value and the area that produced it -- those two
    outcomes are the interesting ones and must not be reported as a bound.
    """
    a = np.array([r["S_ref_m2"] for r in rows], dtype=float)
    out = {}
    for k, need in REQUIRED_PCT.items():
        v = np.array([r[k] for r in rows], dtype=float)
        sign = np.sign(v - need)
        idx = [i for i in range(len(a) - 1) if sign[i] != sign[i + 1]]
        best = int(v.argmax())
        if not idx:
            out[k] = {"need_pct": need, "crossings_m2": [],
                      "state": "met at every area tested" if (v >= need).all()
                               else "NEVER met at any area tested",
                      "best_pct": float(v[best]), "best_at_m2": float(a[best])}
            continue
        xs = [float(a[i] + (need - v[i]) * (a[i + 1] - a[i]) / (v[i + 1] - v[i]))
              for i in idx]
        out[k] = {"need_pct": need, "crossings_m2": xs, "state": "crosses",
                  "best_pct": float(v[best]), "best_at_m2": float(a[best])}
    return out


def show(r):
    print(f"  S_ref {r['S_ref_m2']:6.2f} m2 ({r['S_ref_m2']*M2_FT2:6.1f} ft2)  "
          f"AR {r['AR']:5.2f}  span {r['span_m']/FT_TO_M:5.1f} ft  "
          f"motor {r['motor_rating_kw']:.1f} kW x{r['units_per_nac']:.0f}   [{r['seconds']:.0f} s]")
    for k, need in REQUIRED_PCT.items():
        m = r["margins"][k]
        print(f"    {k:>11s} {r[k]:6.2f} %  need {need:4.2f}  margin {m:+5.2f}  "
              f"{'PASS' if m >= 0 else 'FAIL'}")
    print(f"    worst: {r['worst_phase']} at {r['worst_margin_pct']:+.2f} %")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mtop-kw", type=float, default=MTOP_KW_NOMINAL)
    ap.add_argument("--hold", choices=("span", "ar"), default="span")
    ap.add_argument("--s-ref", type=float, action="append",
                    help="m2; repeatable. Evaluate these and stop, no bisection.")
    ap.add_argument("--bisect", nargs=2, type=float, metavar=("LO", "HI"),
                    help="m2; bisect for the area where the worst margin is zero")
    ap.add_argument("--sweep", nargs=3, type=float, metavar=("LO", "HI", "STEP"),
                    help="m2; sweep and report a crossover for EACH requirement")
    ap.add_argument("--tol", type=float, default=0.5, help="bisection tolerance, m2")
    ap.add_argument("--max-runs", type=int, default=8)
    a = ap.parse_args()

    print(f"WATLIM climb gradients, power held at {a.mtop_kw:.0f} kW, "
          f"holding {'span at 118 ft (AR follows)' if a.hold=='span' else 'AR (span follows)'}")
    hist = []
    if a.s_ref:
        for s in a.s_ref:
            r = gradients_at(s, a.mtop_kw, a.hold)
            if r is None:
                print(f"  S_ref {s:.2f} m2: run returned nothing")
                continue
            hist.append(r); show(r)
    elif a.bisect:
        lo, hi = a.bisect
        r_lo = gradients_at(lo, a.mtop_kw, a.hold); hist.append(r_lo); show(r_lo)
        r_hi = gradients_at(hi, a.mtop_kw, a.hold); hist.append(r_hi); show(r_hi)
        if r_lo["worst_margin_pct"] >= 0:
            print(f"\nBOUND: the requirements are met at {lo:.2f} m2 already; "
                  f"the boundary is below the range searched.")
        elif r_hi["worst_margin_pct"] < 0:
            print(f"\nBOUND: not met even at {hi:.2f} m2. No boundary in range.")
        else:
            for _ in range(a.max_runs - 2):
                if hi - lo <= a.tol:
                    break
                mid = 0.5 * (lo + hi)
                r = gradients_at(mid, a.mtop_kw, a.hold); hist.append(r); show(r)
                if r["worst_margin_pct"] >= 0:
                    hi = mid
                else:
                    lo = mid
            print(f"\nBOUND: S_ref >= {hi:.2f} m2 = {hi*M2_FT2:.1f} ft2 "
                  f"(bracket {lo:.2f} .. {hi:.2f} m2), set by "
                  f"{hist[-1]['worst_phase']}")
    elif a.sweep:
        lo, hi, step = a.sweep
        for sv in np.arange(lo, hi + 0.5 * step, step):
            r = gradients_at(float(sv), a.mtop_kw, a.hold)
            if r is None:
                continue
            hist.append(r)
            print(f"  S_ref {r['S_ref_m2']:6.1f} m2 ({r['S_ref_m2']*M2_FT2:6.0f} ft2) "
                  f"AR {r['AR']:5.2f} | " +
                  "  ".join(f"{k} {r[k]:5.2f}{'P' if r['margins'][k] >= 0 else 'F'}"
                            for k in REQUIRED_PCT))
        cx = crossovers(hist)
        print("\nCROSSOVER AREA, PER REQUIREMENT")
        bound, by = 0.0, None
        for k, c in cx.items():
            if c["state"] == "crosses":
                for x in c["crossings_m2"]:
                    print(f"  {k:>11s} need {c['need_pct']:4.2f} %  crosses at "
                          f"{x:7.2f} m2 = {x*M2_FT2:7.1f} ft2")
                    if x > bound:
                        bound, by = x, k
            else:
                print(f"  {k:>11s} need {c['need_pct']:4.2f} %  {c['state']} "
                      f"(best {c['best_pct']:.2f} % at {c['best_at_m2']:.0f} m2)")
        unmet = [k for k, c in cx.items() if c["state"].startswith("NEVER")]
        if by:
            print(f"\nBOUND from the requirements that ARE achievable: S_ref >= "
                  f"{bound:.2f} m2 = {bound*M2_FT2:.1f} ft2, set by {by}")
        if unmet:
            print(f"NOT ACHIEVABLE AT ANY AREA at {a.mtop_kw:.0f} kW: {', '.join(unmet)}. "
                  f"No wing area fixes those; power does.")
        out_cx = cx
    else:
        ap.error("give --s-ref, --sweep or --bisect")

    dst = os.path.join(LOGS, f"watlim_area_{int(a.mtop_kw)}kw_{a.hold}.json")
    os.makedirs(LOGS, exist_ok=True)
    json.dump({"mtop_kw": a.mtop_kw, "hold": a.hold, "span_ft": SPAN_FT,
               "required_pct": REQUIRED_PCT, "runs": hist,
               "crossovers": locals().get("out_cx")}, open(dst, "w"), indent=2)
    print(f"wrote {dst}")
