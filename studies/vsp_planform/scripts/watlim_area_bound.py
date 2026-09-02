"""The wing area at which the fixed-point climb requirements cross, at held power.

THE ENTRY POINT MATTERS, AND AN EARLIER VERSION OF THIS FILE USED THE WRONG ONE.
Atlas has two WATLIM paths and they answer different questions:

  run_watlim_analysis          sweeps altitude and DISA and returns the WORST
                               corner. At 1400 kW it reports a 2nd segment near
                               0.3 %, and nothing passes at any wing area.
  run_fixed_point_watlim       evaluates ONE stated corner -- 2000 ft, DISA +20
                               for phases 1 to 4, sea-level ISA for the AEO climb
                               rate. This is the baseline the programme quotes,
                               and at 1400 kW and MTOW it passes every gradient.

This module drives the second. Reproduced against run_emotor_sizing_fixed_point.py
at TOW 86,000 lbm and 1400 kW: 3.440 / 3.415 / 4.994 / 3.440 % and 1289 ft/min,
which is that script's own printed output.

THE REQUIREMENTS, from _phase_requirements in that script:

    1  WATLIM 2nd segment   >= 3.0 %       2000 ft, DISA +20
    2  WATLIM 4th segment   >= 1.7 %       2000 ft, DISA +20
    3  Landing              >= 3.2 %       2000 ft, DISA +20
    4  Approach             >= 2.7 %       2000 ft, DISA +20
    5  AEO climb rate       >= 1400 ft/min sea level, ISA, 190 KIAS

Phase 5 is a RATE, not a gradient, and it is the one that fails at MTOW.

POWER IS THE COMMANDED POWER, 1400 kW, passed as fixed_power_hp. It is not the
motor rating: this path commands power directly and bypasses the motor shaft-power
cap, exactly as the baseline run does.

SPAN IS HELD, NOT ASPECT RATIO, because this study pins the span at 118 ft. Wing
area therefore moves the aspect ratio. --hold ar gives Atlas's own convention.

HOW THE AREA IS CHANGED. run_fixed_point_watlim reads its aircraft from an Excel
file and takes S_ref from it, so the override is applied by wrapping that loader
for the duration of one call. The wing is changed consistently -- S_ref, S_plan,
S_trap and the three spans -- because S_ref also sets the stall speed and therefore
the whole WATLIM speed schedule.
"""

import argparse
import contextlib
import json
import os
import sys

import numpy as np

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")
M2_FT2 = 10.7639104
FT_TO_M = 0.3048
SPAN_FT = 118.0
HP_PER_KW = 1.341

BASELINE = dict(altitude_ft=2000.0, disa_degC=20.0, payload_lbm=17100.0,
                watlim_2nd_flap_deg=25, aeo_speed_kias=190.0,
                aeo_climb_rate_target_fpm=1400.0, active_turbines=4,
                gt_cap_level=0.0, gas_turbine_map="ACCE",
                bypass_motor_shaft_power_limit=True)


@contextlib.contextmanager
def _atlas_cwd():
    """Atlas reads its Excel and CSV inputs by paths relative to its repo root."""
    import atlas
    root = os.path.dirname(os.path.dirname(os.path.abspath(atlas.__file__)))
    prev = os.getcwd()
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(prev)


def _wing_override(s_ref_m2, hold, span_ft):
    """Return a function that puts this wing into a freshly loaded aircraft dict.

    All three areas and all three spans move together. Setting S_ref alone leaves
    the aircraft with an area and a span that disagree, and S_ref drives the stall
    speed and therefore the whole WATLIM speed schedule, so the disagreement would
    reach the answer rather than sit unused.
    """
    def apply(ac_data):
        w = ac_data["ac"]["geom"]["wing"]
        s0 = float(w["S_ref"]["value"])
        b0 = float(w["span"]["value"])
        if hold == "span":
            span_m = float(span_ft) * FT_TO_M
            ar = span_m**2 / s_ref_m2
        else:
            ar = float(w["AR"]["value"])
            span_m = float(np.sqrt(s_ref_m2 * ar))
        # SCALE, do not assign. The three areas happen to be equal in this aircraft
        # but the three spans are NOT: span and span_plan are 35.301 m while
        # span_trap is 33.700 m, the trapezoidal reference. Assigning one value to
        # all three clobbered span_trap by 1.6 m and moved the second segment from
        # 3.440 to 3.435 percent. Scaling preserves every ratio and is an exact
        # identity when the wing is unchanged, which is how this was caught.
        ks, kb = s_ref_m2 / s0, span_m / b0
        for k in ("S_ref", "S_plan", "S_trap"):
            if k in w:
                w[k]["value"] = float(w[k]["value"]) * ks
        for k in ("span", "span_plan", "span_trap"):
            if k in w:
                w[k]["value"] = float(w[k]["value"]) * kb
        w["AR"]["value"] = ar
        if isinstance(w.get("AR_plan"), dict):
            w["AR_plan"]["value"] = ar
        return {"S_ref_m2": s_ref_m2, "span_ft": span_m / FT_TO_M, "AR": ar,
                "span_trap_m": float(w["span_trap"]["value"]) if "span_trap" in w else None}
    return apply


@contextlib.contextmanager
def _patched_loader(module, apply):
    """Wrap the module's Excel loader so one call sees the overridden wing.

    A wrapper rather than an edit: run_fixed_point_watlim loads the aircraft itself
    and there is no argument for the wing, so this is the seam. It is restored on
    the way out, so nothing leaks into the next call.
    """
    original = module.load_ac_data_from_excel
    captured = {}

    def loader(*a, **kw):
        ac = original(*a, **kw)
        captured.update(apply(ac))
        return ac

    module.load_ac_data_from_excel = loader
    try:
        yield captured
    finally:
        module.load_ac_data_from_excel = original


def evaluate(s_ref_m2, power_kw=1400.0, tow_lbm=86000.0, hold="span",
             span_ft=SPAN_FT, **over):
    """One fixed-point run. Returns the five phase metrics and their margins."""
    from atlas.scenarios.runs.emotor_sizing import run_emotor_sizing_fixed_point as FP
    kw = dict(BASELINE)
    kw.update(over)
    apply = _wing_override(s_ref_m2, hold, span_ft)
    with _atlas_cwd(), _patched_loader(FP, apply) as geom:
        res = FP.run_fixed_point_watlim(
            mode="fixed_power", fixed_power_hp=float(power_kw) * HP_PER_KW,
            tow_lbm_override=float(tow_lbm), **kw)
    out = {"S_ref_m2": s_ref_m2, "power_kw": power_kw, "tow_lbm": tow_lbm}
    out.update(geom)
    out["phases"] = {}
    for r in res:
        out["phases"][int(r.phase_id)] = {
            "label": r.label, "metric": float(r.metric_value),
            "target": float(r.target_value), "margin": float(r.margin),
            "passes": bool(r.passes_requirement), "name": r.metric_name}
    return out


PHASE_ORDER = (1, 2, 3, 4, 5)


def crossovers(rows):
    """Per phase, the area where the metric crosses its own target."""
    a = np.array([r["S_ref_m2"] for r in rows], dtype=float)
    out = {}
    for pid in PHASE_ORDER:
        v = np.array([r["phases"][pid]["metric"] for r in rows], dtype=float)
        need = rows[0]["phases"][pid]["target"]
        lab = rows[0]["phases"][pid]["label"]
        sign = np.sign(v - need)
        idx = [i for i in range(len(a) - 1) if sign[i] != sign[i + 1]]
        best = int(v.argmax())
        base = {"label": lab, "target": need, "best": float(v[best]),
                "best_at_m2": float(a[best])}
        if idx:
            base["state"] = "crosses"
            base["crossings_m2"] = [
                float(a[i] + (need - v[i]) * (a[i + 1] - a[i]) / (v[i + 1] - v[i]))
                for i in idx]
        else:
            base["state"] = ("met at every area tested" if (v >= need).all()
                             else "NEVER met at any area tested")
            base["crossings_m2"] = []
        out[pid] = base
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--power-kw", type=float, default=1400.0)
    ap.add_argument("--tow-lbm", type=float, default=86000.0)
    ap.add_argument("--hold", choices=("span", "ar"), default="span")
    ap.add_argument("--span-ft", type=float, default=SPAN_FT)
    ap.add_argument("--sweep", nargs=3, type=float, metavar=("LO", "HI", "STEP"),
                    required=True, help="S_ref, m2")
    a = ap.parse_args()

    print(f"Fixed-point WATLIM: {BASELINE['altitude_ft']:.0f} ft, DISA "
          f"+{BASELINE['disa_degC']:.0f} C (phases 1-4), sea-level ISA (phase 5)")
    print(f"commanded power {a.power_kw:.0f} kW, TOW {a.tow_lbm:,.0f} lbm, "
          f"holding {'span at %.0f ft' % a.span_ft if a.hold=='span' else 'AR'}\n")

    rows = []
    hdr = None
    for sv in np.arange(a.sweep[0], a.sweep[1] + 0.5 * a.sweep[2], a.sweep[2]):
        r = evaluate(float(sv), a.power_kw, a.tow_lbm, a.hold, a.span_ft)
        rows.append(r)
        if hdr is None:
            hdr = True
            print(f"  {'S_ref m2':>8s} {'ft2':>6s} {'AR':>5s} | " +
                  " ".join(f"{r['phases'][p]['label'][:11]:>13s}" for p in PHASE_ORDER))
        print(f"  {r['S_ref_m2']:8.1f} {r['S_ref_m2']*M2_FT2:6.0f} {r['AR']:5.2f} | " +
              " ".join(f"{r['phases'][p]['metric']:9.2f}"
                       f"{'P' if r['phases'][p]['passes'] else 'F':>4s}"
                       for p in PHASE_ORDER))

    cx = crossovers(rows)
    print("\nCROSSOVER AREA, PER PHASE")
    bound, by = 0.0, None
    for pid in PHASE_ORDER:
        c = cx[pid]
        if c["state"] == "crosses":
            for x in c["crossings_m2"]:
                print(f"  {c['label'][:22]:>22s} need {c['target']:8.2f}  crosses at "
                      f"{x:7.2f} m2 = {x*M2_FT2:7.1f} ft2")
                if x > bound:
                    bound, by = x, c["label"]
        else:
            print(f"  {c['label'][:22]:>22s} need {c['target']:8.2f}  {c['state']} "
                  f"(best {c['best']:.2f} at {c['best_at_m2']:.0f} m2)")
    if by:
        print(f"\nBOUND: S_ref >= {bound:.2f} m2 = {bound*M2_FT2:.1f} ft2, set by {by}")
    never = [cx[p]["label"] for p in PHASE_ORDER if cx[p]["state"].startswith("NEVER")]
    if never:
        print(f"NOT MET AT ANY AREA at {a.power_kw:.0f} kW and TOW {a.tow_lbm:,.0f} lbm: "
              f"{', '.join(never)}")

    os.makedirs(LOGS, exist_ok=True)
    dst = os.path.join(LOGS, f"watlim_fixedpoint_{int(a.power_kw)}kw_"
                             f"{int(a.tow_lbm)}lbm_{a.hold}.json")
    json.dump({"baseline": BASELINE, "power_kw": a.power_kw, "tow_lbm": a.tow_lbm,
               "hold": a.hold, "span_ft": a.span_ft, "runs": rows,
               "crossovers": {str(k): v for k, v in cx.items()}}, open(dst, "w"), indent=2)
    print(f"wrote {dst}")
