"""A LOWER BOUND ON WING AREA, from the Atlas field-length and WAT tables.

WHY A BOUND AND NOT A COUPLED CONSTRAINT. Take-off field length, landing field
length and the WAT limit depend on (TOW, S_ref, MTOP, CLmax). Not one of those
comes out of the OpenAeroStruct solve: the weight is fixed at MTOW, the power is
held at its nominal value, and CLmax comes from the flap geometry. So the smallest
admissible S_ref is a CONSTANT for a given flap setting, and it is solved once here
rather than re-evaluated inside every optimizer iteration. Nothing is lost by that
-- there is no feedback path to lose.

WHERE THE NUMBERS COME FROM. Three tables, all from Atlas, all read through Atlas's
own loaders so the grid completion and axis padding are not re-implemented here:

  TOFL   atlas.mission.takeoff_landing_lookup.TOFLLookupTable
         TOFL_ft(MTOW_kg, Sref_m2, MTOP_kW, CLmax)
  LFL    atlas.mission.takeoff_landing_lookup.LFLLookupTable
         LFL_ft(MTOW_kg, Sref_m2, CLmax)
  WAT    derived from the same take-off sweep CSV. Each row carries WAT_lb beside
         the MTOW_lb it was run at, so the WAT-limited weight for a cell is the
         largest MTOW in that cell that satisfies MTOW <= WAT_lb.

CLmax may be taken from the flap schedule, or estimated from the flap PLANFORM with
atlas.aerodynamics.CL_max_est.WingCLmaxEstimateGroup, which turns flap span and
chord fractions into a flap area and then into a CLmax increment. Use --flap-span
and --flap-chord for that.

READ THE LIMITS OF THE TABLES BEFORE USING THE ANSWER. They are stated in full by
--report, and three of them change what the answer means:

  * The TOFL CLmax axis holds only flap 15 and flap 25, CLmax 2.5396 to 2.8365.
    A landing-flap CLmax is outside it and extrapolates.
  * The LFL CLmax axis spans 3.0144 to 3.0164. That is one landing CLmax padded to
    three points, so LFL is a function of weight and area alone. An estimated CLmax
    cannot drive it, and this script refuses to pretend otherwise.
  * MTOW 86,000 lb is 39,008.9 kg, which is the top edge of the weight axis of both
    tables. The answer is an edge value, not an interpolated one.

Writes out/logs/field_length_bound.json. The OAS side reads S_ref_min_ft2 from it.
"""

import argparse
import contextlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import brentq

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")
LB_TO_KG = 0.45359237
M2_FT2 = 10.7639104
HP_TO_KW = 0.7457

# THE FIELD LENGTH REQUIREMENT FOR THIS PROGRAM: 6000 ft, take-off and landing
# alike (user, 2026-09-01). It is NOT the pair Atlas defaults to. size_aircraft.py
# is written against 4200 ft take-off and 4000 ft landing (its docstring, and the
# defaults at size_aircraft.py:1160), which is a different and much harder
# requirement -- it puts the bound at 918 ft2 and makes every arc in this study
# infeasible, where 6000 ft puts it at 714 ft2 and none of them are close to it.
# Anyone comparing this bound with an Atlas sizing run must check which pair it used.
TOFL_LIMIT_FT = 6000.0
LFL_LIMIT_FT = 6000.0
MTOW_LB = 86000.0
MTOP_KW_NOMINAL = 1400.0


@contextlib.contextmanager
def _atlas_cwd():
    """Atlas loads its CSVs by a path relative to the repo root, so go there.

    Not a patch of Atlas: the loaders cache on a class attribute, so one visit per
    process is enough and the tables are usable from anywhere afterwards.
    """
    import atlas
    root = os.path.dirname(os.path.dirname(os.path.abspath(atlas.__file__)))
    prev = os.getcwd()
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(prev)


def load_tables():
    """The three interpolators, plus the axes so the caller can report their range."""
    from atlas.mission.takeoff_landing_lookup import TOFLLookupTable, LFLLookupTable
    with _atlas_cwd() as root:
        TOFLLookupTable._load_training_data()
        LFLLookupTable._load_training_data()
        csv = os.path.join(root, "atlas", "mission", "data",
                           "TO_power_sweep_combined_unique_wat_filtered.csv")
        df = pd.read_csv(csv, low_memory=False)

    T, L = TOFLLookupTable, LFLLookupTable
    tofl = RGI((T._mtow_kg, T._Sref_m2, T._mtop_kW, T._clmax), T._tofl_matrix,
               method="linear", bounds_error=False, fill_value=None)
    lfl = RGI((L._mtow_kg, L._Sref_m2, L._clmax), L._lfl_matrix,
              method="linear", bounds_error=False, fill_value=None)

    # WAT. Each row is one (MTOW, Sref, MTOP, flap) case and carries the WAT weight
    # it was judged against, so the limit for a cell is the heaviest MTOW in it that
    # met its own WAT. The MTOW axis of the sweep is coarse -- 7 values -- so this
    # is quantized to those, and it is a floor on the true limit, never above it.
    for c in ("WAT_lb", "MTOW_lb", "MTOP_HP", "Sref_ft2", "flap_deg"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ok = df[df.MTOW_lb <= df.WAT_lb]
    g = (ok.groupby(["Sref_ft2", "MTOP_HP", "flap_deg"])["MTOW_lb"].max().reset_index()
           .rename(columns={"MTOW_lb": "WAT_limit_lb"}))
    return {"tofl": tofl, "lfl": lfl, "wat_table": g,
            "tofl_axes": {"mtow_kg": T._mtow_kg, "sref_m2": T._Sref_m2,
                          "mtop_kw": T._mtop_kW, "clmax": T._clmax},
            "lfl_axes": {"mtow_kg": L._mtow_kg, "sref_m2": L._Sref_m2,
                         "clmax": L._clmax}}


def clmax_from_flap_planform(s_ref_m2, span_m, taper, flap_span_frac, flap_chord_frac,
                             flap_angle_deg, cl_max_clean, toverc_pct, delta_y,
                             mach, inbd_start_frac=0.10, gap_frac=0.03):
    """CLmax from the flap PLANFORM, through Atlas's DATCOM chain.

    The flap is two trapezoids per half wing, and this places them from fractions of
    the semi-span: inboard flap from ``inbd_start_frac`` to the start of the gap, and
    outboard flap from the far side of the gap out to ``flap_span_frac``.

    CAUTION. The Atlas chain builds the flap area on a SINGLE TRAPEZOID wing,
    root_chord = 2*S_ref/(span*(1+taper)). The wings in this study are three-region
    planforms with a constant-chord inboard section, so the flap area is an
    approximation and the CLmax that follows is one too.
    """
    import openmdao.api as om
    from atlas.aerodynamics.CL_max_est import WingCLmaxEstimateGroup

    semi_m = 0.5 * span_m
    y_in_0 = inbd_start_frac * semi_m
    y_in_1 = (flap_span_frac - gap_frac) * semi_m * 0.55 + y_in_0 * 0.45
    y_out_0 = y_in_1 + gap_frac * semi_m

    p = om.Problem(reports=False)
    ivc = om.IndepVarComp()
    ivc.add_output("ac|geom|wing|S_ref", s_ref_m2, units="m**2")
    ivc.add_output("ac|geom|wing|span", span_m, units="m")
    ivc.add_output("ac|geom|wing|taper", taper)
    ivc.add_output("ac|geom|wing|y_inbd_flp_inbd", y_in_0, units="m")
    ivc.add_output("ac|geom|wing|y_inbd_flp_outbd", y_in_1, units="m")
    ivc.add_output("ac|geom|wing|y_outbd_flp_inbd", y_out_0, units="m")
    ivc.add_output("ac|geom|wing|outbd_span_ratio", flap_span_frac)
    ivc.add_output("flap_chord_frac", flap_chord_frac)
    ivc.add_output("toverc", toverc_pct)
    ivc.add_output("flap_angle", flap_angle_deg, units="deg")
    ivc.add_output("cl_max_clean", cl_max_clean)
    ivc.add_output("delta_y", delta_y)
    ivc.add_output("mach", mach)
    ivc.add_output("delta_CLmax_s", 0.0)
    p.model.add_subsystem("ivc", ivc, promotes=["*"])
    p.model.add_subsystem("clmax", WingCLmaxEstimateGroup(), promotes=["*"])
    with _atlas_cwd():
        p.setup()
        p.run_model()
    return {"CLmax": float(p.get_val("CL_max")[0]),
            "S_wf_m2": float(p.get_val("S_wf", units="m**2")[0]),
            "delta_CLmax_f": float(p.get_val("delta_CLmax_f")[0])}


def s_ref_min(tab, mtow_lb, mtop_kw, clmax_to, clmax_ld, tofl_limit, lfl_limit,
              lo=55.0, hi=140.0):
    """The smallest S_ref that meets each field-length requirement, m^2.

    Every answer is labelled INSIDE or OUTSIDE the table. That distinction is not
    decoration here. The TOFL surface is nearly flat against area at the top of its
    S_ref axis -- 4621 ft at 90 m2 against 4607 ft at 95 m2 at CLmax 2.54 -- so past
    95 m2 the interpolator extrapolates a nearly horizontal surface and the root
    moves enormously for a small change in CLmax. Measured: CLmax 2.68 gives 115 m2
    and CLmax 2.74 gives 89 m2. Neither is a wing area. Both are the slope of an
    extrapolation.
    """
    mk = mtow_lb * LB_TO_KG
    f_to = lambda s: float(tab["tofl"]([[mk, s, mtop_kw, clmax_to]])[0]) - tofl_limit
    f_ld = lambda s: float(tab["lfl"]([[mk, s, clmax_ld]])[0]) - lfl_limit

    s_max_to = float(tab["tofl_axes"]["sref_m2"].max())
    s_max_ld = float(tab["lfl_axes"]["sref_m2"].max())
    c_ax = tab["tofl_axes"]["clmax"]

    def solve(f, limit, s_axis_max):
        a, b = f(lo), f(hi)
        if a <= 0.0:
            return lo, "met at the smallest area searched", True
        if b > 0.0:
            return None, (f"not met anywhere up to {hi:.0f} m2 "
                          f"({b + limit:.0f} ft there)"), False
        root = brentq(f, lo, hi)
        inside = root <= s_axis_max + 1e-9
        return root, ("solved, inside the table" if inside else
                      f"EXTRAPOLATED -- past the {s_axis_max:.0f} m2 top of the "
                      f"S_ref axis, so it is not a usable bound"), inside

    s_to, n_to, in_to = solve(f_to, tofl_limit, s_max_to)
    s_ld, n_ld, in_ld = solve(f_ld, lfl_limit, s_max_ld)
    cl_in = float(c_ax.min()) - 1e-9 <= clmax_to <= float(c_ax.max()) + 1e-9
    if not cl_in:
        n_to += (f"; take-off CLmax {clmax_to:.4f} is outside the table axis "
                 f"{c_ax.min():.4f}..{c_ax.max():.4f}")
        in_to = False
    return {"TOFL": {"s_ref_m2": s_to, "note": n_to, "limit_ft": tofl_limit,
                     "trustworthy": bool(in_to)},
            "LFL": {"s_ref_m2": s_ld, "note": n_ld, "limit_ft": lfl_limit,
                    "trustworthy": bool(in_ld)}}


def wat_limit_lb(tab, mtop_kw, s_ref_ft2, flap_deg=25.0):
    """The WAT-limited weight at this power and area, lb. Nearest cell, no interpolation."""
    g = tab["wat_table"]
    g = g[np.isclose(g.flap_deg, flap_deg)]
    if g.empty:
        return None
    hp = mtop_kw / HP_TO_KW
    i = ((g.MTOP_HP - hp).abs() / hp + (g.Sref_ft2 - s_ref_ft2).abs() / s_ref_ft2).idxmin()
    r = g.loc[i]
    return {"WAT_limit_lb": float(r.WAT_limit_lb), "at_MTOP_kW": float(r.MTOP_HP * HP_TO_KW),
            "at_Sref_ft2": float(r.Sref_ft2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mtow-lb", type=float, default=MTOW_LB)
    ap.add_argument("--mtop-kw", type=float, default=MTOP_KW_NOMINAL)
    ap.add_argument("--tofl-ft", type=float, default=TOFL_LIMIT_FT)
    ap.add_argument("--lfl-ft", type=float, default=LFL_LIMIT_FT)
    ap.add_argument("--clmax-to", type=float, default=None,
                    help="take-off CLmax; default the flap-25 table value")
    ap.add_argument("--flap-span", type=float, default=None,
                    help="flap outboard span fraction; switches CLmax to the estimator")
    ap.add_argument("--flap-chord", type=float, default=0.35)
    ap.add_argument("--flap-angle", type=float, default=25.0)
    ap.add_argument("--report", action="store_true", help="print the table axes")
    a = ap.parse_args()

    tab = load_tables()
    ax_t, ax_l = tab["tofl_axes"], tab["lfl_axes"]
    clmax_to = a.clmax_to if a.clmax_to is not None else float(ax_t["clmax"].max())
    clmax_ld = float(np.median(ax_l["clmax"]))
    est = None

    if a.flap_span is not None:
        # A representative planform for the estimator. The study's own S_ref and span
        # are the point of interest, so they are the ones passed.
        est = clmax_from_flap_planform(
            s_ref_m2=80.9, span_m=118.0 * 0.3048, taper=0.44,
            flap_span_frac=a.flap_span, flap_chord_frac=a.flap_chord,
            flap_angle_deg=a.flap_angle, cl_max_clean=2.05, toverc_pct=24.6,
            delta_y=5.1, mach=0.2)
        clmax_to = est["CLmax"]

    print(f"MTOW {a.mtow_lb:,.0f} lb ({a.mtow_lb*LB_TO_KG:,.1f} kg)   "
          f"MTOP {a.mtop_kw:,.0f} kW   flap {a.flap_angle:.0f} deg")
    print(f"take-off CLmax {clmax_to:.4f}"
          + (f"  (estimated from a flap of span fraction {a.flap_span:.3f} and chord "
             f"fraction {a.flap_chord:.3f}, S_wf {est['S_wf_m2']:.2f} m2)" if est else
             "  (from the flap schedule)"))
    print(f"landing CLmax  {clmax_ld:.4f}  (the LFL table carries one landing CLmax)")

    if a.report:
        print("\nTABLE AXES -- an answer outside these is an extrapolation")
        for nm, ax in (("TOFL", ax_t), ("LFL", ax_l)):
            for k, v in ax.items():
                print(f"  {nm:5s} {k:9s} {len(v):3d} points  {v.min():10.4f} .. {v.max():10.4f}")

    res = s_ref_min(tab, a.mtow_lb, a.mtop_kw, clmax_to, clmax_ld, a.tofl_ft, a.lfl_ft)
    print("\nSMALLEST WING AREA THAT MEETS EACH REQUIREMENT")
    binding, s_min = None, 0.0
    for nm, r in res.items():
        if r["s_ref_m2"] is None:
            print(f"  {nm:5s} <= {r['limit_ft']:.0f} ft   NO SOLUTION -- {r['note']}")
            continue
        print(f"  {nm:5s} <= {r['limit_ft']:.0f} ft   S_ref >= {r['s_ref_m2']:7.2f} m2 "
              f"= {r['s_ref_m2']*M2_FT2:7.1f} ft2   ({r['note']})")
        if r["s_ref_m2"] > s_min:
            binding, s_min = nm, r["s_ref_m2"]
    trust = all(r["trustworthy"] for r in res.values() if r["s_ref_m2"] is not None)

    w = wat_limit_lb(tab, a.mtop_kw, s_min * M2_FT2)
    if w:
        ok = w["WAT_limit_lb"] >= a.mtow_lb - 1.0
        print(f"\n  WAT   at {w['at_MTOP_kW']:.0f} kW and {w['at_Sref_ft2']:.0f} ft2: "
              f"{w['WAT_limit_lb']:,.0f} lb   {'ok, not binding' if ok else 'BINDING'} "
              f"against MTOW {a.mtow_lb:,.0f} lb")

    print(f"\nBOUND: S_ref >= {s_min:.2f} m2 = {s_min*M2_FT2:.1f} ft2, set by {binding}")
    if not trust:
        print("  DO NOT USE THIS BOUND. It rests on an extrapolation, named above.")
    out = {"mtow_lb": a.mtow_lb, "mtop_kw": a.mtop_kw,
           "tofl_limit_ft": a.tofl_ft, "lfl_limit_ft": a.lfl_ft,
           "clmax_takeoff": clmax_to, "clmax_landing": clmax_ld,
           "clmax_estimate": est, "flap_angle_deg": a.flap_angle,
           "per_requirement": {k: {kk: vv for kk, vv in v.items()} for k, v in res.items()},
           "wat": w, "binding": binding, "trustworthy": bool(trust),
           "S_ref_min_m2": s_min, "S_ref_min_ft2": s_min * M2_FT2}
    os.makedirs(LOGS, exist_ok=True)
    dst = os.path.join(LOGS, "field_length_bound.json")
    json.dump(out, open(dst, "w"), indent=2)
    print(f"wrote {dst}")
