"""Train a surrogate of the five fixed-point climb metrics over (S_ref, CLmax).

WHY A SURROGATE, AND WHY IT IS NOT A SHORTCUT. TOFL and LFL reach an optimizer as
MetaModelStructuredComp because Atlas already trained them that way. The climb
gradients have no such table: run_fixed_point_watlim builds and runs its own
OpenMDAO problem, takes about 15 s, and returns floats with no partials. Inside an
SLSQP loop that is not viable. So the same treatment is applied here -- sample the
real analysis on a grid, then interpolate. The grid is the record; the surrogate
never invents a point outside it, and the component that reads it refuses to
extrapolate.

WHAT THE AXES ARE, AND WHY ONLY TWO. At held power, held TOW and held span, the
five metrics depend on the wing only through its AREA and through the CLmax that
sets the reference speeds. Everything else in the aircraft is fixed.

  S_ref    m2, the wing area
  CLmax25  the flap-25 maximum lift coefficient

CLmax25 IS AN AXIS ONLY BECAUSE THE SPEED SCHEDULE IS PATCHED. Atlas hardcodes four
CLmax values in build_watlim_speed_schedule_mps -- flap 0, 15, 25 and 35 at 1.4651,
2.4922, 2.7836 and 2.9591. They are constants in the function body, not arguments
and not a table, so the only way an estimated CLmax can reach the climb gradients is
to replace them. All four are scaled by the same ratio CLmax25/2.7836, which keeps
the flap schedule's shape and moves its level. That is an assumption, and it is the
main modelling liberty this file takes.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

import watlim_area_bound as WA                                       # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")
CLMAX25_REF = 2.7836          # the value Atlas hardcodes for flap 25
PHASES = (1, 2, 3, 4, 5)


def _patched_speed_schedule(module, clmax25):
    """Scale Atlas's hardcoded flap CLmax ladder to a new flap-25 value."""
    original = module.build_watlim_speed_schedule_mps
    k = float(clmax25) / CLMAX25_REF

    def sched(mass_kg, s_ref, watlim_2nd_flap_deg, aeo_speed_kias=190.0):
        vs = lambda cl: module._stall_speed_mps(cl * k, mass_kg, s_ref)
        return {
            "v_watlim_2nd_mps": 1.13 * (vs(2.4922) if watlim_2nd_flap_deg == 15
                                        else vs(2.7836)),
            "v_watlim_4th_mps": 1.18 * vs(1.4651),
            "v_approach_mps": 1.13 * vs(2.7836),
            "v_ldg_mps": 1.23 * vs(2.9591),
            "v_aeo_mps": float(aeo_speed_kias) * 0.514,
        }
    return original, sched


def sample(s_ref_grid, clmax_grid, power_kw, tow_lbm, hold, span_ft):
    """Run the real analysis at every grid point. Returns the metric tensors."""
    from atlas.scenarios.runs.emotor_sizing import run_emotor_sizing_fixed_point as FP
    out = {p: np.full((len(s_ref_grid), len(clmax_grid)), np.nan) for p in PHASES}
    targets, labels = {}, {}
    t0 = time.time()
    for j, cl in enumerate(clmax_grid):
        orig, sched = _patched_speed_schedule(FP, cl)
        FP.build_watlim_speed_schedule_mps = sched
        try:
            for i, s in enumerate(s_ref_grid):
                r = WA.evaluate(float(s), power_kw, tow_lbm, hold, span_ft)
                for p in PHASES:
                    out[p][i, j] = r["phases"][p]["metric"]
                    targets[p] = r["phases"][p]["target"]
                    labels[p] = r["phases"][p]["label"]
                print(f"  S_ref {s:6.2f} m2  CLmax25 {cl:.4f} | " +
                      " ".join(f"{r['phases'][p]['metric']:8.2f}" for p in PHASES),
                      flush=True)
        finally:
            FP.build_watlim_speed_schedule_mps = orig
    print(f"  {len(s_ref_grid)*len(clmax_grid)} runs in {time.time()-t0:.0f} s")
    return out, targets, labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--power-kw", type=float, default=1400.0)
    ap.add_argument("--tow-lbm", type=float, default=86000.0)
    ap.add_argument("--hold", choices=("span", "ar"), default="span")
    ap.add_argument("--span-ft", type=float, default=WA.SPAN_FT)
    ap.add_argument("--s-ref", nargs=3, type=float, default=(60.0, 100.0, 9),
                    metavar=("LO", "HI", "N"))
    ap.add_argument("--clmax", nargs=3, type=float, default=(2.40, 3.00, 5),
                    metavar=("LO", "HI", "N"))
    a = ap.parse_args()

    s_grid = np.linspace(a.s_ref[0], a.s_ref[1], int(a.s_ref[2]))
    c_grid = np.linspace(a.clmax[0], a.clmax[1], int(a.clmax[2]))
    print(f"Training grid: {len(s_grid)} areas x {len(c_grid)} CLmax = "
          f"{len(s_grid)*len(c_grid)} runs of the real analysis")
    print(f"  S_ref   {s_grid.min():.1f} .. {s_grid.max():.1f} m2")
    print(f"  CLmax25 {c_grid.min():.3f} .. {c_grid.max():.3f}")
    print(f"  power {a.power_kw:.0f} kW, TOW {a.tow_lbm:,.0f} lbm, span {a.span_ft:.0f} ft\n")

    tensors, targets, labels = sample(s_grid, c_grid, a.power_kw, a.tow_lbm,
                                      a.hold, a.span_ft)
    dst = os.path.join(LOGS, f"watlim_surrogate_{int(a.power_kw)}kw_"
                             f"{int(a.tow_lbm)}lbm.npz")
    np.savez(dst, s_ref_m2=s_grid, clmax25=c_grid,
             **{f"phase_{p}": tensors[p] for p in PHASES},
             targets=np.array([targets[p] for p in PHASES]),
             labels=np.array([labels[p] for p in PHASES]),
             power_kw=a.power_kw, tow_lbm=a.tow_lbm, span_ft=a.span_ft, hold=a.hold)
    print(f"wrote {dst}")
    for p in PHASES:
        v = tensors[p]
        print(f"  phase {p} {labels[p][:20]:20s} need {targets[p]:8.2f}  "
              f"range {np.nanmin(v):9.2f} .. {np.nanmax(v):9.2f}")
