"""OAS run of the wing 2 design point (queued item 3 in HANDOFF.md).

Every wing 2 number in the study so far came from the simplified model (fixed-e
induced drag + 2D section Cd on S_ref), which runs ~9% optimistic against OAS.
This puts the same design point through the full VLM.

It was blocked because ``param.py`` only knew about a box with one rear-spar
fraction checked at one station; wing 2's rear spar kinks from 0.750c to 0.499c
and has to be checked at four. That is now a schedule
(``config.WINGBOX_REAR_SCHEDULE``) and a vector of stations
(``config.WINGBOX_WIDTH_STATIONS``), so the design point can be built.

Three cases, all at MTOW with the span pinned, all trimmed to the same lift:

A  baseline      the as-built ConstChord wing
B  design point  region A shortened to the inboard nacelle, junction chord 66 in
C  optimized     min drag under wing 2's box constraints, to see where OAS puts
                 the design when it is allowed to choose

Depth is NOT computed here. It needs the section's thickness at the local spar
fraction, and the only retention ratio the study has measured is 0.527 at 0.75c
for the as-built section. Where the spar sits at 0.75c that gives a depth
directly; at the junction the spar is at 0.499c and the script reports the
retention the section would need instead of inventing one.
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

_HERE = os.path.abspath(__file__)
# .../studies/vsp_planform/out/scripts/wing2_oas.py -> the repository root.
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

from studies.vsp_planform import config  # noqa: E402
from studies.vsp_planform.mesh import N_CHORDWISE, half_mesh, resample, spanwise_stations  # noqa: E402
from studies.vsp_planform.param import baseline_planform, rear_spar_fraction, reloft_region_a  # noqa: E402
import studies.vsp_planform.run_opt as ro  # noqa: E402
from studies.vsp_planform.run_opt import POINT, load_baseline, trim_alpha  # noqa: E402
from studies.vsp_planform.regions import detect_regions  # noqa: E402

BASELINE = "const_chord"

# MTOW, newtons. Same figure the decomposition and every other full-OAS run used.
W = 382547.0

# The wing 2 design point, out/logs/wing2_design_point.json.
REGION_A_END_IN = 176.0  # the inboard nacelle
FRONT_PCT = 0.12
REAR_SCHEDULE = ((356.0, 0.750), (674.9, 0.499))

# Required box width at the winglet junction, inches. User-set on 2026-08-15,
# replacing the 25 in soft pick the study had been carrying. This one number
# sets the junction chord and therefore the whole design point -- see below.
JUNCTION_BOX_IN = 20.0

# (y in inches, required box width in inches).
WIDTH_STATIONS = ((100.0, 65.0), (176.0, 65.0), (356.0, 55.0), (674.9, JUNCTION_BOX_IN))

# The junction chord is DERIVED, not picked. The box there is
# (rear - front) * chord with the spar schedule pinned, so the required width
# fixes the chord exactly. The old 66.0 in was this same calculation at 25 in;
# hardcoding it is what made the requirement look like a free choice when it is
# the single most load-bearing input in the study.
JUNCTION_CHORD_IN = JUNCTION_BOX_IN / (REAR_SCHEDULE[-1][1] - FRONT_PCT)

# As-built section: fraction of max thickness still present at 0.75c.
T75_RATIO_AS_BUILT = 0.527

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "logs", "wing2_oas.json")

q = 0.5 * config.RHO * config.V_MS**2


def apply_wing2_box():
    """Point the shared config at wing 2's box before anything builds a model."""
    config.WINGBOX_FRONT_PCT = FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = REAR_SCHEDULE
    config.WINGBOX_WIDTH_STATIONS = WIDTH_STATIONS


def load_relofted(name, y_a_new_in):
    """Parse a baseline, move its A|B breakpoint, and resample the result.

    Mirrors ``run_opt.load_baseline``, with the re-loft slotted in between
    parsing and resampling so everything downstream -- region detection,
    ``baseline_planform``, the spanwise station distribution -- reads the wing
    that will actually be flown.
    """
    mesh_native, stick = half_mesh(config.BASELINES[name])
    regions = detect_regions(stick, config.REGION_OVERRIDES.get(name))
    planform0 = baseline_planform(stick, regions, name=name)

    mesh_native, stick, regions = reloft_region_a(mesh_native, stick, regions, y_a_new_in, planform0)
    planform0 = baseline_planform(stick, regions, name=name)

    y_new = spanwise_stations(mesh_native[0, :, 1], config.N_SPANWISE_HALF, regions.y_c_start * config.SCALE)
    mesh, _ = resample(mesh_native, y_new, N_CHORDWISE)
    return mesh, stick, regions, planform0


def junction_chord_in(prob, y_c_in):
    """Chord at the B|C junction of the current design, inches."""
    mesh = prob.get_val("wing.mesh", units="m")
    y = np.abs(mesh[0, :, 1])
    chord = mesh[-1, :, 0] - mesh[0, :, 0]
    return float(np.interp(y_c_in * config.SCALE, y, chord)) / config.SCALE


def evaluate(prob, y_c_in):
    """Trim to MTOW and report the full drag breakdown plus the box widths."""
    prob.run_model()
    s_ref = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    alpha = trim_alpha(prob, W / (q * s_ref))
    s_ref = float(prob.get_val(f"{POINT}.wing.S_ref")[0])

    def cd(key):
        return float(prob.get_val(f"{POINT}.wing_perf.{key}")[0])

    widths = prob.get_val("wingbox_width", units="m") / config.SCALE
    chords = prob.get_val("station_chord", units="m") / config.SCALE
    required = np.array([w for _, w in WIDTH_STATIONS], dtype=float)

    return {
        "alpha": alpha,
        "S_ref": s_ref,
        "CL": cd("CL"),
        "CD": cd("CD"),
        "induced_N": q * s_ref * cd("CDi"),
        "viscous_N": q * s_ref * cd("CDv"),
        "wave_N": q * s_ref * cd("CDw"),
        "drag_N": q * s_ref * (cd("CDi") + cd("CDv") + cd("CDw")),
        "taper_B": float(prob.get_val("wing.taper_B")[0]),
        "wingbox_pct": float(prob.get_val("wing.wingbox_pct")[0]),
        "junction_chord_in": junction_chord_in(prob, y_c_in),
        "box_width_in": widths.tolist(),
        "box_margin_in": (widths - required).tolist(),
        "station_chord_in": chords.tolist(),
        "twist_root": float(prob.get_val("twist_abs", units="deg")[0]),
        "twist_tip": float(prob.get_val("twist_abs", units="deg")[-1]),
        # The full design vector, so the geometry can be rebuilt for plotting
        # without paying for the optimization again.
        "twist_cp": prob.get_val("wing.twist_cp", units="deg").tolist(),
    }


def build(name, y_a_new_in=None):
    """Build the OAS problem for a baseline, optionally with a moved breakpoint."""
    if y_a_new_in is None:
        mesh, stick, regions, planform0, _, _ = load_baseline(name)
    else:
        mesh, stick, regions, planform0 = load_relofted(name, y_a_new_in)
    prob, _ = ro.build_problem(name, mesh, stick, regions, planform0)
    return prob, mesh, stick, regions, planform0


def set_junction_chord(prob, y_c_in, target_in):
    """Solve for the ``taper_B`` that puts the junction chord on target."""

    def resid(lam):
        prob.set_val("wing.taper_B", lam)
        prob.run_model()
        return junction_chord_in(prob, y_c_in) - target_in

    lam = brentq(resid, 0.05, 1.0, xtol=1e-12)
    resid(lam)
    return float(lam)


def report(label, r):
    print(f"\n  {label}")
    print(f"    S_ref {r['S_ref']:8.3f} m^2   CL {r['CL']:.4f} at alpha {r['alpha']:+.3f} deg")
    print(
        f"    drag  {r['drag_N']:8.1f} N  = induced {r['induced_N']:.1f}"
        f" + viscous {r['viscous_N']:.1f} + wave {r['wave_N']:.1f}"
    )
    print(f"    taper_B {r['taper_B']:.5f}, wingbox_pct {r['wingbox_pct']:.5f}")
    print(f"    junction chord {r['junction_chord_in']:.2f} in")
    print(f"    {'station':>10} {'chord':>9} {'rear x/c':>9} {'box':>8} {'req':>7} {'margin':>8}")
    for (y_in, req), c, w, m in zip(WIDTH_STATIONS, r["station_chord_in"], r["box_width_in"], r["box_margin_in"]):
        rear = float(rear_spar_fraction(y_in, REAR_SCHEDULE))
        flag = "" if m >= -1e-9 else "   <-- SHORT"
        print(f"    {y_in:10.1f} {c:8.2f}\" {rear:9.3f} {w:7.2f}\" {req:6.1f}\" {m:+7.2f}\"{flag}")


def depth_note(r):
    """What the as-built section gives, and what the junction would need."""
    print("\n  Rear-spar depth (as-built section, t/c from the loft):")
    for (y_in, _), c in zip(WIDTH_STATIONS, r["station_chord_in"]):
        rear = float(rear_spar_fraction(y_in, REAR_SCHEDULE))
        if abs(rear - 0.75) < 1e-6:
            # Measured retention at 0.75c; t/c at the root of the as-built loft.
            depth = T75_RATIO_AS_BUILT * TOC_AT[y_in] * c
            print(f"    y={y_in:6.1f}  spar 0.750c  t/c {TOC_AT[y_in]:.4f}  depth {depth:5.2f} in")
        else:
            t_max = TOC_AT[y_in] * c
            print(
                f"    y={y_in:6.1f}  spar {rear:.3f}c  max thickness {t_max:5.2f} in  ->"
                f" needs {7.0 / t_max:.3f} of it for 7 in, {6.0 / t_max:.3f} for 6 in"
            )


if __name__ == "__main__":
    apply_wing2_box()

    print("=" * 78)
    print("Wing 2 design point in full OAS, at MTOW, span pinned")
    print("=" * 78)

    # --- A: the as-built baseline, with its own detected regions.
    prob_a, mesh_a, stick_a, regions_a, p0_a = build(BASELINE)
    y_c_in = regions_a.y_c_start
    print(f"  baseline regions: A ends y={regions_a.y_a_end:.1f} in, B|C at y={y_c_in:.1f} in")

    # t/c of the as-built loft at each constraint station, for the depth note.
    y_stick = np.abs(np.asarray(stick_a.le[:, 1], dtype=float))
    TOC_AT = {y_in: float(np.interp(y_in, y_stick, stick_a.toc)) for y_in, _ in WIDTH_STATIONS}

    res = {"A_baseline": evaluate(prob_a, y_c_in)}
    report("A  as-built baseline", res["A_baseline"])

    # --- B: the design point. Region A ends at the inboard nacelle, and the
    # junction chord is grown to 66 in to make the box fit.
    prob_b, mesh_b, stick_b, regions_b, p0_b = build(BASELINE, REGION_A_END_IN)
    y_a_actual = abs(regions_b.y_a_end)
    print(f"\n  region A re-lofted to end at y = {y_a_actual:.1f} in (asked {REGION_A_END_IN})")
    print(f"  re-lofted taper_B0 = {p0_b['taper_B']:.5f}, spar straightness {p0_b['spar_max_dev']:.4f} in")
    lam = set_junction_chord(prob_b, y_c_in, JUNCTION_CHORD_IN)
    print(f"  junction chord {JUNCTION_CHORD_IN} in reached at taper_B = {lam:.6f}")
    res["B_design_point"] = evaluate(prob_b, y_c_in)
    report("B  wing 2 design point", res["B_design_point"])
    depth_note(res["B_design_point"])

    # --- C: let OAS optimize under wing 2's box constraints.
    prob_c, mesh_c, stick_c, regions_c, p0_c = build(BASELINE, REGION_A_END_IN)
    prob_c.run_model()
    s0 = float(prob_c.get_val(f"{POINT}.wing.S_ref")[0])
    alpha0 = trim_alpha(prob_c, W / (q * s0))
    # add_optimization keys the width constraint off the baseline name; wing 2 is
    # a ConstChord derivative, so ask for it explicitly.
    ro.add_optimization(prob_c, "plan_l", mesh_c, p0_c, s0, mode="fixed_lift", weight=W)
    prob_c.set_val("alpha", alpha0, units="deg")
    prob_c.run_model()
    prob_c.run_driver()
    res["C_optimized"] = evaluate(prob_c, y_c_in)
    res["C_optimized"]["success"] = bool(prob_c.driver.result.success)
    res["C_optimized"]["exit_status"] = str(prob_c.driver.result.exit_status)
    report("C  optimized under the wing 2 box constraints", res["C_optimized"])
    print(f"    driver: {res['C_optimized']['exit_status']}")

    base = res["A_baseline"]["drag_N"]
    print("\n" + "=" * 78)
    for k in ("A_baseline", "B_design_point", "C_optimized"):
        print(f"  {k:<16} drag {res[k]['drag_N']:9.1f} N   {res[k]['drag_N'] / base - 1:+7.2%} vs baseline")
    print("=" * 78)
    print("  All three are full-OAS numbers and comparable to each other. They are")
    print("  NOT comparable to the simplified-model figures in README.md.")

    res["_meta"] = {
        "weight_N": W,
        "region_a_end_in": y_a_actual,
        "junction_chord_target_in": JUNCTION_CHORD_IN,
        "front_pct": FRONT_PCT,
        "rear_schedule": REAR_SCHEDULE,
        "width_stations": WIDTH_STATIONS,
        "model": "full OAS VLM (CDi + CDv + CDw)",
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n  wrote {OUT}")
