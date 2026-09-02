"""Root chord sweep, 105 -> 120 in (queued item 2 in HANDOFF.md).

The inboard box margin on the as-built wing is ~1.8% (103.2 in needed against
105.0 in built), which is too tight to be comfortable and too tight to be
coincidence -- the wing was sized to it. The question is what buying margin
costs.

The kink is what makes the question interesting. Region B is rebuilt as the
straight taper from the A|B breakpoint to the winglet junction, so widening
region A does NOT propagate to the junction: the junction chord is held and the
extra chord is absorbed by region B's taper. A uniform chord scale -- which is
what `wingbox_pct` does under the "root_le_fixed" rule -- cannot express this,
because it moves every station together.

At each root chord the outboard planform is RE-OPTIMIZED under wing 2's box
constraints (full OAS, fixed lift at MTOW, span pinned), so the reported drag is
the best that root chord can do, not the drag of a frozen shape.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))
sys.path.insert(0, os.path.dirname(_HERE))

from studies.vsp_planform import config  # noqa: E402
from studies.vsp_planform.mesh import N_CHORDWISE, half_mesh, resample, spanwise_stations  # noqa: E402
from studies.vsp_planform.param import baseline_planform, reloft_region_a  # noqa: E402
from studies.vsp_planform.degen_csv import DegenStick  # noqa: E402
from studies.vsp_planform.regions import detect_regions  # noqa: E402
import studies.vsp_planform.run_opt as ro  # noqa: E402
from studies.vsp_planform.run_opt import POINT, trim_alpha  # noqa: E402

import wing2_oas as w2  # noqa: E402

ROOT_CHORDS_IN = [105.0, 108.0, 111.0, 114.0, 117.0, 120.0]

OUT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs", "root_chord_sweep.json")

q = 0.5 * config.RHO * config.V_MS**2


def widen_inboard(mesh, stick, y_a_in, k, p0, ref_axis_pos=0.25, scale=None):
    """Scale every chord inboard of ``y_a_in`` by ``k``, holding the spar line.

    Deliberately the same transform ``reloft_region_a`` applies, with a step
    factor in place of its taper factor: ScaleX about the reference axis, then
    ShearX so ``x_le + p0 * c`` -- the line the whole parameterization is built
    around -- stays put. Thickness and twist ride along as ratios. Outboard
    stations are untouched; region B is rebuilt afterwards to rejoin them.
    """
    if scale is None:
        scale = config.SCALE

    le_s = np.asarray(stick.le, dtype=float)
    te_s = np.asarray(stick.te, dtype=float)
    y_s = np.abs(le_s[:, 1])
    cx_s = te_s[:, 0] - le_s[:, 0]

    step = lambda y: np.where(y <= y_a_in + 1e-9, k, 1.0)

    mesh = np.asarray(mesh, dtype=float)
    y_mesh = np.abs(mesh[0, :, 1]) / scale
    f_mesh = step(y_mesh)
    cx0_mesh = mesh[-1, :, 0] - mesh[0, :, 0]
    ref_axis = ref_axis_pos * mesh[-1] + (1.0 - ref_axis_pos) * mesh[0]
    mesh_new = np.einsum("ijk,j->ijk", mesh - ref_axis, f_mesh) + ref_axis
    mesh_new[:, :, 0] += (p0 - ref_axis_pos) * cx0_mesh * (1.0 - f_mesh)

    f_s = step(y_s)
    lex_new = le_s[:, 0] + p0 * cx_s * (1.0 - f_s)
    z_ref = ref_axis_pos * te_s[:, 2] + (1.0 - ref_axis_pos) * le_s[:, 2]

    columns = dict(stick.columns)
    columns["lex"] = lex_new
    columns["tex"] = lex_new + f_s * cx_s
    columns["lez"] = z_ref + (le_s[:, 2] - z_ref) * f_s
    columns["tez"] = z_ref + (te_s[:, 2] - z_ref) * f_s
    columns["chord"] = np.asarray(stick.chord, dtype=float) * f_s
    return mesh_new, DegenStick(num_secs=stick.num_secs, columns=columns)


def build_case(root_chord_in):
    """Baseline re-lofted to wing 2's breakpoint, with region A at the target chord."""
    mesh0, stick0 = half_mesh(config.BASELINES[w2.BASELINE])
    regions0 = detect_regions(stick0, config.REGION_OVERRIDES.get(w2.BASELINE))
    p0 = baseline_planform(stick0, regions0, name=w2.BASELINE)

    # 1. Move the breakpoint to the inboard nacelle (snaps to a native section).
    mesh1, stick1, regions1 = reloft_region_a(mesh0, stick0, regions0, w2.REGION_A_END_IN, p0)
    y_a = abs(regions1.y_a_end)
    p1 = baseline_planform(stick1, regions1, name=w2.BASELINE)

    # 2. Widen region A to the target root chord.
    c_root0 = float(np.interp(0.0, np.abs(stick1.le[:, 1]), stick1.te[:, 0] - stick1.le[:, 0]))
    k = root_chord_in / c_root0
    mesh2, stick2 = widen_inboard(mesh1, stick1, y_a, k, p1["wingbox_pct"])

    # 3. Rebuild region B so the widened root rejoins the untouched junction.
    mesh3, stick3, regions3 = reloft_region_a(mesh2, stick2, regions1, y_a, p1)
    planform0 = baseline_planform(stick3, regions3, name=w2.BASELINE)

    y_new = spanwise_stations(mesh3[0, :, 1], config.N_SPANWISE_HALF, regions3.y_c_start * config.SCALE)
    mesh, _ = resample(mesh3, y_new, N_CHORDWISE)
    return mesh, stick3, regions3, planform0, c_root0, k


if __name__ == "__main__":
    w2.apply_wing2_box()

    print("=" * 84)
    print("Root chord sweep, full OAS, outboard re-optimized at each point")
    print(f"box: {w2.WIDTH_STATIONS}, spar {w2.REAR_SCHEDULE}, front {w2.FRONT_PCT}")
    print("=" * 84)

    results = []
    for c_root in ROOT_CHORDS_IN:
        mesh, stick, regions, planform0, c_root0, k = build_case(c_root)
        y_c_in = regions.y_c_start

        prob, _ = ro.build_problem(w2.BASELINE, mesh, stick, regions, planform0)
        prob.run_model()
        s0 = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
        alpha0 = trim_alpha(prob, w2.W / (q * s0))
        ro.add_optimization(prob, "plan_l", mesh, planform0, s0, mode="fixed_lift", weight=w2.W)
        prob.set_val("alpha", alpha0, units="deg")
        prob.run_model()
        prob.run_driver()

        r = w2.evaluate(prob, y_c_in)
        r["root_chord_target_in"] = c_root
        r["root_chord_baseline_in"] = c_root0
        r["scale_k"] = k
        r["success"] = bool(prob.driver.result.success)
        r["exit_status"] = str(prob.driver.result.exit_status)
        results.append(r)

        print(f"\n  root chord {c_root:.1f} in  (k = {k:.4f} on {c_root0:.2f} in)")
        print(f"    S_ref {r['S_ref']:7.3f} m^2   CL {r['CL']:.4f}   drag {r['drag_N']:9.1f} N   [{r['exit_status']}]")
        print(f"    junction chord {r['junction_chord_in']:6.2f} in")
        for (y_in, req), c, w, m in zip(w2.WIDTH_STATIONS, r["station_chord_in"], r["box_width_in"], r["box_margin_in"]):
            print(f"      y={y_in:6.1f}  chord {c:7.2f}\"  box {w:6.2f}\"  req {req:5.1f}\"  margin {m:+6.2f}\"")

    base = results[0]["drag_N"]
    print("\n" + "=" * 84)
    print(f"  {'root in':>8} {'S_ref m2':>9} {'drag N':>10} {'vs 105':>8} {'junction':>9} {'min margin':>11}")
    for r in results:
        mm = min(r["box_margin_in"])
        print(
            f"  {r['root_chord_target_in']:8.1f} {r['S_ref']:9.3f} {r['drag_N']:10.1f} "
            f"{r['drag_N'] / base - 1:+7.2%} {r['junction_chord_in']:9.2f} {mm:+11.2f}"
        )
    print("=" * 84)

    with open(OUT, "w") as fh:
        json.dump({"cases": results, "meta": {"weight_N": w2.W, "width_stations": w2.WIDTH_STATIONS}}, fh, indent=2)
    print(f"\n  wrote {OUT}")
