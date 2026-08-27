"""The 7 inch spar airfoil swap, through the real OAS model.

Replaces the back-of-envelope estimate (representative section Cd, e-factor
induced drag) with the full VLM plus OAS's own viscous and wave drag.

The airfoil reaches OAS as exactly two scalars, through the Raymer form factor
at ``aerodynamics/viscous_drag.py:103``::

    k_FF = 1.34 * M^0.18 * (1 + 0.6*(t/c)/c_max_t + 100*(t/c)^4)

so ``c_max_t`` (0.3021 as built -> 0.45 for the 66-series) and the ``t_over_c``
distribution are the whole of the swap. The form factor *falls* as ``c_max_t``
rises, so part of the benefit is real physics OAS can see.

Cases, all at MTOW 382,547 N with span pinned at 118 ft:
  1. as-built ConstChord
  2. 66(4)-221 family, junction chord 56.0 in, t/c uniform at 0.21
  3. 66(4)-221 family, junction chord 56.0 in, t/c the minimum that meets 7 in
  4. fx2 family, junction chord 57.2 in, minimum t/c
"""

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from studies.vsp_planform import config, param

param.REGION_A_RULE["const_chord"] = "preserved"

from studies.vsp_planform.run_opt import POINT, build_problem, load_baseline, trim_alpha  # noqa: E402

WEIGHT = 382547.0
SPAR_X_C = 0.75
SPAR_MIN_IN = 7.0
IN = 0.0254
ROOT_CHORD_IN = 105.0
N_TOC_CP = 35

# t(0.75c)/t_max for each family, measured from the real coordinates.
T75_RATIO = {"as-built": 0.527, "naca664221": 0.595, "fx2": 0.621}

CASES = [
    # label, c_max_t, junction chord (in), t/c mode, t75 ratio key
    ("as-built ConstChord", None, None, "as-built", "as-built"),
    ("66(4)-221 c=56, t/c >= as-built", 0.45, 56.0, "spar_floor", "naca664221"),
    ("66(4)-221 c=56, min t/c", 0.45, 56.0, "spar", "naca664221"),
    ("66(4)-221 c=64, min t/c", 0.45, 64.0, "spar", "naca664221"),
    ("fx2 c=57.2, min t/c", 0.50, 57.2, "spar", "fx2"),
]


def spline_map_from_model(prob, n_cp):
    """Measure the control-point -> panel map from the live model.

    Rebuilding the spline standalone got this wrong: OAS interpolates at
    ``get_normalized_span_coords(surface, mid_panel=True)``, not at evenly spaced
    points, so a hand-built SplineComp targets the wrong stations. Perturbing the
    real model with basis vectors cannot disagree with itself.
    """
    cols = []
    saved = np.asarray(prob.get_val("wing.t_over_c_cp")).copy()
    for i in range(n_cp):
        e = np.zeros(n_cp)
        e[i] = 1.0
        prob.set_val("wing.t_over_c_cp", e)
        prob.run_model()
        cols.append(np.asarray(prob.get_val("wing.t_over_c")).ravel())
    prob.set_val("wing.t_over_c_cp", saved)
    prob.run_model()
    return np.column_stack(cols)


def cp_for(M, target_panel, lo=0.08, hi=0.30):
    """Bounded solve for control points matching a target t/c distribution.

    Plain least squares is unusable: inverting a smoothing B-spline against a
    sharp corner is ill-conditioned and returned oscillating control points with
    *negative* thickness (measured: -0.45). Bounds keep it physical.
    """
    from scipy.optimize import lsq_linear

    return lsq_linear(M, target_panel, bounds=(lo, hi)).x


def spar_depth_in(toc, chord_m, ratio):
    """Thickness at the 0.75c spar, in inches."""
    return ratio * toc * chord_m / IN


def main():
    mesh, stick, regions, planform0, _, _ = load_baseline("const_chord", config.N_SPANWISE_HALF, 9)
    q = 0.5 * config.RHO * config.V_MS**2
    y_stick = np.abs(stick.le[:, 1]) * config.SCALE

    print("=" * 108)
    print(f"ConstChord at MTOW {WEIGHT:,.0f} N, span 118 ft pinned, full OAS VLM + viscous + wave")
    print("=" * 108)
    print(
        f"{'case':>30} {'c_junc':>7} {'S_ref':>7} {'AR':>6} {'CL':>6} "
        f"{'D_ind N':>9} {'D_visc N':>9} {'D_wave N':>9} {'D_tot N':>9} {'minB spar':>10}"
    )
    print("-" * 108)

    results = {}
    import studies.vsp_planform.run_opt as run_opt

    original_build_surface = run_opt.build_surface

    for label, c_max_t, c_junc, toc_mode, ratio_key in CASES:
        # c_max_t is read once when the viscous component is set up, so it has to
        # be injected into the surface dict before the problem is built.
        def surface_builder(mesh_, stick_, regions_, _c=c_max_t):
            s = original_build_surface(mesh_, stick_, regions_)
            if _c is not None:
                s["c_max_t"] = _c
            # The default 5 control points cannot represent the steep t/c rise the
            # spar rule needs near the winglet; give the spline enough freedom.
            s["t_over_c_cp"] = np.full(N_TOC_CP, float(np.mean(s["t_over_c_cp"])))
            return s

        run_opt.build_surface = surface_builder
        prob, surface = run_opt.build_problem("const_chord", mesh, stick, regions, planform0)
        run_opt.build_surface = original_build_surface

        # taper_B sets the junction chord as a fraction of region A's chord.
        if c_junc is not None:
            prob.set_val("wing.taper_B", c_junc / ROOT_CHORD_IN)
        prob.run_model()

        geom = prob.get_val("wing.mesh", units="m")
        chord_local = geom[-1, :, 0] - geom[0, :, 0]
        y_mesh = np.abs(geom[0, :, 1])
        y_junc = regions.y_c_start * config.SCALE
        j_junc = int(np.argmin(np.abs(y_mesh - y_junc)))

        ratio = T75_RATIO[ratio_key]
        if toc_mode == "as-built":
            toc = np.interp(y_mesh, y_stick, stick.toc)
        elif toc_mode == "uniform":
            toc = np.full(y_mesh.size, 0.21)
        else:
            toc = np.interp(y_mesh, y_stick, stick.toc)
            need = SPAR_MIN_IN * IN / (ratio * np.maximum(chord_local, 1e-6))
            if toc_mode == "spar_floor":
                need = np.maximum(need, toc)  # never thinner than as built
            # By index, not by comparing floats: the junction node and y_c_start
            # differ by ~0.2 mm, which a tolerance-free mask silently drops.
            inboard = np.arange(y_mesh.size) <= j_junc
            toc = np.where(inboard, np.clip(need, 0.08, 0.30), toc)

        # Drive t/c per station: invert the spline instead of sampling it.
        n_cp = prob.get_val("wing.t_over_c_cp").size
        M = spline_map_from_model(prob, n_cp)
        # Sample the target at the true mid-panel stations, which is where OAS
        # interpolates -- not at evenly spaced eta.
        y_panel_t = 0.5 * (y_mesh[:-1] + y_mesh[1:])
        target_panel = np.interp(y_panel_t, y_mesh, toc)
        prob.set_val("wing.t_over_c_cp", cp_for(M, target_panel))
        prob.run_model()

        trim_alpha(prob, WEIGHT / (q * float(prob.get_val(f"{POINT}.wing.S_ref")[0])))

        s_ref = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
        span = 2.0 * float(y_mesh.max())
        cdi = float(prob.get_val(f"{POINT}.wing_perf.CDi")[0])
        cdv = float(prob.get_val(f"{POINT}.wing_perf.CDv")[0])
        cdw = float(prob.get_val(f"{POINT}.wing_perf.CDw")[0])
        cl = float(prob.get_val(f"{POINT}.wing_perf.CL")[0])
        qs = q * s_ref
        # Spar depth from the t/c the model actually used, not the one requested.
        toc_actual = np.asarray(prob.get_val("wing.t_over_c")).ravel()
        y_panel = 0.5 * (y_mesh[:-1] + y_mesh[1:])
        c_panel = 0.5 * (chord_local[:-1] + chord_local[1:])
        depth_panel = ratio * toc_actual * c_panel / IN
        y_a = regions.y_a_end * config.SCALE
        inB = (y_panel >= y_a) & (y_panel <= y_junc)
        depth = float(depth_panel[inB].min())  # worst point in region B
        toc_junc = float(np.interp(y_junc, y_panel, toc_actual))
        toc_req = float(np.interp(y_junc, y_mesh, toc))

        results[label] = (qs * (cdi + cdv + cdw), qs * cdi, toc_req, toc_junc)
        print(
            f"{label:>30} {float(chord_local[j_junc]) / IN:7.1f} {s_ref:7.2f} {span**2 / s_ref:6.2f} "
            f"{cl:6.3f} {qs * cdi:9.1f} {qs * cdv:9.1f} {qs * cdw:9.1f} "
            f"{qs * (cdi + cdv + cdw):9.1f} {depth:10.2f}"
        )

    base, base_i = results["as-built ConstChord"][:2]
    print("\nt/c at the junction, requested vs delivered by the spline:")
    for label, (_, _, req, act) in results.items():
        print(f"  {label:>30}  requested {req:.4f}   delivered {act:.4f}")
    print("\nchange vs as-built:")
    for label, (total, di, _, _) in results.items():
        if label != "as-built ConstChord":
            print(f"  {label:>30}  {total - base:+9.1f} N  ({total / base - 1:+7.2%})"
                  f"   induced {di - base_i:+7.1f} N")
    print("\nclosed form predicted induced drag invariant with area; "
          "the VLM computes it from the vortex system, so any drift above is a real check.")


if __name__ == "__main__":
    main()
