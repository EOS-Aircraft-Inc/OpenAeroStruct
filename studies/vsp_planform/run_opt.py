"""Aero-only VLM drag optimization of the two VSP planform baselines.

Structure follows ``openaerostruct/examples/rectangular_wing/run_rect_wing.py``
and ``tests/integration_tests/test_aero.py``: an IndepVarComp of flight
conditions, a geometry group, an ``AeroPoint``, and the two mesh connections
(``wing.mesh`` -> ``<point>.wing.def_mesh`` and ->
``<point>.aero_states.wing_def_mesh``). The geometry group is ours
(:class:`studies.vsp_planform.param.RegionGeometry`) rather than OAS's, for the
reasons in ``param.py``.

Design variables
----------------
``wingbox_pct``, ``taper_B``, ``twist_cp`` (5), ``alpha``. Leading-edge sweep is
*not* a design variable: the straight-spar rule makes it a function of the first
two (see ``param.py``), and it is reported as a derived output.

Objective and lift constraint
-----------------------------
Two formulations, switched with ``--mode``:

``fixed_cl`` (default)
    Minimize CD with CL pinned to ``config.CL_TARGET`` and ``S_ref`` pinned to
    the baseline. Pinning the area is necessary here because CD is
    non-dimensionalized by it, so without the pin the optimizer just shrinks the
    wing. It is also the formulation that runs today, with no extra data needed.

``fixed_lift``
    Minimize *drag force* with *lift force* pinned to the cruise weight, and the
    area free above a floor set by ``config.MAX_CRUISE_CL``. This is the
    physically correct way to let the area trade, and it is what should be used
    as soon as a cruise weight is available: pass ``--weight <newtons>``.

Read the S_ref/taper warning printed by ``report_area_coupling`` before
believing a ``fixed_cl`` result: for the ConstChord geometry, the area is a
function of ``taper_B`` alone, so pinning it removes ``taper_B``'s freedom
entirely.
"""

import argparse
import os
import sys

# Runnable both as `python studies/vsp_planform/run_opt.py` and as
# `python -m studies.vsp_planform.run_opt`; the former puts this directory on
# sys.path instead of the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import openmdao.api as om

from openaerostruct.aerodynamics.aero_groups import AeroPoint

from studies.vsp_planform import config
from studies.vsp_planform.mesh import N_CHORDWISE, half_mesh, resample, spanwise_stations
from studies.vsp_planform.param import (
    baseline_planform,
    build_geometry_group,
    build_surface,
    mac_quarter_chord,
    twist_cp_bounds,
)
from studies.vsp_planform.regions import detect_regions

POINT = "aero_point_0"


def load_baseline(name, n_spanwise=None, nx=None):
    """Parse, region-detect and resample one baseline.

    Returns ``(mesh, stick, regions, planform0, residual)`` with the mesh in
    metres and the stick in native inches -- ``half_mesh`` scales one and not
    the other, so every use of a stick quantity below is multiplied by
    ``config.SCALE`` explicitly.
    """
    if n_spanwise is None:
        n_spanwise = config.N_SPANWISE_HALF
    if nx is None:
        nx = N_CHORDWISE

    mesh_native, stick = half_mesh(config.BASELINES[name])
    regions = detect_regions(stick, config.REGION_OVERRIDES.get(name))
    planform0 = baseline_planform(stick, regions, name=name)

    y_new = spanwise_stations(mesh_native[0, :, 1], n_spanwise, regions.y_c_start * config.SCALE)
    mesh, residual = resample(mesh_native, y_new, nx)
    return mesh, stick, regions, planform0, residual, mesh_native


def build_problem(name, mesh, stick, regions, planform0, extra=None):
    """Flight conditions -> geometry -> AeroPoint, plus dimensional lift/drag.

    ``extra`` is an optional ``f(model, mesh, regions)`` called just before
    ``setup()``. OpenMDAO refuses ``add_subsystem`` on an already-set-up model,
    so a study that needs its own component -- the region-B twist monotonicity
    constraint, for one -- has no way in without this hook. It is deliberately
    the last thing before setup, so anything it adds sees the finished model.
    """
    surface = build_surface(mesh, stick, regions)

    prob = om.Problem(reports=False)

    ivc = om.IndepVarComp()
    ivc.add_output("v", val=config.V_MS, units="m/s")
    ivc.add_output("alpha", val=2.0, units="deg")
    ivc.add_output("Mach_number", val=config.MACH)
    ivc.add_output("re", val=config.RE_PER_M, units="1/m")
    ivc.add_output("rho", val=config.RHO, units="kg/m**3")
    # AeroPoint computes CM about `cg`. The VSP models are in global VSP
    # coordinates -- x is 24 to 27 m, nowhere near the wing -- so leaving this
    # at the origin would give every moment a 25 m arm. Fix it at the
    # quarter-chord of the mean aerodynamic chord of the *baseline*: it is a
    # reference point, so it deliberately does not move with the design.
    # Nothing in this study constrains or optimizes CM; it is reported only.
    cg, mac, y_mac = mac_quarter_chord(mesh)
    ivc.add_output("cg", val=cg, units="m")
    prob.model.add_subsystem("prob_vars", ivc, promotes=["*"])

    prob.model.add_subsystem(
        "wing",
        build_geometry_group(surface, regions, planform0),
        promotes_outputs=["sweep_B", "station_chord", "wingbox_width", "twist_abs"],
    )
    prob.model.add_subsystem(
        POINT,
        AeroPoint(surfaces=[surface]),
        promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "cg"],
    )

    prob.model.connect("wing.mesh", f"{POINT}.wing.def_mesh")
    prob.model.connect("wing.mesh", f"{POINT}.aero_states.wing_def_mesh")
    prob.model.connect("wing.t_over_c", f"{POINT}.wing_perf.t_over_c")

    # Dimensional forces, so the objective/constraint pair can be swapped from
    # "min CD at fixed CL" to "min D at fixed L" without touching the model.
    # S_ref is already the full-wing area: VLMGeometry doubles it under symmetry.
    prob.model.add_subsystem(
        "forces",
        om.ExecComp(
            ["lift = 0.5 * rho * v**2 * S_ref * CL", "drag = 0.5 * rho * v**2 * S_ref * CD"],
            lift={"units": "N"},
            drag={"units": "N"},
            rho={"units": "kg/m**3", "val": config.RHO},
            v={"units": "m/s", "val": config.V_MS},
            S_ref={"units": "m**2", "val": 1.0},
            CL={"val": 0.5},
            CD={"val": 0.02},
        ),
        promotes_inputs=["rho", "v"],
        promotes_outputs=["lift", "drag"],
    )
    prob.model.connect(f"{POINT}.wing.S_ref", "forces.S_ref")
    prob.model.connect(f"{POINT}.wing_perf.CL", "forces.CL")
    prob.model.connect(f"{POINT}.wing_perf.CD", "forces.CD")

    if extra is not None:
        extra(prob.model, mesh, regions)

    prob.setup()
    return prob, surface


def add_optimization(prob, name, mesh, planform0, s_ref0, mode="fixed_cl", weight=None,
                     pct_dv=True):
    """Attach the driver, design variables, constraints and objective.

    ``pct_dv=False`` leaves ``wing.wingbox_pct`` OUT of the design variables, for a
    design that fixes the spar fraction rather than optimizing it. Do not emulate
    this by collapsing its bounds to a single value: SLSQP then carries a zero-range
    variable sitting on its own bound and fails with "positive directional derivative
    for linesearch". Measured on arc A -- the control converges, bounds pinned at
    0.750 fails, and the same run with the variable simply absent converges. A fixed
    quantity should not be a design variable at all.
    """
    model = prob.model

    n_cp = config.N_TWIST_CP
    tw_lower, tw_upper = twist_cp_bounds(mesh, n_cp)

    if pct_dv:
        model.add_design_var(
            "wing.wingbox_pct", lower=config.WINGBOX_CHORD_PCT_BOUNDS[0], upper=config.WINGBOX_CHORD_PCT_BOUNDS[1]
        )
    model.add_design_var("wing.taper_B", lower=config.TAPER_B_BOUNDS[0], upper=config.TAPER_B_BOUNDS[1])
    model.add_design_var("wing.twist_cp", lower=tw_lower, upper=tw_upper, units="deg")
    model.add_design_var("alpha", lower=-5.0, upper=12.0, units="deg")

    # The twist DV is incremental (the parsed mesh already carries the baseline
    # twist), so the user's absolute limits are imposed on the sum. The bounds
    # above are the same limits pushed onto the control points, which is only
    # approximate because a B-spline does not interpolate its control points.
    model.add_constraint(
        "twist_abs", lower=config.TWIST_BOUNDS[0], upper=config.TWIST_BOUNDS[1], units="deg", ref=config.TWIST_BOUNDS[1]
    )

    if name == "plan_l":
        # The wingbox must stay at least as wide as required at every station in
        # config.WINGBOX_WIDTH_STATIONS -- 65 in out to y = 100 in on the stock
        # settings, more stations once the rear spar kinks. The width comes out
        # of the geometry in metres; the requirements are in inches, so they are
        # scaled here rather than anywhere near the mesh. One `ref` has to serve
        # the whole vector, so it is the mean requirement.
        width_min = np.array([w for _, w in config.WINGBOX_WIDTH_STATIONS], dtype=float) * config.SCALE
        model.add_constraint("wingbox_width", lower=width_min, units="m", ref=float(np.mean(width_min)))

    # Every constraint is normalized to O(1) with `ref`. SLSQP works on the
    # scaled problem and merges all constraints into one tolerance test, so an
    # S_ref residual measured in the 86 m^2 the wing happens to have would
    # otherwise swamp a CL residual measured against 0.5, and the line search
    # stalls long before either is actually converged.
    if mode == "fixed_cl":
        model.add_constraint(f"{POINT}.wing_perf.CL", equals=config.CL_TARGET, ref=config.CL_TARGET)
        model.add_constraint(f"{POINT}.wing.S_ref", equals=s_ref0, ref=s_ref0)
        model.add_objective(f"{POINT}.wing_perf.CD", ref=1e-2)
    elif mode == "fixed_lift":
        if weight is None:
            raise ValueError("fixed_lift mode needs a cruise weight in newtons (--weight)")
        model.add_constraint("lift", equals=weight, ref=weight)
        # Size the wing by the cruise-CL limit instead of leaving area free.
        # Span is not a design variable, so at fixed lift the induced drag does
        # not respond to area at all and only profile drag does -- so on its own,
        # area is a one-way trade that runs to whatever floor a constraint
        # provides. That was measured: with only a CL ceiling, Plan_L stopped at
        # 74.1947 m^2 against a floor of 74.1948.
        #
        # This is a FLOOR, not an equality. An earlier version pinned it, on the
        # grounds that the optimizer ran to the floor anyway so the two were
        # equivalent and the equality said so more honestly. That equivalence
        # held only while nothing in the problem pushed area *up*. The wingbox
        # width constraints do: meeting a 25 in box at the winglet junction takes
        # a 66 in junction chord, which costs area. Against an equality that is
        # simply infeasible; against a floor the optimizer buys the area it needs
        # and the cruise CL falls out as a result worth reporting.
        q = 0.5 * config.RHO * config.V_MS**2
        s_ref_sized = weight / (q * config.MAX_CRUISE_CL)
        model.add_constraint(f"{POINT}.wing.S_ref", lower=s_ref_sized, ref=s_ref_sized)
        model.add_objective("drag", ref=1.0e4)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # pyOptSparse/SNOPT is not installed here, so SLSQP it is.
    prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", tol=1e-7, maxiter=100, disp=True)
    prob.driver.options["debug_print"] = ["objs", "desvars"]
    prob.setup()


def trim_alpha(prob, cl_target, tol=1e-10, max_iter=20):
    """Solve for the angle of attack that puts the model at ``cl_target``.

    Without this the "baseline" would be whatever CL alpha = 2 deg happens to
    produce, and comparing its CD against an optimized point that sits exactly
    at CL = 0.5 would credit the optimizer with the drag saved by simply flying
    at less lift. CL is very nearly linear in alpha, so a secant iteration gets
    there in a handful of model evaluations.
    """

    def cl_at(alpha):
        prob.set_val("alpha", alpha, units="deg")
        prob.run_model()
        return float(prob.get_val(f"{POINT}.wing_perf.CL")[0]) - cl_target

    a0 = float(prob.get_val("alpha", units="deg")[0])
    a1 = a0 + 1.0
    f0, f1 = cl_at(a0), cl_at(a1)

    for _ in range(max_iter):
        if abs(f1) < tol or f1 == f0:
            break
        a0, a1, f0 = a1, a1 - f1 * (a1 - a0) / (f1 - f0), f1
        f1 = cl_at(a1)

    if abs(f1) > 1e-8:
        raise RuntimeError(f"baseline trim failed: CL is off target by {f1:.3e}")
    return a1


def report_area_coupling(name, prob, planform0):
    """Check whether pinning S_ref removes taper_B's freedom.

    Region A's chord is frozen for ConstChord and the semi-span is fixed, so the
    only thing left that can change the planform area is ``taper_B`` -- shearing
    region B in x (which is all that ``wingbox_pct`` does there) preserves area
    exactly. If that is what we measure, then an S_ref equality constraint is an
    implicit equation for ``taper_B`` alone and the "optimization" over taper is
    a fiction.
    """
    p0, lam0 = planform0["wingbox_pct"], planform0["taper_B"]

    def area(p, lam):
        prob.set_val("wing.wingbox_pct", p)
        prob.set_val("wing.taper_B", lam)
        prob.run_model()
        return float(prob.get_val(f"{POINT}.wing.S_ref")[0])

    s00 = area(p0, lam0)
    dp = area(p0 * 1.05, lam0) - s00
    dlam = area(p0, lam0 * 1.05) - s00
    area(p0, lam0)

    print(f"  S_ref sensitivity: d(S_ref) for +5% wingbox_pct = {dp:+.6e} m^2  ({dp / s00:+.2e} relative)")
    print(f"                     d(S_ref) for +5% taper_B     = {dlam:+.6e} m^2  ({dlam / s00:+.2e} relative)")

    # The comparison has to be relative: shearing panels that are not quite
    # planar (the winglet) leaves an area residual a few orders of magnitude
    # above round-off, so an absolute "is it zero" test never fires.
    determined = abs(dp) < 1e-2 * abs(dlam)
    if determined:
        print("  WARNING: S_ref depends on taper_B ALONE. Pinning S_ref fully determines")
        print("           taper_B, leaving the optimizer only (wingbox_pct, twist, alpha).")
        print("           Use --mode fixed_lift --weight <N> to let the area trade honestly.")
    return determined


def _state(prob):
    widths = prob.get_val("wingbox_width", units="m") / config.SCALE
    required = np.array([w for _, w in config.WINGBOX_WIDTH_STATIONS], dtype=float)
    return {
        "CL": float(prob.get_val(f"{POINT}.wing_perf.CL")[0]),
        "CD": float(prob.get_val(f"{POINT}.wing_perf.CD")[0]),
        "S_ref": float(prob.get_val(f"{POINT}.wing.S_ref")[0]),
        "L/D": float(prob.get_val(f"{POINT}.wing_perf.CL")[0] / prob.get_val(f"{POINT}.wing_perf.CD")[0]),
        "drag_N": float(prob.get_val("drag")[0]),
        "alpha": float(prob.get_val("alpha", units="deg")[0]),
        "wingbox_pct": float(prob.get_val("wing.wingbox_pct")[0]),
        "taper_B": float(prob.get_val("wing.taper_B")[0]),
        "sweep_B": float(prob.get_val("sweep_B", units="deg")[0]),
        "wingbox_width_in": widths,
        # One number for the table: how much slack the tightest station has left.
        "wingbox_margin_in": float(np.min(widths - required)),
        "twist_root": float(prob.get_val("twist_abs", units="deg")[0]),
        "twist_tip": float(prob.get_val("twist_abs", units="deg")[-1]),
        "twist_cp": prob.get_val("wing.twist_cp", units="deg").copy(),
    }


ROWS = [
    ("CD", "{:.7f}"),
    ("CL", "{:.6f}"),
    ("L/D", "{:.3f}"),
    ("drag_N", "{:.1f}"),
    ("S_ref", "{:.4f}"),
    ("alpha", "{:+.4f}"),
    ("wingbox_pct", "{:.5f}"),
    ("taper_B", "{:.5f}"),
    ("sweep_B", "{:.4f}"),
    ("wingbox_margin_in", "{:+.2f}"),
    ("twist_root", "{:+.4f}"),
    ("twist_tip", "{:+.4f}"),
]


def print_table(results):
    names = list(results)
    print("\n" + "=" * 78)
    print("BASELINE vs OPTIMIZED")
    print("=" * 78)
    header = f"{'quantity':<18}" + "".join(f"{n + ' base':>19}{n + ' opt':>19}" for n in names)
    print(header)
    print("-" * len(header))
    for key, fmt in ROWS:
        line = f"{key:<18}"
        for n in names:
            for which in ("baseline", "optimized"):
                line += f"{fmt.format(results[n][which][key]):>19}"
        print(line)
    print("-" * len(header))
    for n in names:
        base, opt = results[n]["baseline"], results[n]["optimized"]
        print(f"{n:>12}: CD {base['CD']:.7f} -> {opt['CD']:.7f}  ({100 * (opt['CD'] / base['CD'] - 1):+.2f}%)")
        print(f"{'':>12}  twist_cp (deg, incremental) = {np.array2string(opt['twist_cp'], precision=3)}")


def run_one(name, mode="fixed_cl", weight=None, n_spanwise=None, nx=None):
    print("\n" + "=" * 78)
    print(f"{name}")
    print("=" * 78)

    mesh, stick, regions, planform0, residual, _ = load_baseline(name, n_spanwise, nx)
    print(f"  mesh {mesh.shape}, regions A|B at section {regions.idx_a_end}, B|C at {regions.idx_c_start}")
    cg, mac, y_mac = mac_quarter_chord(mesh)
    print(f"  MAC = {mac:.4f} m at y = {y_mac:.4f} m; moment reference cg = {np.array2string(cg, precision=4)} m")
    print(f"  baseline wingbox_pct = {planform0['wingbox_pct']:.5f} at x = {planform0['x_spar']:.4f} in")
    print(f"           (spar straightness: max deviation {planform0['spar_max_dev']:.4f} in)")
    print(f"  baseline taper_B = {planform0['taper_B']:.5f}, sweep_B = {planform0['sweep_B']:.4f} deg")
    print(f"  region A rule = {planform0['rule']}")
    print(
        f"  resampling residual: spanwise max {residual['spanwise']['max'] * 1e3:.3f} mm, "
        f"chordwise max {residual['chordwise']['max'] * 1e3:.3f} mm"
    )

    prob, _ = build_problem(name, mesh, stick, regions, planform0)

    # Trim the baseline to the same lift the optimizer will be held to, so the
    # comparison is like for like. Under fixed_lift that means the CL which puts
    # `weight` on the *baseline* area -- trimming to CL_TARGET instead would
    # compare a 211 kN baseline against a 383 kN optimum.
    if mode == "fixed_lift":
        prob.run_model()
        q = 0.5 * config.RHO * config.V_MS**2
        cl_baseline = weight / (q * float(prob.get_val(f"{POINT}.wing.S_ref")[0]))
    else:
        cl_baseline = config.CL_TARGET
    alpha_trim = trim_alpha(prob, cl_baseline)
    baseline = _state(prob)
    s_ref0 = baseline["S_ref"]
    print(f"  baseline trimmed to CL = {cl_baseline:.4f} at alpha = {alpha_trim:+.4f} deg")
    print(
        f"  baseline twist {baseline['twist_root']:+.3f} -> {baseline['twist_tip']:+.3f} deg, S_ref = {s_ref0:.4f} m^2"
    )
    coupled = report_area_coupling(name, prob, planform0)

    add_optimization(prob, name, mesh, planform0, s_ref0, mode=mode, weight=weight)
    # setup() reset the model, so start the optimizer from the trimmed baseline.
    prob.set_val("alpha", alpha_trim, units="deg")
    prob.run_model()
    prob.run_driver()

    return {
        "baseline": baseline,
        "optimized": _state(prob),
        "area_coupled": coupled,
        "success": bool(prob.driver.result.success),
        "exit_status": str(prob.driver.result.exit_status),
        "iterations": int(prob.driver.result.iter_count),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixed_cl", "fixed_lift"), default="fixed_cl")
    parser.add_argument("--weight", type=float, default=None, help="cruise weight in newtons, for fixed_lift")
    parser.add_argument("--baselines", nargs="*", default=list(config.BASELINES))
    parser.add_argument("--n-spanwise", type=int, default=None)
    parser.add_argument("--nx", type=int, default=None)
    args = parser.parse_args(argv)

    results = {}
    for name in args.baselines:
        results[name] = run_one(name, args.mode, args.weight, args.n_spanwise, args.nx)

    print_table(results)
    for name, res in results.items():
        status = "converged" if res["success"] else "FAILED"
        status += f" ({res['exit_status']}, {res['iterations']} iterations)"
        note = " (taper_B pinned by the S_ref constraint)" if res["area_coupled"] else ""
        print(f"  {name}: {status}{note}")
    return results


if __name__ == "__main__":
    main()
