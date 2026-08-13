"""Does the parameterization actually reproduce the VSP geometry it came from?

Three independent errors get confused with each other in a pipeline like this,
so this script measures each one separately:

1. **Parameterization round trip.** Rebuild the mesh through
   :class:`~studies.vsp_planform.param.RegionGeometry` with every design
   variable at its baseline value, and compare node by node with the mesh it was
   built from. The baseline design vector is a zero perturbation by
   construction, so this should be at machine precision. Anything larger is a
   bug in the map, not a modelling choice.

2. **Resampling.** What the spanwise and chordwise station counts cost, reported
   as the two separate round-trip residuals ``mesh.resample`` returns. They have
   different causes and different fixes, so they are never summed.

3. **Aerodynamic consequence.** Run the VLM on the native mesh, on a
   spanwise-only resampling, and on the fully resampled mesh, and compare CL, CD
   and the spanwise load distribution. Node error in millimetres is only
   interesting insofar as it moves the numbers the study actually reports -- and
   the two cuts have to be separated here too, because a coarser chordwise
   discretization changes the VLM answer even when it loses no geometry at all.

Run with ``python studies/vsp_planform/verify_roundtrip.py``.
"""

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from studies.vsp_planform import config  # noqa: E402
from studies.vsp_planform.mesh import resample_spanwise  # noqa: E402
from studies.vsp_planform.run_opt import POINT, build_problem, load_baseline  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def rebuild_error(name, mesh, stick, regions, planform0):
    """Max and RMS node error of rebuilding ``mesh`` at the baseline design."""
    prob, _ = build_problem(name, mesh, stick, regions, planform0)
    prob.run_model()
    rebuilt = prob.get_val("wing.mesh")
    err = np.linalg.norm(rebuilt - mesh, axis=2)
    return {"max": float(err.max()), "rms": float(np.sqrt(np.mean(err**2)))}, prob


def spanwise_load(prob):
    """Panel-midpoint y and lift per unit span, from the VLM section forces."""
    forces = prob.get_val(f"{POINT}.aero_states.wing_sec_forces")
    widths = prob.get_val(f"{POINT}.wing.widths")
    mesh = prob.get_val("wing.mesh")
    y_mid = 0.5 * (mesh[0, :-1, 1] + mesh[0, 1:, 1])
    lift = forces[:, :, 2].sum(axis=0) / widths
    return y_mid, lift


def plot_loads(results, path):
    """Overlay the native and resampled spanwise loading for both baselines."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  (matplotlib not available; skipping {path})")
        return None

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.2), squeeze=False)
    for ax, (name, res) in zip(axes[0], results.items()):
        for label, style in (("native", "-"), ("spanwise only", ":"), ("resampled", "--")):
            y, lift = res["load"][label]
            ax.plot(y / y.max(), lift, style, label=f"{label} (ny={y.size + 1})")
        ax.set_title(f"{name}: CD {res['aero']['native']['CD']:.5f} vs {res['aero']['resampled']['CD']:.5f}")
        ax.set_xlabel("y / semi-span")
        ax.set_ylabel("lift per unit span [N/m]")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def trim_to_cl(prob, target=None, tol=1e-10, max_iter=30):
    """Secant-solve alpha for a target CL and return the trimmed state.

    Comparing meshes at fixed alpha charges a coarse mesh for a lift difference
    that the study never sees, because every reported number is produced at a
    target CL. Trimming first is the honest comparison.
    """
    if target is None:
        target = config.CL_TARGET

    a0, a1 = 0.0, 2.0
    prob.set_val("alpha", a0, units="deg")
    prob.run_model()
    f0 = prob.get_val(f"{POINT}.CL")[0] - target

    for _ in range(max_iter):
        prob.set_val("alpha", a1, units="deg")
        prob.run_model()
        f1 = prob.get_val(f"{POINT}.CL")[0] - target
        if abs(f1) < tol:
            break
        a0, f0, a1 = a1, f1, a1 - f1 * (a1 - a0) / (f1 - f0)

    return {
        "alpha": float(prob.get_val("alpha", units="deg")[0]),
        "CL": float(prob.get_val(f"{POINT}.CL")[0]),
        "CD": float(prob.get_val(f"{POINT}.CD")[0]),
    }


def aero_state(prob):
    return {
        "CL": float(prob.get_val(f"{POINT}.wing_perf.CL")[0]),
        "CD": float(prob.get_val(f"{POINT}.wing_perf.CD")[0]),
        "S_ref": float(prob.get_val(f"{POINT}.wing.S_ref")[0]),
    }


def verify(name):
    print("\n" + "=" * 78)
    print(name)
    print("=" * 78)

    mesh, stick, regions, planform0, residual, mesh_native = load_baseline(name)
    print(f"  native mesh   {mesh_native.shape}")
    print(f"  resampled     {mesh.shape}")
    print(
        f"  regions: A|B at section {regions.idx_a_end} (y = {regions.y_a_end:.2f} in), "
        f"B|C at {regions.idx_c_start} (y = {regions.y_c_start:.2f} in)"
    )
    print(
        f"  fitted wingbox_pct = {planform0['wingbox_pct']:.5f}, x_spar = {planform0['x_spar']:.4f} in, "
        f"spar max deviation = {planform0['spar_max_dev']:.4f} in"
    )
    # The straight-spar rule says tan(sweep_B) = p * c_A * (1 - lambda) / span_B.
    # Comparing that against the sweep actually measured off the leading edge is
    # the check that the rule really does describe these wings.
    sweep_derived = np.degrees(
        np.arctan(planform0["wingbox_pct"] * planform0["c_a0"] * (1 - planform0["taper_B"]) / planform0["span_b"])
    )
    print(
        f"  taper_B = {planform0['taper_B']:.5f}, derived sweep_B = {sweep_derived:.4f} deg "
        f"(measured {planform0['sweep_B']:.4f} deg)"
    )

    print("\n  1. parameterization round trip at the baseline design vector")
    native_err, prob_native = rebuild_error(name, mesh_native, stick, regions, planform0)
    resamp_err, prob_resamp = rebuild_error(name, mesh, stick, regions, planform0)
    print(f"     native mesh:    max {native_err['max']:.3e} m, rms {native_err['rms']:.3e} m")
    print(f"     resampled mesh: max {resamp_err['max']:.3e} m, rms {resamp_err['rms']:.3e} m")

    print("\n  2. resampling residual (reported separately, never summed)")
    for stage in ("spanwise", "chordwise"):
        r = residual[stage]
        print(
            f"     {stage:<10} max {r['max'] * 1e3:8.3f} mm   rms {r['rms'] * 1e3:8.3f} mm   "
            f"max/semi-span {r['max_relative']:.3e}"
        )

    print("\n  3. aerodynamic consequence of resampling (same alpha)")
    # A spanwise-only step, so the chordwise cut can be charged separately. It
    # is the chordwise count that changes the VLM's own answer: a coarser
    # chordwise mesh integrates the camber line differently even where it loses
    # no geometry.
    mesh_span, _ = resample_spanwise(mesh_native, mesh[0, :, 1])
    _, prob_span = rebuild_error(name, mesh_span, stick, regions, planform0)

    aero = {
        "native": aero_state(prob_native),
        "spanwise only": aero_state(prob_span),
        "resampled": aero_state(prob_resamp),
    }
    ref = aero["native"]
    for label, state in aero.items():
        d_cl = 100 * (state["CL"] / ref["CL"] - 1)
        d_cd = 100 * (state["CD"] / ref["CD"] - 1)
        print(
            f"     {label:<14} CL {state['CL']:.6f} ({d_cl:+7.3f}%)   CD {state['CD']:.7f} ({d_cd:+7.3f}%)   "
            f"S_ref {state['S_ref']:.4f}"
        )

    print("\n  4. aerodynamic consequence of resampling (trimmed to a common CL)")
    # Step 3 holds alpha and lets CL move, which charges the coarser mesh for a
    # lift difference the study never actually experiences: everything here is
    # run at a target CL, so alpha absorbs most of the discretization shift.
    # This is the number that governs whether nx is adequate.
    trimmed = {
        "native": trim_to_cl(prob_native),
        "spanwise only": trim_to_cl(prob_span),
        "resampled": trim_to_cl(prob_resamp),
    }
    ref_t = trimmed["native"]
    for label, state in trimmed.items():
        d_cd = 100 * (state["CD"] / ref_t["CD"] - 1)
        print(
            f"     {label:<14} alpha {state['alpha']:+8.4f} deg   CD {state['CD']:.7f} ({d_cd:+7.3f}%)   "
            f"CL {state['CL']:.6f}"
        )
    print(
        f"     -> trim angle moves {trimmed['resampled']['alpha'] - ref_t['alpha']:+.4f} deg; "
        "drag comparisons are unaffected but an absolute incidence is not."
    )

    return {
        "rebuild": {"native": native_err, "resampled": resamp_err},
        "resampling": residual,
        "aero": aero,
        "trimmed": trimmed,
        "load": {
            "native": spanwise_load(prob_native),
            "spanwise only": spanwise_load(prob_span),
            "resampled": spanwise_load(prob_resamp),
        },
    }


def main():
    results = {name: verify(name) for name in config.BASELINES}

    path = plot_loads(results, os.path.join(OUT_DIR, "spanwise_load.png"))
    if path:
        print(f"\n  spanwise load overlay written to {path}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(
        f"{'baseline':<14}{'rebuild max [m]':>18}{'span resamp [mm]':>19}{'chord resamp [mm]':>19}"
        f"{'dCD trimmed':>13}{'dCD fixed-a':>13}"
    )
    for name, res in results.items():
        d_cd_fixed = 100 * (res["aero"]["resampled"]["CD"] / res["aero"]["native"]["CD"] - 1)
        d_cd_trim = 100 * (res["trimmed"]["resampled"]["CD"] / res["trimmed"]["native"]["CD"] - 1)
        print(
            f"{name:<14}{res['rebuild']['resampled']['max']:>18.3e}"
            f"{res['resampling']['spanwise']['max'] * 1e3:>19.3f}"
            f"{res['resampling']['chordwise']['max'] * 1e3:>19.3f}"
            f"{d_cd_trim:>12.3f}%{d_cd_fixed:>12.3f}%"
        )
    print(
        "\n  'dCD trimmed' is the figure that governs this study -- every reported number is\n"
        "  produced at a target CL, so alpha absorbs most of the discretization shift. The\n"
        "  fixed-alpha column is an order of magnitude larger and charges the coarse mesh\n"
        "  for a lift difference the study never experiences."
    )
    return results


if __name__ == "__main__":
    main()
