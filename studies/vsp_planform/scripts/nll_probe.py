"""Nonlinear lifting-line stall probe for ConstChord, baseline vs optimum.

The VLM cannot see stall: it is linear and inviscid, and will report a local Cl
of 1.5 without complaint. ``asb.NonlinearLiftingLine`` couples 2D viscous
section data (NeuralFoil) to a 3D lifting-line solution, so it can say where and
when the wing actually stalls.

Question being settled: the VLM says the optimum needs local Cl = 1.491 at
eta = 0.96. Is the tip at or past stall?

Section airfoils are chosen from the *measured* geometry, not assumed: camber,
camber position and thickness are read off each mesh section (the mesh IS the
camber surface) and mapped to the nearest NACA 4-digit. That choice drives the
answer, so it is printed per station.

ConstChord's region-A rule is forced back to "preserved" here: the optimum being
probed was computed under that rule, before the switch to "root_le_fixed".
"""

import sys

import numpy as np

sys.path.insert(0, "/home/alex/repos/OpenAeroStruct")

import aerosandbox as asb

from studies.vsp_planform import config, param

param.REGION_A_RULE["const_chord"] = "preserved"

from studies.vsp_planform.run_opt import POINT, build_problem, load_baseline, trim_alpha  # noqa: E402

NAME = "const_chord"
WEIGHT = 382547.0
N_SPANWISE = 21  # coarse: NLL is an implicit solve, far slower than the VLM
NX = 9
SPANWISE_RESOLUTION = 3

OPT = {
    "alpha": 4.15828236,
    "taper_B": 0.16313306,
    "twist_cp": np.array([-2.65077686, -2.5275155, -1.32748637, 3.89971517, 3.6938989]),
    "wingbox_pct": 0.75,
}

DOE_RE_FLOOR = 1.7e6


def section_shape(mesh, j):
    """Chord, twist, max camber and its position for one mesh section.

    The mesh nodes are the mean camber line, so camber is measured directly as
    the offset from the leading-edge/trailing-edge chord line.
    """
    pts = mesh[:, j, :]
    le, te = pts[0], pts[-1]
    chord_vec = te - le
    chord = float(np.linalg.norm(chord_vec[[0, 2]]))
    twist = float(np.degrees(np.arctan2(-(te[2] - le[2]), te[0] - le[0])))

    # Rotate into the local chord frame: xi along the chord, zeta normal to it.
    ct, st = chord_vec[0] / chord, -chord_vec[2] / chord
    d = pts - le
    xi = (d[:, 0] * ct - d[:, 2] * st) / chord
    zeta = (d[:, 0] * st + d[:, 2] * ct) / chord

    k = int(np.argmax(np.abs(zeta)))
    return chord, twist, float(zeta[k]), float(xi[k])


def naca4_for(camber, camber_pos, toc):
    """Nearest NACA 4-digit to a measured section."""
    m = int(np.clip(round(camber * 100), 0, 9))
    p = 0 if m == 0 else int(np.clip(round(camber_pos * 10), 1, 9))
    t = int(np.clip(round(toc * 100), 6, 30))
    return f"naca{m}{p}{t:02d}"


def build_airplane(mesh, toc, label):
    """An AeroSandbox half-wing (symmetric) from an OAS half mesh."""
    ny = mesh.shape[1]
    # Thin the sections: NLL subdivides each panel anyway.
    idx = np.unique(np.linspace(0, ny - 1, 13).astype(int))

    print(f"\n  {label}: section table")
    print(f"    {'eta':>6} {'chord':>7} {'twist':>7} {'camber':>7} {'pos':>5} {'t/c':>6} {'Re':>10}  airfoil")
    xsecs, root_le = [], mesh[0, 0, :].copy()
    for j in idx:
        chord, twist, camber, camber_pos = section_shape(mesh, j)
        t = float(toc[j])
        name = naca4_for(camber, camber_pos, t)
        re = config.RE_PER_M * chord
        flag = "  <-- below DOE floor" if re < DOE_RE_FLOOR else ""
        eta = mesh[0, j, 1] / mesh[0, -1, 1]
        print(
            f"    {eta:6.3f} {chord:7.3f} {twist:+7.2f} {camber:7.4f} {camber_pos:5.2f} "
            f"{t:6.4f} {re:10.3e}  {name}{flag}"
        )
        le = mesh[0, j, :] - root_le
        xsecs.append(
            asb.WingXSec(
                xyz_le=[float(le[0]), float(le[1]), float(le[2])],
                chord=float(chord),
                twist=float(twist),
                airfoil=asb.Airfoil(name),
            )
        )

    wing = asb.Wing(name="wing", symmetric=True, xsecs=xsecs)
    plane = asb.Airplane(name=label, wings=[wing], s_ref=wing.area(), b_ref=wing.span(), c_ref=wing.mean_geometric_chord())
    print(f"    -> asb area {wing.area():.3f} m2, span {wing.span():.3f} m, AR {wing.aspect_ratio():.2f}")
    return plane


def sweep(plane, label, alphas):
    """Run NLL up through stall. Returns the alpha/CL curve actually achieved."""
    print(f"\n  {label}: NLL sweep (spanwise_resolution={SPANWISE_RESOLUTION})")
    rows = []
    for a in alphas:
        op = asb.OperatingPoint(velocity=config.V_MS, alpha=float(a))
        try:
            r = asb.NonlinearLiftingLine(
                airplane=plane, op_point=op, spanwise_resolution=SPANWISE_RESOLUTION, verbose=False
            ).run()
            cl = float(np.asarray(r["CL"]).ravel()[0])
            cd = float(np.asarray(r["CD"]).ravel()[0])
            rows.append((float(a), cl, cd))
            print(f"    alpha {a:6.2f}   CL {cl:7.4f}   CD {cd:8.5f}   L/D {cl / cd:7.2f}", flush=True)
        except Exception as exc:  # noqa: BLE001 - the failure itself is the datum
            print(f"    alpha {a:6.2f}   NOT CONVERGED: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            rows.append((float(a), np.nan, np.nan))
    return np.array(rows)


def main():
    mesh0, stick, regions, planform0, _, _ = load_baseline(NAME, N_SPANWISE, NX)
    prob, _ = build_problem(NAME, mesh0, stick, regions, planform0)
    q = 0.5 * config.RHO * config.V_MS**2

    # t/c per mesh station, from the stick.
    y_stick = np.abs(stick.le[:, 1]) * config.SCALE

    cases = {}
    prob.run_model()
    trim_alpha(prob, WEIGHT / (q * float(prob.get_val(f"{POINT}.wing.S_ref")[0])))
    cases["baseline"] = prob.get_val("wing.mesh", units="m").copy()

    prob.set_val("alpha", OPT["alpha"], units="deg")
    prob.set_val("wing.taper_B", OPT["taper_B"])
    prob.set_val("wing.twist_cp", OPT["twist_cp"], units="deg")
    prob.set_val("wing.wingbox_pct", OPT["wingbox_pct"])
    prob.run_model()
    cases["optimized"] = prob.get_val("wing.mesh", units="m").copy()

    alphas = np.arange(0.0, 22.1, 2.0)
    results = {}
    for label, mesh in cases.items():
        toc = np.interp(np.abs(mesh[0, :, 1]), y_stick, stick.toc)
        plane = build_airplane(mesh, toc, label)
        results[label] = sweep(plane, label, alphas)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for label, rows in results.items():
        good = rows[~np.isnan(rows[:, 1])]
        if good.size == 0:
            print(f"  {label}: no converged points")
            continue
        k = int(np.argmax(good[:, 1]))
        peaked = k < len(good) - 1
        print(
            f"  {label}: max converged CL {good[k, 1]:.4f} at alpha {good[k, 0]:.1f} deg"
            f"   ({'CL peaked and fell' if peaked else 'still rising at last converged alpha'})"
        )
        n_fail = int(np.isnan(rows[:, 1]).sum())
        if n_fail:
            print(f"           {n_fail} of {len(rows)} alphas did not converge")


if __name__ == "__main__":
    main()
