"""The spanwise lift distribution of a design point, at a stated weight.

WHY THIS EXISTS. Every number the study reports so far is integrated -- CL, CD,
drag, S_ref. A structures reviewer cannot use an integral: a spar is sized by the
load ALONG the span, and where that load sits. This pulls the distribution out of
the same OpenAeroStruct model the drag came from, so the aero and the loads are
the same wing rather than two models that agree approximately.

WHICH WEIGHT. The study SIZES at MTOW and FLIES at mid-cruise
(mission.cruise_weight_lb, README: "The wing is SIZED at MTOW but FLOWN at
mid-cruise weight"). Those are different distributions, so both are written by
default and each file states its own weight. Neither is a limit load: this is 1 g,
with no gust and no manoeuvre factor. Do not size a spar on it without applying
the factors the structures method requires.

WHAT ``cl`` IS. It is the SECTIONAL lift coefficient, lower case, normalized on the
LOCAL chord -- not the wing CL. OAS computes it in lift_coeff_2D.py:90 as

    cl(y) = [ lift per unit span at y ] / (0.5 * rho * v^2 * c(y))

and that file's own docstring says "these are the sectional Cls". The variable is
spelled ``Cl`` in OAS; this script writes it as ``cl``, because upper case C_L is
the whole-wing coefficient and the two must not share a name. The chord column is
the model's own mid-panel chord, so a reader can divide the columns and get the
cl column back. The script checks that identity before it writes.

HOW THE LIFT IS TAKEN. OAS gives ``aero_states.wing_sec_forces`` as the force on
every panel, shape (nx-1, ny-1, 3) in newtons. Summing the chordwise axis gives the
force on each spanwise STRIP; the lift is that force resolved perpendicular to the
freestream,

    L_strip = Fz * cos(alpha) - Fx * sin(alpha)

which is the same rotation OAS applies for its own CL. The mesh is a half wing, so
the total is twice the sum. Both identities are CHECKED rather than assumed: the
script compares 2 * sum(L_strip) against CL * q * S_ref and against the target
weight, and refuses to write a file if either disagrees by more than 0.1 percent.

THE ELLIPTICAL REFERENCE is the distribution with the same total lift on the same
semi-span, L'(y) = L'_0 * sqrt(1 - (y/s)^2). It is a yardstick for induced drag,
not a target. Read it with care outboard of the winglet junction at y = 674.95 in:
the model carries the winglet IN THE PLANE OF THE WING, so the last few strips are
a flattened winglet and not wing. The junction column in the CSV says which is which.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

from studies.vsp_planform import config                              # noqa: E402
from studies.vsp_planform.run_opt import POINT, trim_alpha           # noqa: E402
from studies.vsp_planform.coupling import mission                    # noqa: E402
import compare_classes as CC                                         # noqa: E402
# export_dat imports THIS module to ship the load with the geometry bundle, so the
# import back the other way is deferred into __main__ rather than taken here.

LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")
LOADS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "loads")
q = 0.5 * config.RHO * config.V_MS**2
LB_TO_N = 4.4482216152605
N_PER_M_TO_LB_PER_IN = (1.0 / LB_TO_N) * config.SCALE
Y_JUNCTION_IN = 674.95
TOL_REL = 1e-3          # the identity checks; 0.1 percent


def weights_lb():
    """The two weights the study actually uses, named the way the study names them."""
    return {"mtow": mission.MTOW_LB, "cruise": mission.cruise_weight_lb()}


def distribution(prob, w_lb):
    """Trim to this weight, then take the spanwise load off the trimmed model."""
    s_ref = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    w_n = w_lb * LB_TO_N
    cl_target = w_n / (q * s_ref)
    trim_alpha(prob, cl_target)

    alpha = float(prob.get_val("alpha", units="deg")[0])
    ca, sa = np.cos(np.radians(alpha)), np.sin(np.radians(alpha))
    f = np.asarray(prob.get_val(f"{POINT}.aero_states.wing_sec_forces"))   # (nx-1, ny-1, 3), N
    strip_n = f.sum(axis=0)                                                # (ny-1, 3)
    lift_n = strip_n[:, 2] * ca - strip_n[:, 0] * sa
    drag_n = strip_n[:, 0] * ca + strip_n[:, 2] * sa

    m_in = np.asarray(prob.get_val("wing.mesh", units="m")) / config.SCALE
    y_node = np.abs(m_in[0, :, 1])
    y_mid = 0.5 * (y_node[:-1] + y_node[1:])
    # The model's OWN mid-panel chord, which is what normalizes cl (lift_coeff_2D).
    # It is the chord LENGTH, marginally longer than the x-extent te_x - le_x.
    chord = np.asarray(prob.get_val(f"{POINT}.wing.chords", units="m")) / config.SCALE
    chord = 0.5 * (chord[:-1] + chord[1:])
    width_in = np.asarray(prob.get_val(f"{POINT}.wing.widths", units="m")) / config.SCALE
    cl = np.asarray(prob.get_val(f"{POINT}.wing_perf.Cl")).ravel()   # SECTIONAL, on c(y)

    s_ref_in2 = s_ref / config.SCALE**2
    cl_total = float(prob.get_val(f"{POINT}.wing_perf.CL")[0])
    cd_total = float(prob.get_val(f"{POINT}.wing_perf.CD")[0])
    semi_in = float(y_node[-1])
    c_bar_in = s_ref_in2 / (2.0 * semi_in)          # mean geometric chord, both wings

    lift_per_in = lift_n / width_in                                  # N per inch of span
    # The elliptical distribution with the SAME total lift on the SAME semi-span.
    l0 = lift_n.sum() / (semi_in * np.pi / 4.0)
    ell_per_in = l0 * np.sqrt(np.clip(1.0 - (y_mid / semi_in) ** 2, 0.0, None))

    cum = np.cumsum(lift_n)
    y_cp = float((lift_n * y_mid).sum() / lift_n.sum())

    # ---- the identities, checked ----
    lift_total_n = 2.0 * float(lift_n.sum())
    from_cl = cl_total * q * s_ref
    # cl is claimed to be sectional and normalized on the local chord, so that claim
    # is tested too: dividing the written columns must give the written cl back.
    cl_from_cols = (lift_per_in / config.SCALE) / (q * chord * config.SCALE)
    checks = {
        "sum_strips_N": lift_total_n,
        "CL_q_Sref_N": from_cl,
        "target_weight_N": w_n,
        "rel_vs_CL": abs(lift_total_n / from_cl - 1.0),
        "rel_vs_weight": abs(lift_total_n / w_n - 1.0),
        "rel_cl_sectional": float(np.max(np.abs(cl_from_cols / cl - 1.0))),
    }
    return {
        "alpha_deg": alpha, "CL": cl_total, "CD": cd_total,
        "L_over_D": cl_total / cd_total,
        "S_ref_in2": s_ref_in2, "semi_span_in": semi_in, "c_bar_in": c_bar_in,
        "weight_lb": w_lb, "weight_N": w_n,
        "y_in": y_mid, "chord_in": chord, "width_in": width_in,
        "cl": cl, "lift_N": lift_n, "drag_N": drag_n,
        "lift_N_per_in": lift_per_in,
        "lift_lb_per_in": lift_per_in / LB_TO_N,
        "elliptical_N_per_in": ell_per_in,
        "cl_c_over_CL_cbar": cl * chord / (cl_total * c_bar_in),
        "cum_lift_frac": cum / cum[-1],
        "y_cp_in": y_cp, "y_cp_frac": y_cp / semi_in,
        "root_bending_N_in": float((lift_n * y_mid).sum()),
        "checks": checks,
    }


def write_csv(path, d, name, prov):
    """One row per spanwise strip, with the conditions in the header."""
    ck = d["checks"]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        w = lambda s: fh.write(s + "\n")
        w(f"# {name} -- spanwise lift distribution, 1 g, TRIMMED")
        w(f"# Source: {prov}")
        w("#")
        w(f"# WEIGHT      {d['weight_lb']:.1f} lb ({d['weight_N']:.0f} N). This is 1 g and")
        w("#             carries NO gust and NO manoeuvre factor. It is not a limit load.")
        w(f"# CONDITION   {config.KTAS:.0f} KTAS at {config.ALTITUDE_FT:.0f} ft, "
          f"Mach {config.MACH:.4f}, rho {config.RHO:.6f} kg/m^3,")
        w(f"#             q {q:.1f} Pa, alpha {d['alpha_deg']:.4f} deg (trimmed to the weight above).")
        w(f"# INTEGRATED  CL {d['CL']:.5f}   CD {d['CD']:.6f}   L/D {d['L_over_D']:.2f}"
          f"   S_ref {d['S_ref_in2']/144.0:.1f} ft2")
        w(f"# CENTRE OF LIFT  y {d['y_cp_in']:.2f} in = {d['y_cp_frac']:.4f} of the "
          f"{d['semi_span_in']:.1f} in semi-span")
        w(f"# ROOT BENDING    {d['root_bending_N_in']:.0f} N.in per side "
          f"({d['root_bending_N_in']/LB_TO_N:.0f} lb.in), from this 1 g load alone")
        w("#")
        w("# CHECKED, not assumed. The strip sum is compared with two independent values:")
        w(f"#   2 * sum(lift_N)  {ck['sum_strips_N']:12.1f} N")
        w(f"#   CL * q * S_ref   {ck['CL_q_Sref_N']:12.1f} N   "
          f"(differs by {ck['rel_vs_CL']*100:.4f} %)")
        w(f"#   target weight    {ck['target_weight_N']:12.1f} N   "
          f"(differs by {ck['rel_vs_weight']*100:.4f} %)")
        w(f"# And cl is checked to BE sectional: lift_N_per_in / (q * chord_in) "
          f"reproduces the")
        w(f"#   cl column to {ck['rel_cl_sectional']*100:.4f} %.")
        w("#")
        w("# COLUMNS, one row per spanwise panel of the half wing. INCHES and NEWTONS.")
        w("#   y_in            panel centre, outboard of the aircraft centre line")
        w("#   chord_in        local chord LENGTH at the panel centre. This is the chord")
        w("#                   that normalizes cl, so the columns divide back to it.")
        w("#   width_in        spanwise width of the panel")
        w("#   cl              SECTIONAL lift coefficient, lower case, normalized on")
        w("#                   the LOCAL chord: cl = (lift per unit span) / (q * chord_in).")
        w("#                   It is NOT the wing CL, which is given as CL in this header.")
        w("#   lift_N          lift on the strip, ONE WING")
        w("#   lift_N_per_in   lift_N / width_in -- the running load")
        w("#   lift_lb_per_in  the same running load in pounds per inch")
        w("#   elliptical_N_per_in   the elliptical load with the same total and semi-span")
        w("#   cl_c_over_CL_cbar     the normalized loading, Cl*c / (CL*c_bar)")
        w("#   cum_lift_frac   fraction of one wing's lift inboard of this panel")
        w("#   region          wing, or winglet outboard of y = 674.95 in")
        w("#")
        w("# The model carries the winglet IN THE PLANE OF THE WING, so the winglet rows")
        w("# are a flattened winglet, not wing. The elliptical reference spans the whole")
        w("# semi-span including them, so compare it over the wing rows.")
        cols = ["y_in", "chord_in", "width_in", "cl", "lift_N", "lift_N_per_in",
                "lift_lb_per_in", "elliptical_N_per_in", "cl_c_over_CL_cbar",
                "cum_lift_frac", "region"]
        wr = csv.writer(fh)
        wr.writerow(cols)
        for i in range(len(d["y_in"])):
            reg = "wing" if d["y_in"][i] <= Y_JUNCTION_IN else "winglet"
            wr.writerow([f"{d['y_in'][i]:.4f}", f"{d['chord_in'][i]:.4f}",
                         f"{d['width_in'][i]:.4f}", f"{d['cl'][i]:.6f}",
                         f"{d['lift_N'][i]:.4f}", f"{d['lift_N_per_in'][i]:.5f}",
                         f"{d['lift_lb_per_in'][i]:.6f}",
                         f"{d['elliptical_N_per_in'][i]:.5f}",
                         f"{d['cl_c_over_CL_cbar'][i]:.6f}",
                         f"{d['cum_lift_frac'][i]:.6f}", reg])
    return path


if __name__ == "__main__":
    import export_dat as ED          # deferred: ED imports this module

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arc", default="A", choices=sorted(ED.ARCS))
    ap.add_argument("--weight", action="append", choices=sorted(weights_lb()),
                    help="repeatable; default both")
    ap.add_argument("--out", default=LOADS)
    a = ap.parse_args()
    which = a.weight or ["mtow", "cruise"]

    # load_design replays the design point and cross-checks it against its logged
    # drag, and hands back the built problem, so there is one build rather than two.
    _, _, _, prov, case, _, _, prob = ED.load_design(a.arc)
    print(f"Arc {a.arc}: {prov}")
    print(f"  replayed and matched its logged drag: {case['drag_N']:.1f} N")

    os.makedirs(a.out, exist_ok=True)
    for tag in which:
        w_lb = weights_lb()[tag]
        print(f"\n{tag.upper()}  {w_lb:.0f} lb")
        d = distribution(prob, w_lb)
        ck = d["checks"]
        print(f"  alpha {d['alpha_deg']:.4f} deg   CL {d['CL']:.5f}   L/D {d['L_over_D']:.2f}")
        print(f"  strip sum {ck['sum_strips_N']:.1f} N   vs CL*q*S_ref "
              f"{ck['rel_vs_CL']*100:+.4f} %   vs weight {ck['rel_vs_weight']*100:+.4f} %"
              f"   cl sectional to {ck['rel_cl_sectional']*100:.4f} %")
        bad = [k for k in ("rel_vs_CL", "rel_vs_weight", "rel_cl_sectional")
               if ck[k] > TOL_REL]
        if bad:
            raise SystemExit(f"  REFUSING TO WRITE: {bad} exceed {TOL_REL*100:.1f} % -- "
                             f"the strip forces do not integrate to the trimmed lift.")
        print(f"  centre of lift y {d['y_cp_in']:.1f} in ({d['y_cp_frac']:.3f} semi-span)"
              f"   root bending {d['root_bending_N_in']/LB_TO_N:.0f} lb.in per side")
        wing = d["y_in"] <= Y_JUNCTION_IN
        pk = int(np.argmax(d["lift_N_per_in"][wing]))
        print(f"  peak running load {d['lift_lb_per_in'][wing][pk]:.2f} lb/in at y "
              f"{d['y_in'][wing][pk]:.1f} in   max cl {d['cl'][wing].max():.4f} at y "
              f"{d['y_in'][wing][int(np.argmax(d['cl'][wing]))]:.1f} in")
        path = os.path.join(a.out, f"Arc{a.arc}_lift_{tag}.csv")
        write_csv(path, d, f"Arc {a.arc}", prov)
        print(f"  wrote {path}")
