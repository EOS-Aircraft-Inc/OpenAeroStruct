"""Is our CSV-built mesh a legitimate OpenAeroStruct mesh?

Three checks, cheapest first:

1. Convention. Compare our mesh against what ``generate_mesh`` produces, on the
   invariants OAS relies on (chordwise LE-first, spanwise increasing y).
2. Handedness. OAS's two mesh sources disagree: ``generate_mesh(symmetry=True)``
   returns the LEFT half (y from -b/2 to 0) while ``generate_vsp_surfaces``
   returns the RIGHT half. Run the VLM on both handednesses of the same wing and
   see whether it cares.
3. Consistency. Run the VLM on our full mesh (symmetry=False) and on our half
   mesh (symmetry=True). A legitimate mesh gives the same CL and CD both ways.
"""

import sys

import numpy as np
import openmdao.api as om

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from openaerostruct.aerodynamics.aero_groups import AeroPoint
from openaerostruct.meshing.mesh_generator import generate_mesh

from studies.vsp_planform import config
from studies.vsp_planform.mesh import full_mesh, half_mesh, resample, spanwise_stations
from studies.vsp_planform.regions import detect_regions


def run_vlm(mesh, symmetry, alpha=3.0):
    """CL and CD from a bare VLM run on one mesh."""
    surface = {
        "name": "wing",
        "symmetry": symmetry,
        "S_ref_type": "wetted",
        "mesh": mesh,
        "twist_cp": np.zeros(2),
        "CL0": 0.0,
        "CD0": 0.0,
        "k_lam": 0.05,
        "t_over_c_cp": np.array([0.15]),
        "c_max_t": 0.303,
        "with_viscous": False,
        "with_wave": False,
        "fem_origin": 0.35,
    }

    ivc = om.IndepVarComp()
    ivc.add_output("v", val=config.V_MS, units="m/s")
    ivc.add_output("alpha", val=alpha, units="deg")
    ivc.add_output("Mach_number", val=config.MACH)
    ivc.add_output("re", val=config.RE_PER_M, units="1/m")
    ivc.add_output("rho", val=config.RHO, units="kg/m**3")
    ivc.add_output("cg", val=np.zeros(3), units="m")

    prob = om.Problem(reports=False)
    prob.model.add_subsystem("prob_vars", ivc, promotes=["*"])

    geom = om.Group()
    geom.add_subsystem("mesh_ivc", om.IndepVarComp("mesh", val=mesh), promotes=["*"])
    prob.model.add_subsystem(surface["name"], geom)

    point = AeroPoint(surfaces=[surface])
    prob.model.add_subsystem("aero", point, promotes=["v", "alpha", "Mach_number", "re", "rho", "cg"])
    prob.model.connect("wing.mesh", "aero.wing.def_mesh")
    prob.model.connect("wing.mesh", "aero.aero_states.wing_def_mesh")

    prob.setup()
    prob.run_model()
    return {
        "CL": float(prob.get_val("aero.wing_perf.CL")[0]),
        "CD": float(prob.get_val("aero.wing_perf.CD")[0]),
        "S_ref": float(prob.get_val("aero.wing.S_ref")[0]),
    }


def mirror(mesh):
    """Turn a right-half mesh into a left-half one, or the reverse."""
    out = mesh[:, ::-1, :].copy()
    out[:, :, 1] *= -1.0
    return out


print("=" * 78)
print("1. Convention check against generate_mesh")
print("=" * 78)
rect = generate_mesh(
    {
        "num_y": 15,
        "num_x": 5,
        "wing_type": "rect",
        "symmetry": False,
        "span": 10.0,
        "root_chord": 1.0,
        "span_cos_spacing": 0.0,
        "offset": [-0.5, 0, 0],
    }
)
ours = full_mesh(config.BASELINES["plan_l"])
for label, m in (("generate_mesh (full)", rect), ("ours (full)", ours)):
    le_first = bool(np.all(m[0, :, 0] < m[-1, :, 0]))
    y_increasing = bool(np.all(np.diff(m[0, :, 1]) > 0))
    print(f"  {label:24s} shape {str(m.shape):14s} LE at index 0: {le_first}   y increasing: {y_increasing}")

print()
print("=" * 78)
print("2. Does the VLM care which half it is given?")
print("=" * 78)
left = generate_mesh(
    {
        "num_y": 15,
        "num_x": 5,
        "wing_type": "rect",
        "symmetry": True,
        "span": 10.0,
        "root_chord": 1.0,
        "span_cos_spacing": 0.0,
        "offset": [-0.5, 0, 0],
    }
)
right = mirror(left)
print(f"  left half  y: {left[0, 0, 1]:+.2f} -> {left[0, -1, 1]:+.2f}")
print(f"  right half y: {right[0, 0, 1]:+.2f} -> {right[0, -1, 1]:+.2f}")
res_left = run_vlm(left, symmetry=True)
res_right = run_vlm(right, symmetry=True)
print(f"  left  half, symmetry=True : CL {res_left['CL']:.6f}  CD {res_left['CD']:.6f}  S_ref {res_left['S_ref']:.4f}")
print(f"  right half, symmetry=True : CL {res_right['CL']:.6f}  CD {res_right['CD']:.6f}  S_ref {res_right['S_ref']:.4f}")
print(f"  -> CL difference: {abs(res_left['CL'] - res_right['CL']):.3e}")

print()
print("=" * 78)
print("3. Our full mesh vs our half mesh, same wing")
print("=" * 78)
for name in config.BASELINES:
    # Resample first: the native const_chord full mesh is 5264 panels, whose AIC
    # matrix alone is 222 MB. Build the full mesh by mirroring the resampled
    # half, so both runs see exactly the same geometry.
    half_native, stick = half_mesh(config.BASELINES[name])
    regions = detect_regions(stick)
    y_new = spanwise_stations(half_native[0, :, 1], 21, regions.y_c_start * config.SCALE)
    half, _ = resample(half_native, y_new, 5)
    full = np.hstack((mirror(half)[:, :-1, :], half))

    res_full = run_vlm(full, symmetry=False)
    res_half = run_vlm(half, symmetry=True)
    dcl = abs(res_full["CL"] - res_half["CL"]) / abs(res_full["CL"])
    dcd = abs(res_full["CD"] - res_half["CD"]) / abs(res_full["CD"])
    print(f"  {name}")
    print(f"    full, symmetry=False : CL {res_full['CL']:.6f}  CD {res_full['CD']:.6f}  S_ref {res_full['S_ref']:.4f}")
    print(f"    half, symmetry=True  : CL {res_half['CL']:.6f}  CD {res_half['CD']:.6f}  S_ref {res_half['S_ref']:.4f}")
    print(f"    -> relative CL diff {dcl:.3e}   CD diff {dcd:.3e}")
