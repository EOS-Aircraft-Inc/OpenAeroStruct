"""Export a design as the OpenVSP .dat bundle a colleague's tool reads.

WingCalc is fed the same thing already (coupling/geometry.py writes it into the
deck), so this is that export packaged the way the airfoil zips are shipped:

    <name>/
        <name>_af.csv                  airfoil metadata, one block per station
        <name>_<GeomID>_<i>.dat        Selig contour, one per station

The metadata block matches the format of the reference bundles
(Inputs/V3.5.3/OpenVSP/S12_t0.zip and the S12_t2.zip the tool is driven from),
including the five keys WingCalc itself does not read -- XSec Flag, XSec Index,
XSecSurf ID, FoilSurf u Value, Global u Value -- so the file is interchangeable
with a genuine VSP export rather than merely sufficient.

WHERE THE SHAPE COMES FROM. Not from OAS: its mesh is a camber surface and it
knows a section only as t/c and c_max_t.

  a design on a DATABASE SECTION (--airfoil e694, the default) exports THAT
  section's own contour, thickness-scaled to the design's local t/c with the
  camber left alone. Scaling thickness only is what keeps the two things the
  study actually used -- c_max_t, and the thickness RETENTION at the spar -- and
  both are properties of where the thickness sits, so a bodily y-scale would
  preserve them while a reshaping would not.

  a design on the AS-BUILT loft exports the baseline DegenGeom's plate (zCamber
  and t, normalized by chord), blended between bracketing baseline sections and
  scaled bodily to the design's local t/c. So it exports the as-built section
  SCALED, not a section designed at that thickness. That is the study's own
  assumption; it is not extra licence taken here, but it must travel with the file.

Getting this wrong is not cosmetic. e694 was chosen for its retention at the
0.574c spar -- it keeps 0.935 of its thickness there where the as-built keeps
0.814 -- and retention is entirely a matter of where the thickness sits
(c_max_t 0.405 against 0.310). Exporting the as-built shape carrying e694's t/c
NUMBERS would hand a colleague a wing whose spar depth does not match the study
that justified it.

Unlike the WingCalc path, the winglet is kept by default: it is dropped there
only because the provider's spanwise interpolation needs a monotonic ws, which is
a WingCalc constraint, not a geometry one.
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

from studies.vsp_planform import config                          # noqa: E402
from studies.vsp_planform.degen_csv import read_degen_csv, lifting_surfaces  # noqa: E402
from studies.vsp_planform.coupling.geometry import normalized_sections, section_at  # noqa: E402
import wing2_oas as w2                                           # noqa: E402
from wing8_constchord_toc import REGION_A_AS_BUILT_IN            # noqa: E402
from compare_classes import replay, baseline_case                # noqa: E402

LOGS = Path(os.path.dirname(os.path.dirname(_HERE))) / "out" / "logs"
# One folder for wing geometry hand-offs, beside the logs and figures.
GEOMS = Path(os.path.dirname(os.path.dirname(_HERE))) / "out" / "geometries"
SCALE = config.SCALE

ARCS = {                       # region A end, region A rule, design-point source
    "A": (REGION_A_AS_BUILT_IN, "root_le_fixed", ("wing8_design_point.json", "constchord_asbuilt")),
    "B": (w2.REGION_A_END_IN, "preserved", ("wing7_design_point.json", "wing7_mtow")),
    "C": (w2.REGION_A_END_IN, "root_le_fixed", ("wing7_design_point.json", "wing3_mtow")),
}
GEOM_ID = {"A": "ARCAOASEXP", "B": "ARCBOASEXP", "C": "ARCCOASEXP"}


def write_dat(path, header, x, upper, lower):
    """Selig order: TE along the upper surface to the LE, then lower back to TE."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header + "\n")
        for xi, yi in zip(x[::-1], upper[::-1]):
            fh.write(f"{xi:.16f} {yi:.16f}\n")
        for xi, yi in zip(x[1:], lower[1:]):
            fh.write(f"{xi:.16f} {yi:.16f}\n")


def database_profile(name, x_grid):
    """Camber and thickness of a database section, normalized by chord, on x_grid.

    Split rather than returned as upper/lower because the two are scaled
    differently: the thickness carries the design's t/c, the camber does not.
    """
    import aerosandbox as asb
    af = asb.Airfoil(name)
    t = np.array([float(af.local_thickness(x_over_c=float(x))) for x in x_grid])
    cam = np.array([float(af.local_camber(x_over_c=float(x))) for x in x_grid])
    return cam, t


def blended_profile(blend, x_grid):
    """Camber and thickness of a SPANWISE section pair, as functions of y (inches).

    A real lofted wing interpolates between two defining sections, so both camber and
    thickness are blended; the thickness is then scaled to the station's t/c by the
    caller, which is what preserves c_max_t and the retention curve.
    """
    cam_i, t_i = database_profile(blend["inboard"], x_grid)
    cam_o, t_o = database_profile(blend["outboard"], x_grid)
    f0, f1 = float(blend["f_start"]), float(blend["f_end"])
    semi = 708.0

    def at(y_in):
        w = float(np.clip((abs(y_in) / semi - f0) / (f1 - f0), 0.0, 1.0))
        return (1.0 - w) * cam_i + w * cam_o, (1.0 - w) * t_i + w * t_o

    return at


def export(mesh_m, toc, name, geom_id, out_dir, n_x=201, dir_hint=None, airfoil=None,
           blend=None):
    """Write <name>_af.csv plus one .dat per spanwise station. Returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    comp = list(lifting_surfaces(read_degen_csv(config.BASELINES[w2.BASELINE])).values())[0][0]
    frac, x_n, up_n, lo_n = normalized_sections(comp.plate, comp.stick)

    le, te = mesh_m[0] / SCALE, mesh_m[-1] / SCALE          # inches
    if le[0, 1] > le[-1, 1]:                                # OAS runs tip -> root
        le, te = le[::-1], te[::-1]
        toc = np.asarray(toc)[::-1]
    ny = le.shape[0]
    toc = np.asarray(toc, dtype=float)
    toc_node = toc if len(toc) == ny else np.interp(np.arange(ny), np.arange(len(toc)) + 0.5, toc)

    chord = np.linalg.norm(te - le, axis=1)
    ws = le[:, 1]
    span0, span1 = ws[0], ws[-1]
    # cosine spacing, clustered at the leading edge where curvature is highest
    x_grid = 0.5 * (1 - np.cos(np.linspace(0.0, np.pi, n_x)))

    hint = dir_hint or f"./{name}/"
    # A named section is the same shape at every station, so its camber and
    # thickness are built once; only the t/c scale changes down the span.
    db = None if airfoil in (None, "", "as-built") else database_profile(airfoil, x_grid)
    # A spanwise pair overrides the single section: the contour changes down the span.
    db_at = blended_profile(blend, x_grid) if blend else None
    blocks, dats = [], []
    for i in range(ny):
        f = 0.0 if span1 == span0 else (ws[i] - span0) / (span1 - span0)
        if db_at is not None:
            # lofted between two sections, then thickness-scaled to this station's t/c
            cam, t_n = db_at(ws[i])
            k = float(toc_node[i]) / float(t_n.max())
            up, lo = cam + 0.5 * t_n * k, cam - 0.5 * t_n * k
        elif db is not None:
            # thickness scaled to this station's t/c, camber untouched -- keeps
            # c_max_t and the retention curve, which is why this section was picked
            cam, t_n = db
            k = float(toc_node[i]) / float(t_n.max())
            up, lo = cam + 0.5 * t_n * k, cam - 0.5 * t_n * k
        else:
            up, lo = section_at(f, frac, x_n, up_n, lo_n, x_grid)
            t_now = float(np.max(up - lo))
            if t_now > 1e-9:                                 # rescale bodily
                k = float(toc_node[i]) / t_now
                up, lo = up * k, lo * k
        dat = f"{name}_{geom_id}_{i}.dat"
        write_dat(out_dir / dat, hint + dat, x_grid, up, lo)
        dats.append(out_dir / dat)
        blocks.append((dat, i, le[i], te[i], chord[i], f))

    csv_path = out_dir / f"{name}_af.csv"
    with csv_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("# AIRFOIL METADATA CSV FILE\n\n")
        fh.write(f"Airfoil File Directory, {hint}\n\n")
        for dat, i, lep, tep, c, u in blocks:
            fh.write("#" * 40 + "\n")
            fh.write(f"Airfoil File Name, {dat}\n")
            fh.write(f"Geom Name, {name}\n")
            fh.write(f"Geom ID, {geom_id}\n")
            fh.write(f"Airfoil Index, {i}\n")
            fh.write("XSec Flag, 1\n")
            fh.write(f"XSec Index, {i}\n")
            fh.write(f"XSecSurf ID, {geom_id[:8]}SS\n")
            fh.write(f"FoilSurf u Value, {u:.6f}\n")
            fh.write(f"Global u Value, {u:.6f}\n")
            fh.write(f"Leading Edge Point, {lep[0]:.6f}, {lep[1]:.6f}, {lep[2]:.6f}\n")
            fh.write(f"Trailing Edge Point, {tep[0]:.6f}, {tep[1]:.6f}, {tep[2]:.6f}\n")
            fh.write(f"Chord, {c:.6f}\n")
            fh.write("#" * 40 + "\n\n")
    return csv_path, dats


def load_design(arc):
    """Replay the arc's design point and return (mesh_m, toc, provenance)."""
    y_a, rule, (fname, key) = ARCS[arc]
    # Prefer a t/c-optimised design point when one has been produced. The
    # section is part of the file's identity, so it is named here too -- a bundle
    # exported from the wrong section would carry the wrong contours entirely.
    airfoil = os.environ.get("ARC_AIRFOIL", "e694")
    sfx = "" if airfoil in ("", "as-built") else f"_{airfoil}"
    # Arc A is CONSTRUCTED, not optimized: its straight aft spar is a geometric
    # requirement, built to rather than searched for. Never export one that failed
    # its own constraints.
    if arc == "A":
        for prof in ("optimal", "capped"):
            pc = LOGS / f"arc_a_constructed_{prof}{sfx}.json"
            if pc.exists():
                case = json.loads(pc.read_text())
                if case.get("feasible"):
                    r = replay(case, y_a, rule)
                    return (r["mesh"], r["toc"], case.get("airfoil") or "as-built",
                            f"{pc.name} ({prof} t/c, CONSTRUCTED straight aft spar)",
                            case)
                print(f"  NOTE: {pc.name} is not feasible -- not exported")
    for prof in ("optimal", "capped"):
        p = LOGS / f"arc_optimal_toc_{arc}_{prof}{sfx}.json"
        if p.exists():
            case = json.loads(p.read_text())
            r = replay(case, y_a, rule)
            # The section the design was BUILT on, taken from the design point
            # rather than the environment, so a bundle cannot be exported on a
            # section the aero was never run with.
            sec = case.get("airfoil") or "as-built"
            return r["mesh"], r["toc"], sec, f"{p.name} ({sec}, {prof} t/c)", case
    case = json.loads((LOGS / fname).read_text())[key]
    r = replay(case, y_a, rule)
    return r["mesh"], r["toc"], "as-built", f"{fname}:{key} (as-built t/c)", case


ARCH_NOTE = {
    "A": "constant chord over region A",
    "B": "straight forward spar (wingbox_pct pinned at 0.12c)",
    "C": "free -- region A re-lofted, straight line unpinned",
}


def write_readme(path, arc, case, prov, n_stations, n_x):
    """What a reader of this bundle has to know, shipped inside the bundle.

    The .dat files and _af.csv are pure geometry in the VSP format -- nothing in
    them records which section the contours came from, how they were scaled, or
    which of the numbers behind them are still soft. That belongs with the files
    rather than in a commit message the recipient will never see.
    """
    w, R = case.get("w_wing_lb"), case.get("R_nmi")
    conv = case.get("converged")
    hist = case.get("weight_history") or []
    resid = hist[-1]["residual_lb"] if hist else None
    sec = case.get("airfoil") or "as-built"
    blend = case.get("section_blend")
    sched = case.get("rear_schedule")
    constructed = bool(case.get("constructed"))
    lines = [
        f"Arc {arc} -- {ARCH_NOTE.get(arc, '')}",
        "=" * 72,
        "",
        f"Generated by studies/vsp_planform/scripts/export_dat.py from {prov}.",
        "",
        "CONTENTS",
        f"  Arc{arc}_af.csv                  airfoil metadata, one block per station",
        f"  Arc{arc}_<GeomID>_<i>.dat        Selig contour, one per station",
        f"  {n_stations} stations, {n_x} points per contour.",
        "  Contours are normalized by local chord. Selig order: trailing edge along",
        "  the upper surface to the leading edge, then lower surface back. Leading /",
        "  Trailing Edge Point and Chord in _af.csv are INCHES in the aircraft frame.",
        "",
        "HOW THIS DESIGN WAS PRODUCED",
        ("  CONSTRUCTED, not drag-optimized. Its straight aft spar is a geometric\n"
         "  requirement, so the design is built to satisfy the constraints rather than\n"
         "  searched for: the twist and alpha come from the optimized Arc A and the\n"
         "  taper is solved on the 7 in aileron depth. Its drag is therefore FEASIBLE,\n"
         "  not best-in-class, and is about 0.5% above the drag-optimal Arc A that had\n"
         "  a kinking aft spar. Do not read it as an architecture comparison."
         if constructed else
         "  Drag-optimized within its constraint class (OAS, SLSQP): minimum drag at\n"
         "  MTOW subject to the box-width requirements, trim, and an area floor."),
        "",
    ] + ([
        "AFT SPAR -- STRAIGHT",
        f"  rear(y) = p + K/c(y) with p = {case.get('wingbox_pct', float('nan')):.4f} and "
        f"K = {case.get('K_in', float('nan')):.2f} in,",
        "  which puts the spar at CONSTANT x from root to winglet junction. Verified on",
        f"  the built geometry: its offset from the construction line varies by "
        f"{case.get('x_aft_spread_in', float('nan')):.4f} in.",
        "  K is set by the hardest-binding box-width station, not chosen.",
        "  Consequence worth knowing: constant x is a LARGER chord fraction where the",
        "  chord is smaller, so the spar moves aft in section terms going outboard --",
        f"  {sched[0][1]:.3f}c at the root to {sched[-1][1]:.3f}c at the junction. That is"
        f" why the",
        "  aileron depth had to be bought back with chord.",
        "",
    ] if (constructed and sched) else []) + [
        "SECTION",
    ] + ([
        f"  SPANWISE: {blend['inboard']} inboard, blending to {blend['outboard']} between",
        f"  {blend['f_start']*708.0:.0f} in and {blend['f_end']*708.0:.0f} in "
        f"({blend['f_start']:.0%}-{blend['f_end']:.0%} semi-span).",
        "  Camber and thickness are lofted linearly between the two; the thickness is",
        "  then scaled to each station's t/c with the camber left alone.",
        f"  Why: at the far-aft spar this design carries, {blend['inboard']} keeps only",
        f"  0.640 of its thickness while {blend['outboard']} keeps 0.811, and outboard of",
        f"  {blend['f_start']:.0%} semi-span there is little area for its poorer L/D to be paid on.",
        f"  Retention at the aileron spar: {case.get('retention_at_spar', float('nan')):.4f}",
    ] if blend else [
        f"  {sec}, thickness-scaled to each station's t/c with the CAMBER LEFT ALONE.",
        f"  c_max_t {case.get('c_max_t', float('nan')):.4f}"
        f"   thickness retention at the aft spar {case.get('retention_at_spar', float('nan')):.4f}",
        "  Scaling thickness only is deliberate: it preserves c_max_t and the",
        "  retention curve, and retention is why this section was chosen (it keeps",
        "  0.935 of its thickness at the spar where the as-built loft keeps 0.814).",
        "  It is NOT a section designed at these thicknesses -- it is this section",
        "  scaled to them.",
    ]) + [
        "",
        "THICKNESS PROFILE",
        f"  root t/c {case.get('toc_root', float('nan')):.4f} -> tip {case.get('toc_tip', float('nan')):.4f}"
        f"   (requested root {case.get('root_toc_req', float('nan')):.3f}, tip/root ratio {case.get('ratio_req', float('nan')):.2f})",
        f"  profile '{case.get('profile')}' -- set as a linear ramp on 5 spline control points.",
        "",
        "DESIGN POINT  (full OAS, MTOW 382 547 N, span pinned at 118 ft, trimmed)",
        f"  drag            {case.get('drag_N', float('nan')):9.1f} N   "
        f"(induced {case.get('induced_N', float('nan')):.1f}, viscous {case.get('viscous_N', float('nan')):.1f}, wave {case.get('wave_N', 0.0):.1f})",
        f"  S_ref           {case.get('S_ref', float('nan')) * 10.7639104:9.1f} ft2",
        f"  aft-spar depth  {case.get('depth_delivered_in', float('nan')):9.2f} in delivered "
        f"at the aileron ({case.get('depth_req_in', 7.0):.2f} required)",
        f"  wing weight     {(f'{w:9.1f} lb' if w else '  UNSIZED')}"
        f"{'' if w is None else ('   CONVERGED' if conv else '   NOT CONVERGED')}"
        f"{'' if resid is None else f' (residual {resid:+.1f} lb, tolerance 25)'}",
        f"  electric range  {(f'{R:9.1f} nmi' if R else '        -')}",
        "",
        "READ THESE BEFORE USING THE NUMBERS",
        "  * Drag is WING-ONLY. Absolutes understate aircraft drag and overstate",
        "    range; the comparisons between arcs are the point, not the absolutes.",
        "  * The t/c profile was optimised against the AS-BUILT section's retention",
        "    (0.814 at a 0.574c spar) and then applied to this design, whose retention",
        f"    at its own spar is {case.get('retention_at_spar', float('nan')):.4f}. Retention sets",
        "    c_req = depth / (retention * t/c) and therefore chord, area and weight, so",
        "    this profile is APPLIED, not jointly optimised with the section. It is the",
        "    largest open item on this geometry.",
        "  * The wing is SIZED at MTOW but FLOWN at mid-cruise weight.",
        "  * Weight comes from WingCalc through a damped bi-level fixed point. Where",
        "    it says NOT CONVERGED the loop hit its pass limit; the value is close",
        "    but carries the residual shown.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc", action="append", choices=sorted(ARCS), help="repeatable; default all")
    ap.add_argument("--n-x", type=int, default=201, help="points per contour")
    a = ap.parse_args()
    arcs = a.arc or sorted(ARCS)

    GEOMS.mkdir(parents=True, exist_ok=True)
    for arc in arcs:
        name = f"Arc{arc}"
        mesh, toc, sec, prov, case = load_design(arc)
        work = GEOMS / name
        csv_path, dats = export(mesh, toc, name, GEOM_ID[arc], work,
                                n_x=a.n_x, airfoil=sec,
                                blend=case.get("section_blend"))
        rd = write_readme(work / f"{name}_README.txt", arc, case, prov,
                          len(dats), a.n_x)
        zpath = GEOMS / f"{name}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(rd, f"{name}/{rd.name}")
            zf.write(csv_path, f"{name}/{csv_path.name}")
            for d in dats:
                zf.write(d, f"{name}/{d.name}")
        print(f"  {name}: {len(dats)} stations, {a.n_x} pts/contour -> {zpath.name} "
              f"({zpath.stat().st_size/1024:.0f} KB)   [from {prov}]")
