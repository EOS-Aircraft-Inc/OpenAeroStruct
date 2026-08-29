"""Export a design as the OpenVSP .dat bundle a colleague's tool reads.

WingCalc is fed the same thing already (coupling/geometry.py writes it into the
deck), so this is that export packaged the way the airfoil zips are shipped:

    <name>/
        <name>_af.csv                  airfoil metadata, one block per station
        <name>_spar.csv                both spars, one row per station
        <name>_<GeomID>_<i>.dat        Selig contour, one per station

A VSP bundle is an OUTER SURFACE and nothing else, so on its own it cannot say
where the wingbox sits -- and the wingbox is what every number in this study was
sized on. ``<name>_spar.csv`` is added for that: both spar lines and the box
depth at every exported station, derived from the SAME mesh and the SAME
contours as the .dat files, so the two cannot disagree.

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
from studies.vsp_planform.param import rear_spar_fraction        # noqa: E402

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


SEMI_IN = 708.0
# The aileron station the depth requirement is written at (arc_optimal_toc.Y_AIL).
# Quoted in the table header so a reader can check the headline depth without
# rebuilding anything.
Y_AIL_IN = 0.90 * SEMI_IN


def write_spar_csv(path, name, ws, chord, le, te, toc, rear, depth, box, note,
                   retention_at):
    """Both spar lines and the box depth, one row per exported station.

    The spars are the whole reason the planform is the shape it is -- the 7 in
    aileron depth sets the chord through ``depth = retention * t/c * chord`` --
    and a VSP bundle carries no such thing. So they are written beside the
    contours, in the same frame and the same units as ``_af.csv``.

    Everything here is read off the exported geometry, not off the design point:
    the chord and the spar positions come from the mesh, and the depth is
    measured on the contour at the rear spar. A reader can therefore check every
    number in the README against the files, which is the point of shipping it.

    ``retention_at(y, x)`` gives the section's thickness at chord fraction ``x``
    as a fraction of its own maximum, at ANY y. The aileron is not one of the
    exported stations, and on Arc A the section blend ends exactly there, so the
    depth has a KINK at the aileron and interpolating the rows across it reads
    0.03 in low. The requirement is written at that station, so it is computed
    there rather than interpolated.
    """
    front = float(box["front_pct"])
    y_c = float(box["y_c_start_in"])
    sched = box["rear_schedule"]

    # Both spars lie on the chord line, so x and z interpolate between the
    # leading and trailing edge points of the same station.
    def on_chord(f, col):
        return le[:, col] + f * (te[:, col] - le[:, col])

    x_f, z_f = on_chord(front, 0), on_chord(front, 2)
    x_r, z_r = on_chord(rear, 0), on_chord(rear, 2)
    width = (rear - front) * chord
    knots = ", ".join(f"({y:.1f} in, {v:.4f}c)" for y, v in sched)

    # The aileron: the station the depth requirement is written at, computed
    # there rather than interpolated between the rows that bracket it.
    wing = ws <= y_c
    rear_a = float(rear_spar_fraction(Y_AIL_IN, sched))
    chord_a = float(np.interp(Y_AIL_IN, ws, chord))
    depth_a = retention_at(Y_AIL_IN, rear_a) * float(np.interp(Y_AIL_IN, ws, toc)) * chord_a
    depth_lin = float(np.interp(Y_AIL_IN, ws, depth))

    # How straight the spar in this table actually is. The model samples the
    # spar as a schedule that is linear in y between its knots; a straight spar
    # is rear(y) = p + K/c(y), which is not linear in y. So a design that asks
    # for a straight aft spar gets one at the knots and a slight bow between
    # them, and the bundle must not quietly hide that.
    cx = te[:, 0] - le[:, 0]
    p_, K_ = box.get("wingbox_pct"), box.get("K_in")
    spread_f, spread_r = float(np.ptp(x_f[wing])), float(np.ptp(x_r[wing]))
    rule = ["#             The parameterization holds one line straight by construction,",
            f"#             at p = {p_:.4f}c. That line is INSIDE the box, not an edge of",
            "#             it (param.py); it is a spar only where p equals a spar fraction."]
    if K_:
        rule += [
            "#             This design asks for a STRAIGHT AFT spar, rear(y) = p + K/c(y)",
            f"#             with K = {K_:.2f} in. The schedule is the model's 5-knot sample",
            "#             of that rule and it interpolates LINEARLY IN Y between the",
            "#             knots, which the rule does not. On the rule itself these",
            f"#             stations hold x_rear to "
            f"{float(np.ptp((le[:, 0] + p_ * cx + K_)[wing])):.2f} in; the "
            f"{spread_r:.2f} in above is that",
            "#             plus the bow the linear schedule puts between the knots."]
    # The README quotes these same measurements, so they are recorded here rather
    # than measured twice and allowed to drift apart.
    box["x_front_spread_in"], box["x_rear_spread_in"] = spread_f, spread_r

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# {name} wingbox -- spar positions and box depth\n")
        fh.write("# INCHES, aircraft frame, the same frame and stations as "
                 f"{name}_af.csv (x aft, y right, z up).\n")
        fh.write("#\n")
        fh.write(f"# FRONT SPAR  {front:.4f}c at every station, constant.\n")
        fh.write(f"# REAR SPAR   scheduled in y: {knots}\n")
        fh.write("#             linear between the knots, held flat outside them.\n")
        fh.write(f"#             {note}\n")
        fh.write("# Both spars lie on the chord line: x = x_le + pct * (x_te - x_le),\n")
        fh.write("#   and z likewise. box_width_in is (rear_pct - front_pct) * chord_in,\n")
        fh.write("#   measured ALONG the chord line. The model constrains the same width\n")
        fh.write("#   on the chordwise x-extent instead, which is shorter by the cosine of\n")
        fh.write("#   the chord line angle -- under 0.2%, about 0.12 in on a 65 in box.\n")
        fh.write("# depth_in is the thickness of THIS station's exported contour at the\n")
        fh.write("#   rear spar, times the chord. It reproduces the study's own depth,\n")
        fh.write("#   retention * t/c * chord, because the contour carries both.\n")
        fh.write(f"# region: wing to the winglet junction at y = {y_c:.1f} in, winglet\n")
        fh.write("#   outboard of it. The winglet is welded to the junction and its box is\n")
        fh.write("#   NOT sized by this study; its rows are geometry, not a requirement.\n")
        fh.write(f"# STRAIGHTNESS over the wing rows: x_front varies by {spread_f:.2f} in,\n")
        fh.write(f"#             x_rear by {spread_r:.2f} in.\n")
        for line in rule:
            fh.write(line + "\n")
        fh.write(f"# AT THE AILERON, y = {Y_AIL_IN:.1f} in -- the station the depth\n")
        fh.write("#   requirement is written at, and NOT one of the exported stations:\n")
        fh.write(f"#   rear spar {rear_a:.4f}c, chord {chord_a:.2f} in, depth "
                 f"{depth_a:.2f} in.\n")
        fh.write("#   Computed from the section at that y. Interpolating depth_in\n")
        fh.write(f"#   between the two rows that bracket it gives {depth_lin:.2f} in "
                 f"instead"
                 + (", because\n#   the section blend ends exactly here and the depth "
                    "has a kink at it.\n" if abs(depth_a - depth_lin) > 0.01 else ".\n"))
        fh.write("station,y_in,region,chord_in,t_over_c,front_pct,rear_pct,"
                 "x_le_in,x_front_in,x_rear_in,x_te_in,z_front_in,z_rear_in,"
                 "box_width_in,depth_in\n")
        for i in range(len(ws)):
            fh.write(
                f"{i},{ws[i]:.3f},{'wing' if ws[i] <= y_c else 'winglet'},"
                f"{chord[i]:.6f},{toc[i]:.6f},{front:.4f},{rear[i]:.4f},"
                f"{le[i, 0]:.4f},{x_f[i]:.4f},{x_r[i]:.4f},{te[i, 0]:.4f},"
                f"{z_f[i]:.4f},{z_r[i]:.4f},{width[i]:.4f},{depth[i]:.4f}\n")
    return path


def export(mesh_m, toc, name, geom_id, out_dir, n_x=201, dir_hint=None, airfoil=None,
           blend=None, box=None, box_note=""):
    """Write the .dat contours, <name>_af.csv and <name>_spar.csv. Returns the paths.

    ``box`` is the wingbox this design was run with -- ``front_pct``,
    ``rear_schedule`` and the winglet junction -- as :func:`compare_classes.replay`
    returns it. Given it, the spar table is written; without it the bundle is the
    outer surface alone, which is what it used to be.
    """
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

    # The rear spar is scheduled in y and the front spar is a constant fraction,
    # so both are known at every station the contours are written at.
    rear = (rear_spar_fraction(ws, box["rear_schedule"]) if box else None)

    hint = dir_hint or f"./{name}/"
    # A named section is the same shape at every station, so its camber and
    # thickness are built once; only the t/c scale changes down the span.
    db = None if airfoil in (None, "", "as-built") else database_profile(airfoil, x_grid)
    # A spanwise pair overrides the single section: the contour changes down the span.
    db_at = blended_profile(blend, x_grid) if blend else None

    def retention_at(y_in, xc):
        """Section thickness at chord fraction ``xc``, over its own maximum, at any y.

        The same quantity the study calls RETENTION, and the reason it is worth
        having off-station: depth = retention * t/c * chord is how the aileron
        requirement sets the chord, and the aileron is between two stations.
        """
        if db_at is not None:
            t_n = db_at(y_in)[1]
        elif db is not None:
            t_n = db[1]
        else:
            f_ = 0.0 if span1 == span0 else (y_in - span0) / (span1 - span0)
            u_, l_ = section_at(f_, frac, x_n, up_n, lo_n, x_grid)
            t_n = u_ - l_
        return float(np.interp(xc, x_grid, t_n)) / float(np.max(t_n))

    blocks, dats, depths = [], [], []
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
        # Box depth read off THIS station's exported contour rather than
        # recomputed from the section: the two agree by construction -- the
        # contour carries the design's t/c and the section's retention -- and
        # reading it here is what makes the table impossible to disagree with
        # the .dat files it ships beside.
        if box:
            depths.append(float(np.interp(rear[i], x_grid, up - lo)) * chord[i])

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

    spar_path = None
    if box:
        spar_path = write_spar_csv(out_dir / f"{name}_spar.csv", name, ws, chord,
                                   le, te, toc_node, rear, np.array(depths), box,
                                   box_note, retention_at)
    return csv_path, spar_path, dats


SCHED_RECORDED = ("Source: recorded in the design point, so this is the box the "
                  "design was run on.")
# Arc B and Arc C were solved before the schedule was written into the design
# point, so replay() falls back to the study schedule for them -- the same one
# they were optimized with. Saying which of the two a bundle carries is the
# difference between a spar a reader can trust and a spar they have to check.
SCHED_DEFAULT = ("Source: the study schedule (wing5_mtow.stations_and_schedule); "
                 "this design point predates the recording of its own.")


def box_of(r, case):
    """The wingbox replay() built this wing with, plus where its schedule came from."""
    box = {k: r[k] for k in ("front_pct", "rear_schedule", "y_c_start_in")}
    # The straight-spar rule, where the design has one. The schedule is the
    # model's 5-knot sample of it, and the two are not the same line between the
    # knots -- which the spar table has to say out loud, on a design whose whole
    # claim is a straight aft spar.
    box["wingbox_pct"] = case.get("wingbox_pct")
    box["K_in"] = case.get("K_in")
    return box, (SCHED_RECORDED if case.get("rear_schedule") else SCHED_DEFAULT)


def load_design(arc):
    """Replay the arc's design point and return (mesh_m, toc, provenance, box)."""
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
                            case) + box_of(r, case)
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
            return (r["mesh"], r["toc"], sec, f"{p.name} ({sec}, {prof} t/c)",
                    case) + box_of(r, case)
    case = json.loads((LOGS / fname).read_text())[key]
    r = replay(case, y_a, rule)
    return (r["mesh"], r["toc"], "as-built", f"{fname}:{key} (as-built t/c)",
            case) + box_of(r, case)


ARCH_NOTE = {
    "A": "constant chord over region A",
    "B": "straight forward spar (wingbox_pct pinned at 0.12c)",
    "C": "free -- region A re-lofted, straight line unpinned",
}


def _wrap(text, indent, width=76):
    """Wrap one README line, hanging the continuation under its label."""
    import textwrap
    return textwrap.wrap(text, width=width, subsequent_indent=" " * indent) or [text]


def write_readme(path, arc, case, prov, n_stations, n_x, box=None, box_note=""):
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
        f"  Arc{arc}_spar.csv                both spars and the box depth, one row per station",
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
        "WINGBOX -- WHERE THE SPARS ARE",
        f"  Arc{arc}_spar.csv gives both spar lines at every one of the "
        f"{n_stations} stations,",
        "  in inches in the aircraft frame: chord fraction, x and z on the chord line,",
        "  box width and box depth. A VSP bundle is an outer surface and carries none",
        "  of that, and the box is what this wing was sized on, so it ships here.",
        f"  front spar   {box['front_pct']:.4f}c at every station, constant.",
    ] + _wrap("  rear spar    scheduled in y: "
              + ", ".join(f"({y:.1f} in, {v:.4f}c)" for y, v in box["rear_schedule"]),
              15) + [
        "               linear between the knots, held flat outside them.",
    ] + _wrap(f"               {box_note}", 15) + [
        "  depth        measured on the exported contour at the rear spar and",
        "               multiplied by the chord, so it reproduces the study's own",
        "               depth = retention * t/c * chord from the shipped files.",
        "",
    ] if box else []) + ([
        "AFT SPAR -- STRAIGHT",
        f"  rear(y) = p + K/c(y) with p = {case.get('wingbox_pct', float('nan')):.4f} and "
        f"K = {case.get('K_in', float('nan')):.2f} in,",
        "  which puts the spar at CONSTANT x from root to winglet junction. The",
        f"  schedule the model carries reproduces that rule to "
        f"{case.get('x_aft_spread_in', float('nan')):.4f} in AT ITS 5 KNOTS,",
        "  which is where the box constraints are read. Measured on the EXPORTED MESH",
        f"  the tabulated spar holds x to "
        f"{(box or {}).get('x_rear_spread_in', float('nan')):.2f} in instead: the schedule",
        "  interpolates linearly in y between the knots and the rule does not, so the",
        f"  spar bows a little between them. Arc{arc}_spar.csv reports both, and gives",
        "  the spar station by station so the bow can be seen rather than taken on trust.",
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
        mesh, toc, sec, prov, case, box, box_note = load_design(arc)
        work = GEOMS / name
        csv_path, spar_path, dats = export(mesh, toc, name, GEOM_ID[arc], work,
                                           n_x=a.n_x, airfoil=sec,
                                           blend=case.get("section_blend"),
                                           box=box, box_note=box_note)
        rd = write_readme(work / f"{name}_README.txt", arc, case, prov,
                          len(dats), a.n_x, box=box, box_note=box_note)
        zpath = GEOMS / f"{name}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(rd, f"{name}/{rd.name}")
            zf.write(csv_path, f"{name}/{csv_path.name}")
            zf.write(spar_path, f"{name}/{spar_path.name}")
            for d in dats:
                zf.write(d, f"{name}/{d.name}")
        print(f"  {name}: {len(dats)} stations, {a.n_x} pts/contour -> {zpath.name} "
              f"({zpath.stat().st_size/1024:.0f} KB)   [from {prov}]")
