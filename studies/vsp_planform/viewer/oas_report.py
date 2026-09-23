"""The OAS wing report: a snapshot in, WingCalc's page out.

This module does one thing -- it reshapes an ``oas_snapshot`` dict into the
payload dicts ``wingcalc_frontend.build_html`` already knows how to draw. It
computes nothing structural and runs no model; if a number is on the page, it
came out of the snapshot, and if it is not in the snapshot it is not on the
page.

WHY THE SHAPES ARE WINGCALC'S AND NOT OAS'S
-------------------------------------------
The report exists to be read next to WingCalc's, so a divergence stands out as
a divergence rather than as two differently-drawn charts. That is only worth
anything if the drawing is identical, which means the frontend has to be the
same code, which means the data has to be the same shape. So a bay here is
WingCalc's bay dict, a margin record is WingCalc's record dict, and the
station tables carry WingCalc's ``{property, units, category, description,
kind, value, decimals}`` rows -- even where the underlying quantity is
something OAS computes quite differently.

WHERE THE TWO MODELS GENUINELY PART
-----------------------------------
Four differences are structural, not cosmetic, and every one of them is stated
on the page it affects rather than only here:

* **The wing root.** WingCalc reacts the wing at the fuselage, at BL 54, and
  carries no lift inboard of it. OAS clamps the FEM at the centreline and lifts
  all the way in. So OAS's root bending moment is the full cantilever moment and
  WingCalc's is not, and the two only agree once both are read at the same
  station -- which is what the VMT tab is for.
* **The span.** WingCalc's wingbox runs to WS 678. OAS's outermost structural
  node inboard of the winglet is the B|C junction at 674.95 in. 3.05 in, 0.45%.
* **The section.** OAS has no stringers, no spar caps, no plies, no buckling, no
  crippling and no Tsai-Wu. Its wingbox is two smeared thicknesses of one
  isotropic-equivalent laminate, with four von Mises stress points per element.
  Every table that would list those things carries an empty state saying so.
* **The weight.** OAS sizes the box. The other 3,437 lb of the calibrated wing
  -- ribs, joints, splices, reinforcements, paint, mesh, secondary -- is a
  constant carried from WingCalc's own split, with no spanwise distribution
  behind it, and is labelled as such everywhere it appears.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from studies.vsp_planform.viewer import oas_snapshot as osnap
from studies.vsp_planform.viewer.wingcalc_frontend import build_html

REPORT_TITLE = "OAS Wing Report"

# The material reaches OAS as one isotropic equivalent, so this is the name on
# every component the page draws. It is the deck's own laminate; what is lost in
# the reduction is the ply stack, not the stiffness.
MATERIAL = "UD HARD (isotropic equiv.)"
NO_PLIES = "n/a"

# ``vonmises[elem, k]`` -- the four stress combinations
# ``openaerostruct/structures/vonmises_wingbox.py`` computes, and what each one
# is a margin ON.
#
# The allowable is the subtlety. OAS divides combinations 0 and 3 by
# ``strength_factor_for_upper_skin`` (= sigma_c / sigma_t) inside the stress
# calculation and then tests every combination against ``yield / safety_factor``
# with ``yield = sigma_t``. So the margin is uniformly
# ``(sigma_t / SF) / vonmises[k] - 1``, and the PHYSICAL stress at a
# compression-critical point is ``vonmises[k] * sigma_c / sigma_t``. Both are
# reported: the margin because it is what the optimizer constrained, and the raw
# stress because a stress quoted against an allowable it was already scaled by
# is not a stress anyone can check.
_MS_POINTS = (
    (0, "Upper skin", "skin", "compression",
     "top-skin bending + rear-spar bending + axial, with torsion shear"),
    (1, "Lower skin", "skin", "tension",
     "bottom-skin bending + front-spar bending + axial, with torsion shear"),
    (2, "Fwd", "spar_web", "tension",
     "front-spar bending + axial, with torsion less vertical shear"),
    (3, "Aft", "spar_web", "compression",
     "rear-spar bending + axial, with torsion plus vertical shear"),
)

# 999 is the frontend's "no data" sentinel: msClass() treats anything >= 900 as
# unconstrained and formatMsVal() prints it as an em dash. Used for the
# categories OAS has no check for, so the overview boxes say "no result" rather
# than "infinite margin".
NO_MS = 999.0

_COMPONENTS_NOTE = (
    "OpenAeroStruct has no section components to list. Its wingbox is two smeared "
    "thicknesses -- one skin, one spar -- of a single isotropic-equivalent laminate, "
    "so there are no stringers, no skin panels, no spar caps and no ply counts. The "
    "thicknesses themselves are real: they are design variables of the sizing run, "
    "and they are in the Cross Section Properties table above and drawn to scale on "
    "the section beside it."
)


# ---------------------------------------------------------------------------
# Row helpers -- WingCalc's table contract
# ---------------------------------------------------------------------------
def _row(prop, value, units, category, description, decimals=3):
    """One ``props-table`` row. ``kind`` follows the value's own type."""
    if isinstance(value, str):
        return {"property": prop, "value": value, "units": units,
                "category": category, "description": description, "kind": "str"}
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return {"property": prop, "value": int(value), "units": units,
                "category": category, "description": description, "kind": "int"}
    return {"property": prop, "value": float(value), "units": units,
            "category": category, "description": description,
            "kind": "float", "decimals": decimals}


def _ov(prop, value, units, decimals=2):
    """One ``Planform overview`` row -- no category, no description column."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return {"property": prop, "value": int(value), "units": units, "kind": "int"}
    return {"property": prop, "value": float(value), "units": units,
            "kind": "float", "decimals": decimals}


def _a(snap, *path):
    """A snapshot array as a float numpy array, with Nones as NaN."""
    node = snap
    for p in path:
        node = node[p]
    return np.array([np.nan if v is None else float(v) for v in node], dtype=float)


def _sweep_deg(y, x):
    """Sweep of a spanwise line, per segment, positive aft."""
    return np.degrees(np.arctan2(np.diff(x), np.diff(y)))


# ---------------------------------------------------------------------------
# Planform
# ---------------------------------------------------------------------------
def _planform_input_rows(snap):
    """The design, the flight condition, the sizing case and the material.

    This is the OAS analogue of ``planformIn.csv``: everything that was an INPUT
    to the run, plus the handful of derived quantities that are worth reading
    beside them. It deliberately does NOT mirror WingCalc's row list -- half of
    those rows are rib pitches, cut-out locations and control-surface factors
    that have no OAS counterpart, and a column of blanks says less than a
    shorter table does.
    """
    d, fl, sz, g, st = (snap["design"], snap["flight"], snap["sizing"],
                        snap["grid"], snap["stations"])
    chord = _a(snap, "stations", "chord_in")
    rows = [
        _row("Total wingbox span", 2.0 * g["cut_ws_in"], "in", "Geom",
             "Twice the outermost structural node inboard of the winglet. The OAS mesh "
             "puts that node on the region B|C junction at 674.95 in; WingCalc's wingbox "
             "runs to 678.00, so the two structural spans differ by 3.05 in (0.45%).", 2),
        _row("Total span (incl. winglet)", 2.0 * g["semispan_full_in"], "in", "Geom",
             "The full aerodynamic span. The winglet is drawn on the planform and "
             "excluded from every structural total on this report.", 2),
        _row("Root chord", float(chord[0]), "in", "Geom",
             "Chord at the centreline station of the OAS mesh.", 4),
        _row("Chord at the wingbox tip", float(chord[-1]), "in", "Geom",
             "Chord at the B|C junction, where the structure is cut.", 3),
        _row("taper_B", d["taper_B"], "-", "Design",
             "Tip chord of region B over the chord at the A|B junction. Solved in closed "
             "form from the 7 in aileron depth requirement, then held: it is frozen "
             "through the whole aerostructural run.", 5),
        _row("wingbox_pct", d["wingbox_pct"], "-", "Design",
             "Chord fraction of the straight, unswept line the planform is built around. "
             "It is what generates region B's sweep, and it is held.", 4),
        _row("Fwd spar x/c", float(st["fwd_ratio"][0]), "-", "Geom",
             "Front spar, constant fraction of chord. The wingbox skin coordinates OAS "
             "integrates run between this and the aft spar.", 4),
        _row("Aft spar x/c (root)", float(st["aft_ratio"][0]), "-", "Geom",
             "Rear spar at the root. Arc A's schedule is constant, so it is the same at "
             "every station; a kinking schedule would vary here.", 4),
        _row("Elastic axis x/c (root)",
             float((_a(snap, "stations", "x_ea_in")[0] - _a(snap, "stations", "x_le_in")[0])
                   / chord[0]), "-", "Geom",
             "Where OAS's FEM nodes sit chordwise. WingCalc derives its elastic axis from "
             "the spar distribution instead, around 0.43c, so the two tools take their "
             "torque about different lines -- which is most of why the two Mx curves "
             "differ in sign as well as in size.", 4),
        _row("Inboard nacelle Y", sz["nacelles"][0]["ws_in"], "in", "Geom",
             "Wing station of the inboard nacelle point mass.", 2),
        _row("Outboard nacelle Y", sz["nacelles"][1]["ws_in"], "in", "Geom",
             "Wing station of the outboard nacelle point mass.", 2),
        _row("Inboard section", d["airfoil_inboard"], "-", "Geom",
             "The section the wingbox skin coordinates are taken from, at every station. "
             "OAS itself never sees a shape -- only t/c and c_max_t -- so this enters the "
             "structure and not the aerodynamics."),
        _row("Outboard section", d["airfoil_outboard"], "-", "Geom",
             "The design lofts to this section between "
             f"{100 * d['blend_start_frac']:.0f}% and {100 * d['blend_end_frac']:.0f}% of "
             "semi-span. arc_aerostruct hands the FEM the INBOARD section's box "
             "coordinates at every station regardless, on the grounds that the box mass "
             "is overwhelmingly inboard of the blend, so the sections drawn on this "
             "report are the inboard one throughout."),
        _row("t/c root", float(d["t_over_c_cp"][0]), "-", "Design",
             "Root control point of the t/c spline. Held through the run.", 4),
        _row("t/c tip", float(d["t_over_c_cp"][-1]), "-", "Design",
             "Tip control point of the t/c spline. Held through the run.", 4),
        _row("Twist root", float(st["twist_deg"][0]), "deg", "Design",
             "Absolute geometric twist at the root, from the converged spline.", 3),
        _row("Twist at the wingbox tip", float(st["twist_deg"][-1]), "deg", "Design",
             "Absolute geometric twist at the B|C junction.", 3),
        _row("alpha", d["alpha_deg"], "deg", "Design",
             "Trim angle of attack at the cruise point.", 4),
        _row("Number of FEM elements", g["n_elements"], "-", "Design",
             "Spanwise wingbox elements inside the cut. Each is one row of every "
             "structural array on this report -- WingCalc's 20 bays against these, which "
             "is why every distribution is also drawn resampled onto its stations.", 0),
    ]

    rows += [
        _row("KTAS", fl["ktas"], "kt", "Flight", "True airspeed of the cruise point.", 1),
        _row("Altitude", fl["altitude_ft"], "ft", "Flight", "ISA pressure altitude.", 0),
        _row("Mach", fl["mach"], "-", "Flight", "Derived in atmosphere.py, not hardcoded.", 6),
        _row("rho", fl["rho_kgm3"], "kg/m^3", "Flight", "ISA density at the cruise point.", 6),
        _row("CL", fl["CL"], "-", "Flight", "Trimmed wing lift coefficient.", 5),
        _row("CD", fl["CD"], "-", "Flight",
             "Wing drag coefficient: induced plus viscous plus wave. Wing only -- the OAS "
             "model carries no fuselage, so absolute drag understates the aircraft.", 6),
        _row("Cruise lift", fl["lift_lb"], "lb", "Flight",
             "Full-wing lift at the trimmed cruise point, 1 g at MTOW.", 1),
        _row("Cruise drag", fl["drag_N"] / osnap.LB_N, "lb", "Flight",
             "Full-wing drag at the trimmed cruise point.", 2),
    ]

    rows += [
        _row("Load case", sz["case_name"], "-", "Load case",
             "The one case OAS sizes against, taken from the deck's loadCasesIn.csv. "
             "WingCalc sizes against eight; this report has one, so a margin here is a "
             "margin for this case alone."),
        _row("Nz", sz["Nz"], "-", "Load case",
             "Limit load factor. The loads on this report are LIMIT loads; the ultimate "
             "factor is applied inside the failure check, not to these arrays.", 2),
        _row("Maneuver weight", sz["W_maneuver_lb"], "lb", "Load case",
             "Aircraft weight the maneuver case is flown at. Fuel_fraction is 0 in this "
             "case, so there is no wing-fuel relief -- which is why the Loading tab's "
             "fuel curve is flat at zero.", 0),
        _row("Ultimate factor", sz["safety_factor"], "-", "Load case",
             "Applied by failure_ks.py, which tests each von Mises combination against "
             "yield / safety_factor.", 2),
        _row("Spanload scale", sz["aero_scale"], "-", "Load case",
             "Nz * W_maneuver / W_cruise. The sizing spanload is the 1 g MTOW one scaled "
             "by this, which is exactly what the deck exports and WingCalc then scales -- "
             "so both tools are fed the same distribution.", 4),
        _row("Inboard nacelle mass", sz["nacelles"][0]["weight_lb"], "lb", "Load case",
             "Nacelle PLUS the main gear inside it, which is the installed mass every "
             "consumer of it wants except the fitting check "
             "(io/inputs.py::inboard_nacelle_weight).", 2),
        _row("Outboard nacelle mass", sz["nacelles"][1]["weight_lb"], "lb", "Load case",
             "Nacelle alone; no gear outboard.", 2),
        _row("Inboard nacelle x/c", sz["nacelles"][0]["xc"], "-", "Load case",
             "Chordwise CG position aft of the local leading edge, gear-retracted row. "
             "It sits FORWARD of the elastic axis, so it is a torque arm at every g -- "
             "which is what sizes the spar webs.", 3),
        _row("Outboard nacelle x/c", sz["nacelles"][1]["xc"], "-", "Load case",
             "Chordwise CG position aft of the local leading edge.", 3),
        _row("Inboard nacelle dZ", sz["nacelles"][0]["dz_in"], "in", "Load case",
             "CG height above the elastic axis, gear-retracted row.", 2),
        _row("Outboard nacelle dZ", sz["nacelles"][1]["dz_in"], "in", "Load case",
             "CG height above the elastic axis.", 2),
    ]

    rows += [
        _row("Material", sz["material"], "-", "Material",
             "The deck's Materials.csv card gives smeared LAMINATE properties for three "
             "layups, not lamina data, so OAS is fed one isotropic equivalent. That is "
             "also why there is no Tsai-Wu path: it needs E2 and sigma_t2, which the card "
             "does not carry."),
        _row("E", sz["E_psi"], "psi", "Material", "Laminate Young's modulus.", 0),
        _row("G", sz["G_psi"], "psi", "Material", "Laminate shear modulus.", 0),
        _row("Allowable, tension", sz["sigma_tension_psi"], "psi", "Material",
             "Open-hole tension. The allowable for the lower skin and the front spar.", 0),
        _row("Allowable, compression", sz["sigma_compression_psi"], "psi", "Material",
             "min(CAI, OHC). Reaches OAS through strength_factor_for_upper_skin, and is "
             "the allowable for the upper skin and the rear spar.", 0),
        _row("Density", sz["mrho_lb_in3"], "lb/in^3", "Material",
             "Laminate density. Every weight on this report is this times a sized area "
             "times a length.", 4),
    ]

    w = snap["weight"]
    rows += [
        _row("KS failure aggregate", sz["failure_ks"], "-", "Weight",
             "The constraint the sizing run drove to zero. A single Kreisselmeier-"
             "Steinhauser aggregate over every element and all four stress points, so "
             "zero means the most critical point is exactly at its allowable.", 6),
    ]
    t = w["buildup"]["terms"]
    rows += [
        _row("Wing weight, build-up", t["W_wing"], "lb", "Weight",
             "WingCalc's build-up (weight_calc.compute_wing_weight), computed by OAS from "
             "its own sized box and geometry: ribs, fasteners, splices, reinforcements, "
             "ply drops, Torenbeek secondary structure, lightning mesh and paint. No "
             "WingCalc output enters it.", 2),
        _row("Nacelle web up-to-cap plies", 0.0, "lb", "Weight",
             "The one build-up rule OAS cannot apply: WingCalc brings each nacelle spar "
             "web up to its own sized cap, and OAS's box has one smeared spar thickness "
             "and no cap. Carried at zero; about 34 lb at the Arc A datum.", 2),
    ]
    for n in snap["sizing"]["nacelles"]:
        rows.append(_row(f"{n.get('name', '')} nacelle".strip(), n["weight_lb"], "lb", "Loading",
                         f"At WS {n['ws_in']:.1f}, x/c {n['xc']:.3f} aft of the local LE and "
                         f"{n['dz_in']:+.2f} in above the elastic axis -- WingCalc's reading "
                         "of wingLoadingIn.csv. The inboard one carries the main gear.", 2))
    return rows


def _reference_area_rows(snap):
    """Reference areas, WingCalc's nine, as the OAS build-up computed them.

    Same labels, same definitions, same order as WingCalc's own table
    (``wing_viewer._reference_area_rows``): the build-up integrates them on the OAS
    chord distribution by WingCalc's rules, because the Torenbeek terms are cast in
    them. The VLM's own S_ref -- winglet included -- is given last, for contrast.
    """
    a = snap["weight"]["buildup"]["reference_areas"]
    specs = [
        ("Sref", "Sref_ft2", "Chord-integrated planform area over the wingbox span (winglet excluded)."),
        ("S_wing", "S_wing_ft2", "Integrated wingbox planform area plus projected winglet area."),
        ("S_LE", "S_le_ft2", "Exposed leading-edge area ahead of the front spar."),
        ("S_fixed_TE", "S_fixed_te_ft2",
         "Fixed trailing-edge shroud area: rear spar back to the stowed movable's cg line, "
         "full depth where no movable sits behind the spar. MFS panels are not deducted."),
        ("S_fixed_TE_flap_aligned", "S_fixed_te_flap_aligned_ft2",
         "Fixed trailing-edge area aligned with the flap span."),
        ("S_flaps", "S_flap_ft2", "Movable trailing-edge flap plan area."),
        ("S_aileron", "S_aileron_ft2", "Outboard aileron plan area."),
        ("S_MFS", "S_mfs_ft2", "Multifunction spoiler plan area."),
        ("S_winglet", "S_winglet_ft2", "Projected winglet plan area."),
    ]
    rows = [_row(lab, a[k], "ft^2", "Derived area", d, 2) for lab, k, d in specs if k in a]
    rows.append(_row(
        "S_ref (VLM)", snap["flight"]["S_ref_m2"] * (1.0 / 0.3048**2), "ft^2", "Derived area",
        "The VLM's projected area, winglet included -- the one the aero coefficients are "
        "referred to. Compare it with S_wing, not Sref.", 2))
    return rows


def _planform_overview_rows(snap):
    g, fl = snap["grid"], snap["flight"]
    ws = _a(snap, "stations", "ws_in")
    chord = _a(snap, "stations", "chord_in")
    x_le = _a(snap, "stations", "x_le_in")
    x_te = _a(snap, "stations", "x_te_in")
    x_qc = _a(snap, "stations", "x_qc_in")
    s_ref_ft2 = fl["S_ref_m2"] / 0.3048**2

    # Area inboard of the cut, by the trapezoid rule on the mesh's own chords --
    # the winglet-excluded area, so it is the number to read against WingCalc's
    # Sref rather than against its S_wing.
    s_box_ft2 = 2.0 * float(np.trapezoid(chord, ws)) / 144.0
    span_box = 2.0 * g["cut_ws_in"]
    span_full = 2.0 * g["semispan_full_in"]

    # Area-weighted t/c, over the elements inside the cut.
    toc = np.array([e["t_c"] for e in snap["elements"]])
    e_chord = np.array([e["chord_in"] for e in snap["elements"]])
    e_width = np.array([e["width_in"] for e in snap["elements"]])
    area_w = e_chord * e_width

    def _line_sweep(x):
        """One representative sweep for a spanwise line: root to the cut."""
        return float(np.degrees(np.arctan2(x[-1] - x[0], ws[-1] - ws[0])))

    return [
        _ov("Sref (VLM, incl. winglet)", s_ref_ft2, "ft^2"),
        _ov("Wingbox planform area (winglet excluded)", s_box_ft2, "ft^2"),
        _ov("Aspect Ratio = b^2/Sref", span_full**2 / 144.0 / s_ref_ft2, "-"),
        _ov("Total span", span_full, "in"),
        _ov("Wingbox span, Bref", span_box, "in"),
        _ov("Area-weighted t/c", float((toc * area_w).sum() / area_w.sum()), "-", 4),
        _ov("LE sweep (root to wingbox tip)", _line_sweep(x_le), "deg", 3),
        _ov("1/4 sweep (root to wingbox tip)", _line_sweep(x_qc), "deg", 3),
        _ov("TE sweep (root to wingbox tip)", _line_sweep(x_te), "deg", 3),
        _ov("Elastic axis sweep", _line_sweep(_a(snap, "stations", "x_ea_in")), "deg", 3),
        _ov("FEM elements per half wing", g["n_elements"], "count"),
        _ov("Mesh stations per half wing (full)", g["n_nodes_full"], "count"),
        _ov("Sized box weight (both wings, winglet excluded)",
            2.0 * snap["weight"]["box_half_lb"], "lb"),
        _ov("Calibrated wing weight", snap["weight"]["W_wing_lb"], "lb"),
        _ov("Cruise drag", fl["drag_N"] / osnap.LB_N, "lb"),
        _ov("Electric range", snap["weight"]["R_nmi"], "nmi"),
    ]


def _planform_bay_geometry(snap):
    """Per-element taper, sweeps and dihedral, as a column-per-element table.

    WingCalc's equivalent has 19 columns; this has one per FEM element, which is
    27. It is the same table read at a finer spacing, and the taper row is the
    one place the two differ in definition -- WingCalc extrapolates each segment
    to an equivalent full-span taper, this reports the segment's own chord ratio,
    because on a 25 in element the extrapolation is dominated by the
    extrapolation.
    """
    ws = _a(snap, "stations", "ws_in")
    chord = _a(snap, "stations", "chord_in")
    z_ea = _a(snap, "stations", "z_ea_in")
    cols = [{"bay_id": e["index"] + 1,
             "ws_inboard": round(e["ws_left_in"], 2),
             "ws_outboard": round(e["ws_right_in"], 2)} for e in snap["elements"]]
    n = len(cols)

    def seg(x):
        return _sweep_deg(ws, x)[:n]

    metrics = [
        ("Taper ratio (c_out / c_in)", (chord[1:] / chord[:-1])[:n], 4),
        ("LE sweep (deg)", seg(_a(snap, "stations", "x_le_in")), 3),
        ("Fwd spar sweep (deg)", seg(_a(snap, "stations", "x_fwd_in")), 3),
        ("1/4 sweep (deg)", seg(_a(snap, "stations", "x_qc_in")), 3),
        ("Elastic axis sweep (deg)", seg(_a(snap, "stations", "x_ea_in")), 3),
        ("Aft spar sweep (deg)", seg(_a(snap, "stations", "x_aft_in")), 3),
        ("TE sweep (deg)", seg(_a(snap, "stations", "x_te_in")), 3),
        ("Dihedral (deg)", _sweep_deg(ws, z_ea)[:n], 3),
        ("t/c", np.array([e["t_c"] for e in snap["elements"]]), 4),
        ("Skin thickness (in)", _a(snap, "struct", "skin_t_in"), 4),
        ("Spar thickness (in)", _a(snap, "struct", "spar_t_in"), 4),
    ]
    return {"columns": cols,
            "rows": [{"label": lab, "values": [round(float(v), d) for v in vals],
                      "decimals": d} for lab, vals, d in metrics]}


def _planform_axis_systems(snap):
    """The local wing triad drawn against the global one, on the LEFT wing.

    Same convention as WingCalc's key: x' outboard along the elastic axis, y'
    aft, z' up, taken on the left wing because that is the half the key is drawn
    over and the half WingCalc's own local system is defined on. Taken at the
    element nearest mid-span -- the key answers "what is the local system", not
    "where is it", and both ends of a wing are atypical.
    """
    ws = _a(snap, "stations", "ws_in")
    x_ea = _a(snap, "stations", "x_ea_in")
    z_ea = _a(snap, "stations", "z_ea_in")
    i = int(np.argmin(np.abs(ws - 0.5 * ws[-1])))
    i = min(max(i, 0), len(ws) - 2)
    # Outboard on the LEFT wing is -Y, so the spanwise step is negated.
    d = np.array([x_ea[i + 1] - x_ea[i], -(ws[i + 1] - ws[i]), z_ea[i + 1] - z_ea[i]])
    e_x = d / np.linalg.norm(d)
    e_z = np.cross(e_x, [1.0, 0.0, 0.0])
    e_z /= np.linalg.norm(e_z)
    e_y = np.cross(e_z, e_x)
    return {
        "bay_id": i + 1,
        "ws": round(float(0.5 * (ws[i] + ws[i + 1])), 1),
        "sweep_ea_deg": round(float(np.degrees(np.arctan2(e_x[0], abs(e_x[1])))), 2),
        "dihedral_deg": round(float(np.degrees(np.arcsin(np.clip(e_x[2], -1, 1)))), 2),
        "e_x": [round(float(c), 6) for c in e_x],
        "e_y": [round(float(c), 6) for c in e_y],
    }


def _winglet_polygons(snap):
    """The winglet outline, mirrored, for the planform plot.

    Drawn in the same faint wash WingCalc uses for it, and for the same reason:
    the wing has one and leaving it off the drawing makes the planform wrong.
    It is excluded from every structural total on this report, which is what
    WingCalc's "Excludes : Flaps, ailerons and winglets" means as well.
    """
    w = snap["winglet"]
    ws, le, te = w["ws_in"], w["x_le_in"], w["x_te_in"]
    pts = ([(y, x) for y, x in zip(ws, le)]
           + [(y, x) for y, x in zip(reversed(ws), reversed(te))])
    out = []
    for sign in (-1.0, 1.0):
        p = [(round(sign * y, 2), round(x, 2)) for y, x in pts]
        p.append(p[0])
        out.append({"ws": [q[0] for q in p], "x": [q[1] for q in p]})
    return out


def _extract_planform(snap):
    ws = _a(snap, "stations", "ws_in")
    le = _a(snap, "stations", "x_le_in")
    te = _a(snap, "stations", "x_te_in")
    fwd = _a(snap, "stations", "x_fwd_in")
    aft = _a(snap, "stations", "x_aft_in")
    qc = _a(snap, "stations", "x_qc_in")
    ea = _a(snap, "stations", "x_ea_in")

    def mirror(a):
        """Root-to-tip half wing to a full-span polyline, port first."""
        return np.concatenate([a[::-1], a[1:]]).tolist()

    ws_full = np.concatenate([-ws[::-1], ws[1:]]).tolist()

    # ``bays`` are the FEM ELEMENTS, at their mid-stations, because that is what
    # RESULTS.bays and BAYS are: one entry per element, and the Stress Results
    # overview pairs them off by index. ``ribs`` is the separate list of node
    # stations the rib lines are drawn at -- WingCalc puts a bay at each rib, so
    # its payload needs only the one array. See the frontend's [OAS] note.
    elems = snap["elements"]

    def at(a, y):
        return round(float(np.interp(y, ws, a)), 2)

    bays = [{
        "bay_id": e["index"] + 1,
        "ws": round(e["ws_in"], 2),
        "le": at(le, e["ws_in"]), "te": at(te, e["ws_in"]),
        "fwd": at(fwd, e["ws_in"]), "aft": at(aft, e["ws_in"]),
        "quarter_chord": at(qc, e["ws_in"]), "elastic_axis": at(ea, e["ws_in"]),
    } for e in elems]
    ribs = [{"bay_id": i, "ws": round(float(ws[i]), 2),
             "fwd": round(float(fwd[i]), 2), "aft": round(float(aft[i]), 2)}
            for i in range(len(ws))]
    bay_edges = [{"ws_left": round(e["ws_left_in"], 2),
                  "ws_right": round(e["ws_right_in"], 2)} for e in elems]
    # Thinned to roughly WingCalc's own label spacing. The mesh is cosine-clustered,
    # so the outboard elements are 4 in wide against 39 in at the root, and one number
    # per element outboard overprints into a smudge. Same rule as the Stress Results
    # overview's callouts (OVERVIEW_ANNO_MIN_GAP in the frontend): walk outboard, keep
    # a label only once it clears the last one kept. Every element is still in both
    # dropdowns and in every table.
    bay_labels, last = [], -1e9
    min_gap = 0.038 * float(ws[-1])
    for b in bays:
        if b["ws"] - last < min_gap:
            continue
        last = b["ws"]
        bay_labels.append({"text": str(b["bay_id"]), "ws": b["ws"],
                           "x": round(0.5 * (b["fwd"] + b["aft"]), 2)})

    return {
        "ws": ws_full, "le": mirror(le), "te": mirror(te),
        "fwd": mirror(fwd), "aft": mirror(aft),
        # No stringer ladder: OAS's wingbox has no stringers at all.
        "stringers_upper": [], "stringers_lower": [],
        "nacelle_ib_y": snap["sizing"]["nacelles"][0]["ws_in"],
        "nacelle_ob_y": snap["sizing"]["nacelles"][1]["ws_in"],
        "overview_rows": _planform_overview_rows(snap),
        "bay_geometry": _planform_bay_geometry(snap),
        "axis_systems": _planform_axis_systems(snap),
        "input_rows": _planform_input_rows(snap) + _reference_area_rows(snap),
        # Only the winglet. OAS models no flap, aileron or spoiler, so those
        # regions are empty rather than guessed at from the WingCalc deck.
        "surfaces": {"flap": [], "aileron": [], "mfs1": [], "mfs2": [],
                     "winglet": _winglet_polygons(snap)},
        "semispan": {
            "ws": ws.tolist(), "le": le.tolist(), "te": te.tolist(),
            "fwd": fwd.tolist(), "aft": aft.tolist(),
            "quarter_chord": qc.tolist(), "elastic_axis": ea.tolist(),
            "bays": bays, "ribs": ribs,
            "bay_edges": bay_edges, "bay_labels": bay_labels,
        },
    }


# ---------------------------------------------------------------------------
# Cross sections
# ---------------------------------------------------------------------------
def _skin_panel(e, upper):
    """The whole box skin as one panel, which is what OAS sized.

    WingCalc splits each surface into a panel between every pair of stringers
    and sizes each one; OAS carries a single thickness for the whole upper-plus-
    lower skin. So there is one panel per surface here, spanning spar to spar,
    and its thickness is a design variable of the sizing run rather than a ply
    count rounded to an integer.
    """
    sk = e["skin_upper"] if upper else e["skin_lower"]
    return {
        "name": "Upper skin" if upper else "Lower skin",
        "material": MATERIAL,
        "plies": NO_PLIES,
        "thickness": e["t_skin_in"],
        "panel_length": sk["length_in"],
        "area": sk["length_in"] * e["t_skin_in"],
        "bay_width": e["width_in"],
        "ex_scale": 1.0,
        "R_chord": sk["r_chord_in"],
        "x": sk["cx_in"],
        "z": sk["cz_sec_in"],
        "x_left": e["chord_in"] * e["fwd_ratio"],
        "x_right": e["chord_in"] * e["aft_ratio"],
        "surface": "upper" if upper else "lower",
    }


def _spar(e, aft):
    """One shear web, as drawn and as sized. No caps: OAS has none.

    The web rectangle is real: its height is the local section depth at the spar
    station and its thickness is the sized ``spar_thickness``. The four cap-leg
    polygons WingCalc draws are left null, which the frontend skips -- so the
    section shows a box with two webs and two skins, which is exactly the box
    ``section_properties_wingbox`` integrated.
    """
    x = e["chord_in"] * (e["aft_ratio"] if aft else e["fwd_ratio"])
    z_up = e["z_up_aft_in"] if aft else e["z_up_fwd_in"]
    z_lo = e["z_lo_aft_in"] if aft else e["z_lo_fwd_in"]
    t = e["t_spar_in"]
    h = z_up - z_lo
    poly = {"x": [x - t / 2, x + t / 2, x + t / 2, x - t / 2, x - t / 2],
            "z": [z_lo, z_lo, z_up, z_up, z_lo]}
    web = {"name": ("Aft" if aft else "Fwd") + " spar web", "material": MATERIAL,
           "plies": NO_PLIES, "height": h, "thickness": t, "area": h * t,
           "x": x, "z": 0.5 * (z_up + z_lo)}
    return {"web": web, "upper_cap": {}, "lower_cap": {},
            "draw": {"web": poly, "upper_skin_leg": None, "upper_web_leg": None,
                     "lower_skin_leg": None, "lower_web_leg": None}}


def _extract_xsec(snap):
    """``AIRFOIL`` and one ``BAYS`` entry per FEM element."""
    sz = snap["sizing"]
    e0 = snap["elements"][0]
    airfoil = dict(e0["airfoil"])
    ws = _a(snap, "stations", "ws_in")
    x_ea = _a(snap, "stations", "x_ea_in")
    z_ea = _a(snap, "stations", "z_ea_in")
    sweep = _sweep_deg(ws, x_ea)
    dihedral = _sweep_deg(ws, z_ea)
    S = snap["struct"]

    bays = []
    for i, e in enumerate(snap["elements"]):
        A = S["A_in2"][i]
        iy_chordwise = S["Iy_in4"][i]     # OAS Iy is the CHORDWISE bending inertia
        iz_vertical = S["Iz_in4"][i]      # OAS Iz is the VERTICAL bending inertia
        bays.append({
            "bay_id": i + 1,
            "ws": e["ws_in"],
            "chord": e["chord_in"],
            "box_chord": e["box_chord_in"],
            "t_c": e["t_c"],
            "geometry_source": "OAS mesh",
            "local_sweep_deg": float(sweep[i]),
            "local_dihedral_deg": float(dihedral[i]),
            "airfoil": e["airfoil"],
            "fwd_ratio": e["fwd_ratio"],
            "aft_ratio": e["aft_ratio"],
            "spar_h_fwd": e["spar_h_fwd_in"],
            "spar_h_aft": e["spar_h_aft_in"],
            "E": sz["E_psi"],
            "G_ref": sz["G_psi"],
            # One material, so the transformed area and the physical area are the
            # same number. They are both reported because WingCalc's table has
            # both and a reader comparing the two should see that here they agree
            # by construction rather than by coincidence.
            "A": A, "A_geom": A, "EA": sz["E_psi"] * A,
            "YCG": e["y_na_sec_in"], "ZCG": e["z_na_sec_in"],
            "YCG_geom": e["y_na_sec_in"], "ZCG_geom": e["z_na_sec_in"],
            "Qy": None, "Qz": S["Qz_in3"][i],
            "IYCG": iz_vertical, "IZCG": iy_chordwise,
            # No product of inertia: section_properties_wingbox computes the two
            # principal-ish inertias about its own neutral axes and never forms
            # Iyz. Null rather than zero -- zero is a claim.
            "IYZCG": None,
            "J": S["J_in4"][i],
            "GJ": sz["G_psi"] * S["J_in4"][i],
            "Am": S["A_enc_in2"][i],
            "upper_skin_panels": [_skin_panel(e, True)],
            "lower_skin_panels": [_skin_panel(e, False)],
            "upper_stringers": [], "lower_stringers": [],
            "cutout_stg_index": None, "cutout_panel_name": None,
            "fwd_spar": _spar(e, aft=False),
            "aft_spar": _spar(e, aft=True),
            # No flap, aileron or spoiler in the OAS model, so no outline to draw.
            "control_surfaces": [],
            "components_note": _COMPONENTS_NOTE,
        })
    return airfoil, bays


# ---------------------------------------------------------------------------
# VMT
# ---------------------------------------------------------------------------
def _extract_vmt(snap):
    """One dataset per load case, on OAS's own grid plus the bay-station markers."""
    ws = _a(snap, "stations", "ws_in")
    bay_ws = np.array(snap["grid"]["wc_bay_ws_in"])
    out = []
    for name in (snap["sizing"]["case_name"], snap["sizing"]["cruise_case_name"]):
        ds = {"name": name, "WS": ws.tolist(), "WS_bay": bay_ws.tolist()}
        for comp in ("Vx", "Vy", "Vz", "Mx", "My", "Mz"):
            col = _a(snap, "vmt", name, comp)
            ds[comp] = col.tolist()
            ds[comp + "_bay"] = osnap._resample(ws, col, bay_ws).tolist()
        out.append(ds)
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _coeff_shape(values, chord):
    """Section-coefficient shape, normalized to 1.0 at its peak.

    Same definition as WingCalc's: the running load divided by the local chord,
    scaled so the peak is one. The dynamic pressure is one constant per case and
    cancels; the chord does not, and on a tapered wing that is the whole
    difference between a spanload shape and a coefficient shape.
    """
    c = np.asarray(chord, dtype=float)
    coeff = np.divide(values, c, out=np.zeros_like(values), where=c > 0.0)
    peak = float(np.max(np.abs(coeff)))
    return (coeff / peak) if peak > 0.0 else np.zeros_like(coeff)


def _extract_loading(snap):
    """The applied loads each VMT was integrated from, per load case.

    Aircraft body axes, X aft, Y outboard, Z up, in lbs per inch of span, which
    is WingCalc's contract. Two cases: the 1 g cruise state OAS actually solved,
    and the 2.5 g maneuver it sized against -- which is the same distribution
    scaled, because that is how the deck feeds WingCalc too.

    Both carry the resampled ``*_bay`` companion arrays, so every curve on this
    tab can be read at WingCalc's own 20 stations as well as on OAS's 34-panel
    grid.
    """
    sz = snap["sizing"]
    # AERO RUNS THE FULL SPAN, STRUCTURE STOPS AT THE CUT. The winglet carries
    # about 1.4% of half-wing lift and 3.1% of the root bending moment, and the FEM
    # carries it, so drawing the lift only as far as the structural cut would show
    # the optimizer being handed free lift it is in fact paying for. The structural
    # weight curve keeps the cut because that is the mass this report totals. The
    # cut itself is marked on the plot rather than left to be inferred.
    ws = _a(snap, "aero", "ws_mid_in_uncut")
    width = _a(snap, "aero", "width_in_uncut")
    chord = _a(snap, "aero", "chord_in_uncut")
    fz1 = _a(snap, "aero", "fz_1g_lb_uncut")
    fx1 = _a(snap, "aero", "fx_1g_lb_uncut")
    ws_struct = _a(snap, "aero", "ws_mid_in")
    width_struct = _a(snap, "aero", "width_in")
    mass = _a(snap, "struct", "element_mass_lb")
    bay_ws = np.array(snap["grid"]["wc_bay_ws_in"])
    cut_ws = float(snap["grid"]["cut_ws_in"])

    cases = []
    for name, k_aero, k_inertia in (
        (sz["case_name"], sz["aero_scale"], sz["Nz"]),
        (sz["cruise_case_name"], 1.0, 1.0),
    ):
        lift = k_aero * fz1 / width
        drag = k_aero * fx1 / width
        # Negative: an inertial load at positive Nz acts DOWN, which is what
        # relieves the wing. The fuel curve is identically zero because the deck
        # gives this case Fuel_fraction 0 -- there is no wing fuel to relieve it.
        w_struct = -k_inertia * mass / width_struct
        w_fuel = np.zeros_like(w_struct)

        # Point loads. The two nacelles are real point masses in the model. The
        # root clamp is OAS's analogue of WingCalc's wing-to-fuselage reaction:
        # it is the force the FEM's built-in end reacts, so it is the load that
        # balances all of the above. It sits at the centreline, not at the side
        # of body, which is the single biggest structural difference between the
        # two models and the reason the two root bending moments differ.
        nacs = sz["nacelles"]
        points = [{"label": lab, "ws": n["ws_in"],
                   "fz": round(-k_inertia * n["weight_lb"], 1)}
                  for lab, n in zip(("Nacelle ib", "Nacelle ob"), nacs)]
        w_nac = sum(n["weight_lb"] for n in nacs)
        total = float((lift * width).sum() + (w_struct * width_struct).sum()
                      - k_inertia * w_nac)
        points.append({"label": "Root clamp", "ws": 0.0, "fz": round(-total, 1)})

        cases.append({
            "name": name,
            "WS": np.round(ws, 4).tolist(),
            "WS_struct": np.round(ws_struct, 4).tolist(),
            "cut_ws_in": cut_ws,
            "WS_bay": bay_ws.tolist(),
            "lift": np.round(lift, 4).tolist(),
            "drag": np.round(drag, 4).tolist(),
            "cl": np.round(_coeff_shape(lift, chord), 5).tolist(),
            "cd": np.round(_coeff_shape(drag, chord), 5).tolist(),
            "w_struct": np.round(w_struct, 4).tolist(),
            "w_fuel": w_fuel.tolist(),
            "w_dist": np.round(w_struct + w_fuel, 4).tolist(),
            "points": points,
            "lift_bay": np.round(osnap._resample(ws, lift, bay_ws), 4).tolist(),
            "drag_bay": np.round(osnap._resample(ws, drag, bay_ws), 4).tolist(),
            "cl_bay": np.round(osnap._resample(ws, _coeff_shape(lift, chord), bay_ws), 5).tolist(),
            "cd_bay": np.round(osnap._resample(ws, _coeff_shape(drag, chord), bay_ws), 5).tolist(),
            "w_struct_bay": np.round(osnap._resample(ws_struct, w_struct, bay_ws), 4).tolist(),
        })
    return {"cases": cases}


# ---------------------------------------------------------------------------
# Stress results
# ---------------------------------------------------------------------------
def _ms_record(snap, i, k, name, kind, sense, combo):
    """One margin, as a record the frontend's detail panel can print verbatim.

    ``MS_von_mises`` is the only numeric margin key, which is why it had to be
    added to the frontend's MS_FIELD_ORDER -- a record whose margin key is not
    in that list comes back with no margin at all and its callout silently
    disappears off the section.
    """
    sz = snap["sizing"]
    vm = snap["vonmises_psi"][i][k]
    sig_t, sig_c = sz["sigma_tension_psi"], sz["sigma_compression_psi"]
    allow = sig_c if sense == "compression" else sig_t
    # See the _MS_POINTS comment: OAS reports combinations 0 and 3 already
    # divided by sigma_c / sigma_t, so the physical stress is scaled back up.
    stress = vm * allow / sig_t
    design = allow / sz["safety_factor"]
    ms = design / stress - 1.0 if stress > 0.0 else NO_MS
    return {
        "Point": name,
        "Stress combination": combo,
        "Source": f"vonmises[{i}, {k}]",
        "von Mises stress (psi)": f"{stress:,.1f}",
        "Allowable (psi)": f"{allow:,.1f}  ({sense})",
        "Safety factor": f"{sz['safety_factor']:.2f}",
        "Design allowable (psi)": f"{design:,.1f}",
        "MS_von_mises": round(float(ms), 6),
    }


def _min_ms(records):
    vals = [r["MS_von_mises"] for r in records.values() if r["MS_von_mises"] < 900]
    return min(vals) if vals else NO_MS


def _worst(records, ws, bay_id, lc):
    best = None
    for key, r in records.items():
        if r["MS_von_mises"] >= 900:
            continue
        if best is None or r["MS_von_mises"] < best["ms"]:
            best = {"ms": r["MS_von_mises"], "location": f"Bay {bay_id}, {r['Point']}",
                    "lc": lc, "failure_mode": "von Mises strength", "bay_id": bay_id,
                    "key": key, "ws": ws}
    return best


def _extract_results(snap):
    """Margins of safety at the four von Mises points of every element.

    What is NOT here, and why. OAS's structural functionals are one KS failure
    aggregate over four combined-stress points per element. There is no buckling
    check, no crippling check, no element-buckling check, no inertia check and no
    Tsai-Wu -- the last because the deck's Materials.csv gives smeared laminate
    properties rather than lamina data, so the Tsai-Wu path cannot be fed at all.
    So the stringer and spar-cap rows of the cross-section summary are omitted
    (the frontend skips a null row) rather than filled with a margin from a mode
    that was never evaluated.
    """
    lc = snap["sizing"]["case_name"]
    bays, all_candidates = [], []
    for i, e in enumerate(snap["elements"]):
        bay_id = i + 1
        skins, webs = {}, {}
        for k, name, kind, sense, combo in _MS_POINTS:
            rec = _ms_record(snap, i, k, name, kind, sense, combo)
            (skins if kind == "skin" else webs)[name] = rec

        ms_skin, ms_web = _min_ms(skins), _min_ms(webs)
        overall = min(ms_skin, ms_web)
        worst_skin = _worst(skins, e["ws_in"], bay_id, lc)
        worst_web = _worst(webs, e["ws_in"], bay_id, lc)
        all_candidates += [c for c in (worst_skin, worst_web) if c]

        def pack(c):
            return None if c is None else {"ms": c["ms"], "location": c["location"],
                                           "lc": c["lc"], "failure_mode": c["failure_mode"]}

        summary = {"min_ms": overall, "skin": pack(worst_skin),
                   # No stringers and no spar caps in the model, so no row.
                   "stringer": None, "spar_cap": None, "spar_web": pack(worst_web)}
        payload = {
            "ms": {"overall": overall, "stringers": NO_MS,
                   "skins": ms_skin, "spars": ms_web},
            "stringers": {}, "skins": skins, "spar_caps": {}, "spar_webs": webs,
            "cross_section_summary": summary,
        }
        bays.append({
            "bay_id": bay_id, "ws": e["ws_in"], "bay_index": i,
            "stability_min_ms": NO_MS, "strength_min_ms": overall,
            **payload, "by_loadcase": {lc: payload},
        })

    worst = min(all_candidates, key=lambda c: c["ms"]) if all_candidates else None
    overall_results = ({"min_ms": NO_MS, "bay_id": None, "location": "", "lc": "",
                        "failure_mode": ""} if worst is None else
                       {"min_ms": worst["ms"], "bay_id": worst["bay_id"],
                        "location": worst["location"], "lc": worst["lc"],
                        "failure_mode": worst["failure_mode"]})
    return {
        "generated_at": snap["created_at"].replace("T", " "),
        "load_cases": [lc],
        "bays": bays,
        "overall_results": overall_results,
        "overall_results_by_loadcase": {lc: overall_results},
    }


# ---------------------------------------------------------------------------
# Weight
# ---------------------------------------------------------------------------
# WingCalc's own colours for the two categories OAS has, so a bar means the same
# thing in both reports. The categories it does not have -- stringers, ribs and
# the nacelle / gear reinforcements -- are left out rather than drawn empty.
_W_CATS = (("skin", "Skin", "#1a5f8a"), ("spar", "Spars", "#7eb8da"))

_SPANWISE_NOTE = (
    "Wingbox only: OAS sizes the box and nothing else, and the rest of the calibrated "
    "wing weight has no spanwise distribution behind it"
)


def _extract_weight(snap):
    w = snap["weight"]
    rho = snap["sizing"]["mrho_lb_in3"]
    box_full = 2.0 * w["box_half_lb"]
    stations = []
    for i, e in enumerate(snap["elements"]):
        # Split OAS's own element mass on OAS's own area split, so the two
        # always add back up to element_mass exactly.
        m = snap["struct"]["element_mass_lb"][i]
        a_tot = e["a_skin_in2"] + e["a_spar_in2"]
        f_skin = e["a_skin_in2"] / a_tot if a_tot > 0 else 1.0
        wt = {"skin": m * f_skin, "spar": m * (1.0 - f_skin)}
        stations.append({
            "bay_id": i + 1,
            "ws": e["ws_left_in"],
            "rib_pitch": e["width_in"],
            # These are FEM elements, not bays: OAS's ribs are weighed by the
            # build-up at WingCalc's rib stations (Weight Summary), not here.
            "rib_type_display": "— (FEM element; ribs in the build-up)",
            "w": wt,
            "w_wingbox": m,
            "w_wing": m,
        })

    # The build-up, computed by OAS from its own box and geometry, in WingCalc's own
    # rows, labels and references -- so the two Weight Summary tabs read line for
    # line and a difference is a modelling difference, not a layout one.
    b = w["buildup"]
    t = b["terms"]
    nac_note = ("" if b["web_match_to_cap"] == "input" else
                "; web brought up to its cap NOT computed (OAS has no spar caps)")
    skin = sum(s_["w"]["skin"] for s_ in stations) * 2
    spar = sum(s_["w"]["spar"] for s_ in stations) * 2
    box = [
        {"key": "W_skin_full", "label": "Skin", "value": skin,
         "ref": "OAS wingbox FEM (stringers smeared in)"},
        {"key": "W_stg_full", "label": "Stringers", "value": 0.0,
         "ref": "OAS models none -- carried in the skin"},
        {"key": "W_spar_full", "label": "Spars", "value": spar, "ref": "OAS wingbox FEM"},
        {"key": "W_ribs_full", "label": "Ribs", "value": t["W_ribs_full"],
         "ref": "Smeared thickness, on OAS A_enc"},
        {"key": "W_wingbox_basic", "label": "Wingbox basic", "value": t["W_wingbox_basic"],
         "ref": "Sum", "is_subtotal": True},
        {"key": "W_joints", "label": "Fasteners",
         "value": t["W_joints"] + t["W_winglet_attachment"], "ref": "Weight estimate"},
        {"key": "W_splices", "label": "Splices", "value": t["W_splices"], "ref": "Weight estimate"},
        {"key": "W_fail_safety_damage_tolerance", "label": "Damage tolerance",
         "value": t["W_fail_safety_damage_tolerance"], "ref": ""},
        {"key": "W_ply_drops_clips", "label": "Ply drops and clips",
         "value": t["W_ply_drops_clips"], "ref": "2% of wingbox"},
        {"key": "W_nacelle_reinforcements", "label": "Nacelle reinforcement",
         "value": t["W_nacelle_reinforcements"], "ref": "Weight estimate" + nac_note},
        {"key": "W_landing_gear_reinforcements", "label": "MLG reinforcement",
         "value": t["W_landing_gear_reinforcements"], "ref": "Torenbeek* eq 11.60"},
        {"key": "W_wing_fuselage_attach", "label": "Wing/fuse attach",
         "value": t["W_wing_fuselage_attach"], "ref": "Torenbeek* eq 11.61"},
        {"key": "W_dynamic_over_swing", "label": "Dynamic over-swing",
         "value": t["W_dynamic_over_swing"], "ref": ""},
        {"key": "W_torsional_stiffness", "label": "Torsional stiffness",
         "value": t["W_torsional_stiffness"], "ref": ""},
        {"key": "W_wingbox", "label": "Wingbox total", "value": t["W_wingbox"],
         "ref": "Sum", "is_total": True},
    ]
    secondary = [
        {"key": "W_leading_edge_fixed", "label": "Fixed LE", "value": t["W_leading_edge_fixed"], "ref": "Torenbeek* eq 11.63"},
        {"key": "W_leading_edge_high_lift", "label": "LE high lift", "value": t["W_leading_edge_high_lift"], "ref": ""},
        {"key": "W_fixed_trailing_edge", "label": "Fixed TE", "value": t["W_fixed_trailing_edge"], "ref": "Torenbeek* eq 11.65"},
        {"key": "W_trailing_edge_flaps", "label": "Flaps", "value": t["W_trailing_edge_flaps"], "ref": "Torenbeek* eq 11.66"},
        {"key": "W_aileron", "label": "Aileron", "value": t["W_aileron"], "ref": "Torenbeek* eq 11.67"},
        {"key": "W_mfs", "label": "MFS", "value": t["W_mfs"], "ref": "Torenbeek* eq 11.68"},
        {"key": "W_wingtip", "label": "Wingtip", "value": t["W_wingtip"], "ref": "Torenbeek* eq 11.70"},
        {"key": "W_paint", "label": "Paint", "value": t["W_paint"], "ref": "Weight estimate"},
        {"key": "W_copper_mesh", "label": "Copper mesh", "value": t["W_copper_mesh"], "ref": "Weight estimate"},
        {"key": "W_secondary", "label": "Secondary total", "value": t["W_secondary"], "ref": "Sum", "is_total": True},
    ]
    adjusted = t["k_misc"] * t["W_secondary"]
    totals = [
        {"key": "W_wingbox", "label": "Wingbox total", "value": t["W_wingbox"], "ref": "Sum"},
        {"key": "W_secondary", "label": "Secondary total", "value": t["W_secondary"], "ref": "Sum"},
        {"key": "k_misc", "label": "Misc factor (k_misc)", "value": t["k_misc"],
         "ref": "Torenbeek* sec 11.6.7", "decimals": 3},
        {"key": "W_secondary_adjusted", "label": "Adjusted secondary", "value": adjusted,
         "ref": "k_misc x secondary"},
        {"key": "W_wing", "label": "Total wing weight", "value": t["W_wing"],
         "ref": "Sum", "is_total": True},
    ]
    return {
        "units": {"weight": "lbs", "volume": "in^3", "station": "in",
                  "area": "in^2", "thickness": "in"},
        "spanwise": {
            "categories": [{"key": k, "label": lab, "short": lab,
                            "scope": "wingbox", "color": col} for k, lab, col in _W_CATS],
            "semispan_totals": {"wingbox": w["box_half_lb"], "wing": w["box_half_lb"]},
            "stations": stations,
            "note": _SPANWISE_NOTE,
        },
        "wingbox": box,
        "secondary": secondary,
        "totals": totals,
        "summary": {
            "w_wingbox": t["W_wingbox"],
            "w_secondary": t["W_secondary"],
            "w_misc": adjusted - t["W_secondary"],
            "k_misc": t["k_misc"],
            "w_secondary_adjusted": adjusted,
            "w_wing": t["W_wing"],
        },
        "density_lb_in3": rho,
    }


def _extract_weight_visuals(snap):
    """Where the sized box's weight acts, element by element.

    Four components per element -- the two skins and the two webs -- each at its
    own cg, because that is as far as OAS's mass model resolves: ``element_mass``
    is one number per element and the only defensible split of it is by the areas
    ``section_properties_wingbox`` built it from. There is no per-rib grouping
    for the same reason there are no ribs.
    """
    stations, comps = [], []
    for i, e in enumerate(snap["elements"]):
        m = snap["struct"]["element_mass_lb"][i]
        parts = []
        for label, kind, area, cx, cz in (
            ("Upper skin", "Skin", e["skin_upper"]["length_in"] * e["t_skin_in"],
             e["skin_upper"]["cx_in"], e["skin_upper"]["cz_sec_in"]),
            ("Lower skin", "Skin", e["skin_lower"]["length_in"] * e["t_skin_in"],
             e["skin_lower"]["cx_in"], e["skin_lower"]["cz_sec_in"]),
            ("Fwd spar web", "Spar web", e["spar_h_fwd_in"] * e["t_spar_in"],
             e["chord_in"] * e["fwd_ratio"], 0.5 * (e["z_up_fwd_in"] + e["z_lo_fwd_in"])),
            ("Aft spar web", "Spar web", e["spar_h_aft_in"] * e["t_spar_in"],
             e["chord_in"] * e["aft_ratio"], 0.5 * (e["z_up_aft_in"] + e["z_lo_aft_in"])),
        ):
            parts.append((label, kind, area, cx, cz))
        # Renormalized onto OAS's own element mass, so the four always sum to it:
        # the skin and web areas above are built from the drawn geometry, and the
        # section integration rounds the box corners slightly differently.
        a_sum = sum(p[2] for p in parts) or 1.0
        rows = []
        for label, kind, area, cx, cz in parts:
            wgt = m * area / a_sum
            rows.append({
                "id": i + 1, "from_bay": i + 1, "label": label, "kind": kind,
                "material": MATERIAL, "w_basic": wgt, "factor": 1.0, "w_reinf": 0.0,
                "w": wgt,
                "x": e["x_le_in"] + cx, "y": e["ws_in"],
                "z": e["z_chordline_in"] + cz,
                "xc": cx / e["chord_in"],
            })
        comps += rows
        tw = sum(r["w"] for r in rows) or 1.0
        stations.append({
            "id": i + 1, "ws": e["ws_in"], "w": m,
            "x": sum(r["w"] * r["x"] for r in rows) / tw,
            "y": e["ws_in"],
            "z": sum(r["w"] * r["z"] for r in rows) / tw,
            "xc": sum(r["w"] * r["xc"] for r in rows) / tw,
            "w_wingbox": m,
        })

    wg = snap["winglet"]
    separate = [{
        "label": "Winglet box (excluded from every total)", "kind": "Skin",
        "material": MATERIAL, "w": wg["box_mass_lb"],
        "x": wg["cg_x_in"], "y": wg["cg_ws_in"], "z": wg["cg_z_in"],
        "xc": None,
    }]
    tot_w = sum(s["w"] for s in stations) or 1.0
    total = {
        "w": sum(s["w"] for s in stations),
        "x": sum(s["w"] * s["x"] for s in stations) / tot_w,
        "y": sum(s["w"] * s["y"] for s in stations) / tot_w,
        "z": sum(s["w"] * s["z"] for s in stations) / tot_w,
        "xc": sum(s["w"] * s["xc"] for s in stations) / tot_w,
        "w_wingbox": sum(s["w"] for s in stations),
    }
    envelope = [{
        "ws": round(e["ws_in"], 2),
        "z_top": round(e["z_chordline_in"] + e["z_na_sec_in"]
                       + snap["struct"]["htop_in"][i], 2),
        "z_bot": round(e["z_chordline_in"] + e["z_na_sec_in"]
                       - snap["struct"]["hbottom_in"][i], 2),
        "z_qc": round(e["z_qc_in"], 2),
    } for i, e in enumerate(snap["elements"])]

    return {
        "groupings": {
            "bay": {"stations": stations, "components": comps,
                    "separate": separate, "total": total},
            # No ribs, so no per-rib grouping; the frontend disables the selector.
            "rib": None,
        },
        "envelope": envelope,
        # WingCalc's own wording for this note explains that a bay carries the rib at
        # its inboard station, which would be a false statement about this wing.
        "group_spec_overrides": {
            "bay": {
                "name": "Bay",
                "trace": "Bay cg",
                "hover": "cg of the sized box",
                "note": "Weight grouped into FEM elements: an element is the wingbox "
                        "between two mesh stations, so its cg falls between them. There "
                        "are no ribs in this model -- the four contributions below are "
                        "the two skins and the two spar webs, split out of "
                        "element_mass on the areas section_properties_wingbox built it "
                        "from. The winglet box is drawn as a separate item and is "
                        "excluded from every total.",
            },
        },
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_oas_viewer(snapshot, output_path, created_at=None):
    """Write the OAS wing report.

    ``snapshot`` is an ``oas_snapshot`` dict or a path to one. Nothing is
    computed here: every number on the page is in that dict.
    """
    if isinstance(snapshot, (str, Path)):
        snapshot = json.loads(Path(snapshot).read_text())
    if snapshot.get("snapshot_version") != osnap.SNAPSHOT_VERSION:
        raise ValueError(
            f"snapshot version {snapshot.get('snapshot_version')} does not match this "
            f"viewer's {osnap.SNAPSHOT_VERSION}; rebuild it with "
            "'python -m studies.vsp_planform.viewer.oas_snapshot'"
        )
    output_path = Path(output_path)
    created_at = created_at or _dt.datetime.now()

    airfoil, bays = _extract_xsec(snapshot)
    src = snapshot["source"]
    html = build_html(
        _extract_planform(snapshot), airfoil, bays, _extract_vmt(snapshot),
        loading=_extract_loading(snapshot),
        results=_extract_results(snapshot),
        weight=_extract_weight(snapshot),
        weight_visuals=_extract_weight_visuals(snapshot),
        output_folder_name=f"{src['case_label']}  |  {Path(src['converged_json']).name}",
        created_at_text=_format_created_at(created_at),
        tool_version="OpenAeroStruct / vsp_planform arc_aerostruct",
        report_title=REPORT_TITLE,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _format_created_at(created_at):
    """Same wording as WingCalc's header line, so the two headers line up."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "June",
              "July", "Aug", "Sept", "Oct", "Nov", "Dec"]
    return (f"Created {months[created_at.month - 1]} {created_at.day}, "
            f"{created_at.year} {created_at.hour:02d}h{created_at.minute:02d}")


def main(argv=None):
    logs = Path(__file__).resolve().parent.parent / "out" / "logs"
    ap = argparse.ArgumentParser(
        description="Write the OAS wing report from a converged design point.")
    ap.add_argument("--snapshot", default=None,
                    help="an oas_snapshot JSON to render; skips OpenMDAO entirely")
    ap.add_argument("--json", default=str(logs / "arc_aerostruct_A_optimal_e694.json"),
                    help="converged arc_aerostruct JSON, when no --snapshot is given")
    ap.add_argument("--case", default="0",
                    help="index into that JSON, or a substring of one design point's label")
    ap.add_argument("--seed", default=None,
                    help="arc_optimal_toc JSON the frozen geometry comes from")
    ap.add_argument("--save-snapshot", default=None,
                    help="also write the snapshot, so later runs can use --snapshot")
    ap.add_argument("--out", default=str(logs / "OAS_Wing_Report.html"))
    a = ap.parse_args(argv)

    if a.snapshot:
        snap = json.loads(Path(a.snapshot).read_text())
    else:
        snap = osnap.build_snapshot(a.json, case=a.case, seed_json=a.seed)
        if a.save_snapshot:
            Path(a.save_snapshot).parent.mkdir(parents=True, exist_ok=True)
            Path(a.save_snapshot).write_text(json.dumps(snap, indent=1))
            print(f"wrote {a.save_snapshot}")

    out = generate_oas_viewer(snap, a.out)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"  case       {snap['source']['case_label']}")
    print(f"  box        {2 * snap['weight']['box_half_lb']:.2f} lb both wings")
    print(f"  wing       {snap['weight']['buildup']['terms']['W_wing']:.2f} lb, WingCalc's "
          "build-up on the OAS box")
    print(f"  root My    {snap['vmt'][snap['sizing']['case_name']]['My'][0]:,.0f} in-lb "
          f"at {snap['sizing']['case_name']}")
    return out


if __name__ == "__main__":
    main()
