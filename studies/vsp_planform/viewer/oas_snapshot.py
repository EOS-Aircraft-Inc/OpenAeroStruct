"""Everything the OAS wing report needs, flattened out of one OpenAeroStruct run.

The report is a pure function of plain dicts (see ``wingcalc_frontend``), and
this module is what stands between those dicts and a live OpenMDAO model. It
runs the Arc A aerostructural problem ONCE at a converged design point and
writes down, in the report's own units, every array the page draws. Nothing
here formats anything; nothing in ``oas_report`` touches OpenMDAO.

That seam earns its keep twice over. A snapshot is a few hundred kilobytes of
JSON, so a report can be re-rendered from one in well under a second with no
solver, no aerosandbox and no FEM -- which is the difference between iterating
on a plot and not. And it makes the numbers auditable: the snapshot is the
whole evidence base for the page, in a file a person can read.

FOUR THINGS HERE ARE NOT OBVIOUS
--------------------------------

1. THE STRUCTURAL ARRAYS COME BACK MIRRORED. ``arc_aerostruct.MirrorSpanwise``
   reflects this study's ROOT -> TIP, y >= 0 mesh into OAS's TIP -> ROOT, y <= 0
   convention, because ``structures/fem.py`` clamps spanwise node ``ny - 1``
   under symmetry. Everything downstream of that mirror -- ``nodes``, ``loads``,
   ``A``, ``Iy``, ``element_mass``, ``vonmises``, the lot -- is therefore in
   TIP -> ROOT order. Every array is flipped back here, once, at the boundary,
   so that nothing further out has to remember. The mirror is exact: it is a
   reflection, so the structure is identical and only the ordering changes.

2. VMT IS INTEGRATED IN THE *MIRRORED* FRAME, ON PURPOSE. WingCalc's VMT sign
   convention (``loading/wing_loading.py``, lines 38-50) is stated in a local
   wing system -- x' outboard along the elastic axis, y' aft, z' up -- and that
   triad is right-handed only on the LEFT wing, which is the wing WingCalc
   models. OAS's mirrored structural mesh IS the left wing. So the integration
   is done there, in the frame it was written for, and only the station label is
   flipped to a positive WS at the end. Doing it on the right wing and negating
   signs afterwards is the same arithmetic with three more chances to get a sign
   backwards.

3. THE WINGLET IS CUT OFF AT THE B|C JUNCTION. WingCalc's wingbox span is
   1356 in (678 per side) and its weights say "Excludes : Flaps, ailerons and
   winglets", so the OAS structure is cut to match. The mesh node nearest 678 in
   is the B|C junction itself at 674.95 in -- the geometric winglet root -- so
   the cut lands on a real design feature rather than mid-element. What is left
   out is reported rather than dropped: it is about 5.8 lb of the 1,685 lb
   half-wing box, and it appears in the Weight Visuals tab as a separate item
   and in the breakdown table on its own line.

4. THE SIZING CASE IS ONE LOAD CASE, AND ITS SPANLOAD IS A SCALED CRUISE ONE.
   The deck exports the 1 g MTOW spanload and WingCalc scales it by Nz; this
   does the same, so the two models are fed the same thing. The three load
   components (aero, structural inertia, nacelle point masses) are kept apart in
   the snapshot precisely so the 1 g case can be reassembled exactly -- it is
   the aero divided by its own scale factor, not a second solve.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Locating the study's own scripts
# ---------------------------------------------------------------------------
# ``scripts/arc_aerostruct.py`` imports its siblings by bare name
# (``import arc_optimal_toc as aot``), so the scripts directory has to be on
# sys.path before it can be imported at all. It is put LAST so it cannot shadow
# a top-level package, and the repo root is put first so ``studies.*`` resolves
# the same way it does under pytest.
_HERE = Path(__file__).resolve()
_STUDY = _HERE.parent.parent
_ROOT = _STUDY.parent.parent
_SCRIPTS = _STUDY / "scripts"
for _p in (str(_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_SCRIPTS) not in sys.path:
    sys.path.append(str(_SCRIPTS))

from studies.vsp_planform import config          # noqa: E402
from studies.vsp_planform.param import rear_spar_fraction  # noqa: E402

# ---------------------------------------------------------------------------
# Units. The report is in inches, pounds and psi throughout, because that is
# what the deck, the CSVs and WingCalc's report are in; OAS is in SI. Every
# conversion in this file goes through one of these, so there is one place to
# check rather than a scattering of 0.0254s.
# ---------------------------------------------------------------------------
IN_M = 0.0254
LB_N = 4.4482216
LB_KG = 0.45359237
PSI_PA = 6894.757293
INLB_NM = LB_N * IN_M           # 1 N*m = 1 / INLB_NM lb*in
G_MS2 = 9.80665

# WingCalc's 20 bay stations, read off
# ``out/logs/wc_arcA_optimal_e694/04.Weights/wingWeightSummary.csv`` (the WS
# column). Every OAS distribution in the report is ALSO resampled onto these, so
# the two reports can be read against each other station by station instead of
# off two different grids. They are hard-coded rather than parsed because a
# report has to be writable with no WingCalc output folder present, and because
# a silently different set would make the comparison wrong rather than absent.
WC_BAY_WS_IN = (
    0.0, 26.5, 53.0, 90.5, 128.0, 165.5, 187.5, 226.125, 264.75, 303.375,
    342.0, 370.0, 408.5, 447.0, 485.5, 524.0, 562.5, 601.0, 639.5, 678.0,
)

# The structural span the report covers, per WingCalc's "Total wingbox span"
# of 1356 in. See note 3 in the module docstring.
WINGBOX_TIP_WS_IN = 678.0

# Bumped whenever the snapshot's shape changes, so a stale cached file is
# rejected with a message instead of half-rendering.
SNAPSHOT_VERSION = 2

# The one sizing load case, named exactly as the deck's ``loadCasesIn.csv``
# names it. The name is what ties this report's VMT, Loading and Stress Results
# tabs to the same rows of the WingCalc report -- ``orderedVmtIndices`` in the
# frontend matches load case names on alphanumerics only, so a spelling
# difference silently unlinks the tabs.
SIZING_CASE = "2.5g Upbend1 flight"
CRUISE_CASE = "1g Flight"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _f(a) -> list:
    """A numpy array as a plain list of floats, JSON-safe.

    Non-finite values become ``None``: JSON has no NaN, and a bare ``NaN``
    token in the embedded payload would make the blob unparseable as JSON even
    though a browser would read it -- which would cost the report its one cheap
    self-check. The frontend prints ``None`` as an em dash.
    """
    arr = np.asarray(a, dtype=float).ravel()
    return [float(v) if np.isfinite(v) else None for v in arr]


def _resample(x_src, y_src, x_dst):
    """Linear resampling onto the WingCalc bay stations, clamped at both ends.

    ``np.interp``'s clamping is deliberate and matches the frontend's own
    ``interpAtWs``: WingCalc's outermost station (678 in) sits just outboard of
    OAS's last structural node (674.95 in, the B|C junction), so without
    clamping the last marker would be an extrapolation. Clamped, it is the tip
    element's value, which is what it is.
    """
    x_src = np.asarray(x_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    return np.interp(np.asarray(x_dst, dtype=float), x_src, y_src)


def _unit(v):
    n = float(np.linalg.norm(v))
    return np.asarray(v, dtype=float) / n if n > 0.0 else np.asarray(v, dtype=float)


# ---------------------------------------------------------------------------
# Airfoil sections
# ---------------------------------------------------------------------------
def _airfoil_curves(name, n=121):
    """Upper and lower surfaces of a named section on a common x/c grid.

    Cosine-clustered in x/c so the nose is resolved -- the cross-section plot
    draws the whole airfoil, not just the box, and a uniform grid puts visible
    corners on the leading edge.
    """
    import aerosandbox as asb

    af = asb.Airfoil(name)
    beta = np.linspace(0.0, np.pi, n)
    xs = 0.5 * (1.0 - np.cos(beta))
    xs = np.clip(xs, 1e-4, 1.0 - 1e-4)
    t = np.array([float(af.local_thickness(x_over_c=float(x))) for x in xs])
    c = np.array([float(af.local_camber(x_over_c=float(x))) for x in xs])
    return xs, c + 0.5 * t, c - 0.5 * t


def _design_blend(J, aot):
    """(inboard, outboard, f_start, f_end) of the design's section loft.

    From the design's own JSON, not Arc A's table. An arc with one section (Arc B)
    comes back as that section twice with the loft placed past the tip, so every
    blend weight is zero and nothing downstream needs a special case.
    """
    sb = J.get("section_blend")
    if isinstance(sb, dict):
        return (sb["inboard"], sb["outboard"], float(sb["f_start"]), float(sb["f_end"]))
    af = J.get("airfoil") or "e694"
    af = "e694" if af == "as-built" else af
    return (af, af, 2.0, 3.0)


def _blend_weight(blend, y_in, semi_in):
    """How far Arc A's inboard-to-outboard section loft has got at a station.

    Reported, but NOT applied to the drawn section, and that is the point. Arc A
    lofts from e694 to goe16k between 74% and 90% of semi-span, but
    ``arc_aerostruct.wingbox_section`` hands the FEM the INBOARD section's box
    coordinates at every station -- its own comment says so, on the grounds that
    the box mass is overwhelmingly inboard of the blend. So every cross-section
    this report draws is the inboard section scaled to the local t/c, because
    that is the section OAS's structure was actually built on, and drawing the
    lofted one would put a shape on the page that no number came from.

    The weight is carried through to the Planform tab so the divergence is
    visible rather than buried: outboard of 74% semi-span the drawn (and sized)
    section is not the design's own.
    """
    _name_in, _name_out, f0, f1 = blend
    return float(np.clip((abs(float(y_in)) / semi_in - f0) / (f1 - f0), 0.0, 1.0))


# ---------------------------------------------------------------------------
# VMT
# ---------------------------------------------------------------------------
def _local_axes(nodes_m):
    """The local wing triad at each node: x' outboard along the EA, y' aft, z' up.

    Built in the MIRRORED (left-wing) frame, where ``nodes_m`` runs TIP -> ROOT
    with y <= 0, so "outboard" is the direction from node i+1 to node i. That is
    WingCalc's own system and it is right-handed there -- see note 2 in the
    module docstring.

    A station's triad is taken from the element just OUTBOARD of it, which is
    the material the shear and moment at that cut are carried into.
    """
    n = nodes_m.shape[0]
    ex = np.zeros((n, 3))
    # Element i spans nodes i -> i+1, i.e. outboard -> inboard in this frame, so
    # the outboard direction at node i+1 is nodes[i] - nodes[i+1].
    for i in range(n - 1):
        ex[i + 1] = _unit(nodes_m[i] - nodes_m[i + 1])
    ex[0] = ex[1]                                   # the tip node reuses its element
    aft = np.array([1.0, 0.0, 0.0])
    ez = np.array([_unit(np.cross(e, aft)) for e in ex])
    ey = np.array([_unit(np.cross(z, e)) for z, e in zip(ez, ex)])
    return ex, ey, ez


def _vmt(nodes_m, loads_n, ex, ey, ez):
    """Shear and moment at every node, in WingCalc's six components.

    At a cut, the resultant of everything OUTBOARD of it, taken about the node
    (which is on the elastic axis), projected onto the local triad:

        Vx = R . x'   axial, positive outboard
        Vy = R . y'   chordwise shear, positive aft
        Vz = R . z'   vertical shear, positive up
        Mx = M . x'   torque, positive leading edge down
        My = M . y'   bending, NEGATIVE for a wing bending up
        Mz = M . z'   chordwise bending

    Those are the signs stated in ``loading/wing_loading.py``. They fall out of
    a straight right-hand-rule moment of the outboard loads with no negation
    anywhere, because the local frame puts SPAN on x': an up load a distance
    ds outboard gives ``(ds x' ) x (L z') = -ds*L y'``, which is the negative
    My. That is worth spelling out because the natural global-frame intuition
    -- span on y -- gets the sign backwards.

    ``loads_n`` carries a moment at each node as well as a force: the load
    transfer puts the panel forces on the elastic axis and hands the offset
    moment over separately, so dropping it would lose most of the torque.
    """
    n = nodes_m.shape[0]
    out = np.zeros((n, 6))
    # Cumulative from the tip inward. In this frame node 0 IS the tip, so the
    # running sums go forward through the array.
    for i in range(n):
        f_out = loads_n[: i + 1, :3]
        m_out = loads_n[: i + 1, 3:]
        r = nodes_m[: i + 1, :] - nodes_m[i, :]
        R = f_out.sum(axis=0)
        M = m_out.sum(axis=0) + np.cross(r, f_out).sum(axis=0)
        out[i, 0] = R @ ex[i]
        out[i, 1] = R @ ey[i]
        out[i, 2] = R @ ez[i]
        out[i, 3] = M @ ex[i]
        out[i, 4] = M @ ey[i]
        out[i, 5] = M @ ez[i]
    return out


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def _pick_case(cases, case):
    """One entry out of ``arc_aerostruct``'s four-point JSON.

    ``case`` is an index or a substring of the label. A substring that matches
    more than one point is an error rather than a silent first-match: the four
    points differ only in their objective and twist floor, so picking the wrong
    one produces a report that looks right and is about a different design.
    """
    if isinstance(case, int) or (isinstance(case, str) and case.lstrip("-").isdigit()):
        return int(case), cases[int(case)]
    hits = [i for i, c in enumerate(cases) if case.lower() in c["label"].lower()]
    if len(hits) != 1:
        raise SystemExit(
            f"--case {case!r} matched {len(hits)} of the {len(cases)} design points in "
            "the converged JSON. Use an index, or a substring that matches one label:\n  "
            + "\n  ".join(f"{i}: {c['label']}" for i, c in enumerate(cases))
        )
    return hits[0], cases[hits[0]]


def _design_digest(case, seed_json):
    """What the sized box actually depends on, as a cache key.

    Keying the thickness cache on the FILE NAME alone is wrong, and wrong in the
    quiet way: ``pair.write_reports`` hands the snapshot a shim that is always
    called ``oas_report_case.json`` and always holds one point, so the old
    ``name#index`` key was the same string on every run. A re-run whose design had
    moved would then load the previous run's box and draw new aero over old
    structure -- a report that looks right and is about two different wings. The
    cache is an optimization, and an optimization that can return the wrong answer
    is a bug.

    The box is sized against the spanload (twist and alpha) on the frozen geometry
    (taper, t/c, spar fraction) the seed carries, so those are what the digest
    covers. The seed file is read whole rather than picking fields out of it: it is
    a few kB, and a field added to it later would otherwise silently fall outside
    the key.
    """
    h = hashlib.sha256()
    h.update(json.dumps(
        {k: case.get(k) for k in ("twist_cp_deg", "alpha_deg", "twist_lower_deg")},
        sort_keys=True).encode())
    h.update(Path(seed_json).read_bytes())
    return h.hexdigest()[:16]


def _buildup_block(r):
    """The build-up's result, JSON-clean: scalar terms, then the detail the page shows."""
    d = r["detail"]
    areas = {k: (float(v) if isinstance(v, (int, float, np.floating)) else
                 (None if v is None else [float(x) for x in v]))
             for k, v in d["reference_areas"].items()}
    nac = [{"nacelle": n["nacelle"], "rib_weight": float(n["rib_weight"]),
            "pad_weight": float(n["pad_weight"]), "other": float(n["other"]),
            "total": float(n["total"]),
            "ribs": [{k: (float(v) if not isinstance(v, str) else v) for k, v in x.items()}
                     for x in n["ribs"]],
            "pads": [{k: (float(v) if not isinstance(v, str) else v) for k, v in p.items()}
                     for p in n["pads"]]}
           for n in d["nacelle_reinforcement"]]
    return {
        "terms": {k: float(v) for k, v in r.items() if k != "detail"},
        "reference_areas": areas,
        "structural_span_m": float(d["structural_span_m"]),
        "painted_area_ft2": float(d["painted_area_ft2"]),
        "nacelle_reinforcement": nac,
        "web_match_to_cap": d["web_match_to_cap"],
        "ribs": d["ribs"],
    }


def _size_box(aa, seed_json, case, config_mod):
    """Re-run stage 2 -- minimum box mass against the frozen aero state.

    Only reached when the converged JSON predates ``skin_cp_m`` / ``spar_cp_m``
    being written into it; a JSON from a current ``arc_aerostruct`` run carries
    the sized box and this is skipped. ``run_case`` sizes the box in a second
    SLSQP stage with the aero frozen, because with drag as the objective the
    thickness variables do not appear in it at all and SLSQP leaves the box
    wherever feasibility put it. That stage is reproduced verbatim here rather
    than approximated: anything else would report a box this study never sized.

    It is also the right box for a RANGE-objective design point, which never ran
    a stage 2. With the aero frozen, drag is a constant and
    ``R = k (m_fixed - W_wing) / D``, where the build-up's ``W_wing`` rises with the
    box at a fixed rate (``weight.buildup.dW_wing_dW_box``) bar the rib cell area,
    whose effect is two orders smaller -- so maximizing range and minimizing box
    mass pick, to within that, the same box.
    """
    seed = {"twist_cp_deg": case["twist_cp_deg"], "alpha_deg": case["alpha_deg"]}
    prob, _mesh, _J = aa._solve(
        seed_json, case["twist_lower_deg"], "mass",
        aero_dv=False, thick_dv=True, seed=seed,
    )
    return (np.asarray(prob.get_val("skin_thickness_cp", units="m")).ravel().tolist(),
            np.asarray(prob.get_val("spar_thickness_cp", units="m")).ravel().tolist())


def build_snapshot(json_path, case=0, seed_json=None, thickness_cache=None):
    """Run the OAS model at a converged design point and flatten it.

    Parameters
    ----------
    json_path : the ``arc_aerostruct_*.json`` written by
        ``studies/vsp_planform/scripts/arc_aerostruct.py``, holding four
        converged design points.
    case : index into that file, or a substring of one point's label.
    seed_json : the ``arc_optimal_toc_*.json`` the frozen geometry comes from.
        Defaults to the Arc A optimal point beside ``json_path``.
    thickness_cache : where to keep the sized skin/spar control points when the
        converged JSON does not carry them, so the (few-minute) sizing stage is
        paid once rather than on every report.
    """
    import arc_aerostruct as aa
    import arc_optimal_toc as aot

    json_path = Path(json_path)
    cases = json.loads(json_path.read_text())
    idx, cs = _pick_case(cases, case)
    seed_json = Path(seed_json) if seed_json else json_path.parent / "arc_optimal_toc_A_optimal_e694.json"

    # --- the sized box. Written into the converged JSON by newer runs; sized
    #     here (once, cached) for a JSON old enough not to carry it.
    skin_cp = cs.get("skin_cp_m")
    spar_cp = cs.get("spar_cp_m")
    if skin_cp is None or spar_cp is None:
        cache = Path(thickness_cache) if thickness_cache else json_path.with_suffix(".thickness.json")
        key = f"{json_path.name}#{idx}@{_design_digest(cs, seed_json)}"
        held = json.loads(cache.read_text()) if cache.is_file() else {}
        if key in held:
            skin_cp, spar_cp = held[key]["skin_cp_m"], held[key]["spar_cp_m"]
        else:
            print(f"  sizing the box for '{cs['label']}' (the converged JSON predates "
                  "skin_cp_m / spar_cp_m); this runs one SLSQP stage", flush=True)
            skin_cp, spar_cp = _size_box(aa, str(seed_json), cs, config)
            held[key] = {"skin_cp_m": skin_cp, "spar_cp_m": spar_cp,
                         "label": cs["label"], "from": str(json_path)}
            cache.write_text(json.dumps(held, indent=2))
            print(f"  cached to {cache}", flush=True)

    # --- one analysis, no driver. ``build`` with neither half of the design
    #     space free adds no design variables and no constraints, which is all
    #     that separates an analysis from the optimization runs.
    prob, mesh, J, surf_s = aa.build(
        str(seed_json), (cs["twist_lower_deg"], config.TWIST_BOUNDS[1]),
        aero_dv=False, thick_dv=False,
    )
    aa.apply_frozen(prob, J, {"twist_cp_deg": cs["twist_cp_deg"],
                              "alpha_deg": cs["alpha_deg"],
                              "skin_cp_m": skin_cp, "spar_cp_m": spar_cp})
    prob.run_model()
    # The nacelle stations are interpolated onto the deflected elastic axis, so
    # they need `nodes`, which only exists after the first run_model.
    aa.place_nacelles(prob)
    prob.run_model()

    return _flatten(prob, mesh, J, cs, aa, aot,
                    source_json=str(json_path), case_index=idx)


def _flatten(prob, mesh, J, cs, aa, aot, source_json, case_index):
    """Everything the report draws, in report units, root -> tip, winglet cut."""
    def gv(name, **kw):
        return np.asarray(prob.get_val(name, **kw))

    # ---------------- geometry, in the study's own ROOT -> TIP frame ---------
    # mesh is (nx, ny, 3) in metres: row 0 is the leading edge, row -1 the
    # trailing edge, and y >= 0 on the right wing.
    x_le_in = mesh[0, :, 0] / IN_M
    x_te_in = mesh[-1, :, 0] / IN_M
    y_in = np.abs(mesh[0, :, 1]) / IN_M
    chord_in = x_te_in - x_le_in
    ny = mesh.shape[1]

    # --- the cut. The LAST node at or inboard of WS 678, not the nearest one:
    #     the nearest is 679.67 in, which is one element INTO the winglet, and
    #     keeping it would put winglet structure inside a total that says it
    #     excludes winglets. Inboard of it sits 674.95 in -- the B|C junction,
    #     the geometric winglet root -- so the cut lands on a real design
    #     feature and 3.05 in short of WingCalc's 678, which is the whole
    #     structural-span difference between the two models. See note 3.
    i_cut = int(np.max(np.nonzero(y_in <= WINGBOX_TIP_WS_IN + 1e-9)[0]))
    n_node = i_cut + 1          # nodes kept: 0 .. i_cut
    n_elem = i_cut              # elements kept: 0 .. i_cut-1

    y_mid_in = 0.5 * (y_in[:-1] + y_in[1:])
    toc = gv("wing.t_over_c").ravel()                       # per panel, root -> tip
    twist_abs = gv("twist_abs", units="deg").ravel()

    # Camber-surface z, and the chord line under it. OAS's mesh is a CAMBER
    # surface, so mesh z at a chord fraction is the camber line and not the
    # chord line; the section properties are quoted against the chord line. The
    # offset between the two is removed per station below so that a global Z on
    # this page means the same thing as a global Z on WingCalc's.
    x_mesh = mesh[:, :, 0] / IN_M
    z_mesh = mesh[:, :, 2] / IN_M

    def _z_at(frac):
        """Mesh (camber) z at a chord fraction, per station."""
        out = np.zeros(ny)
        for k in range(ny):
            out[k] = np.interp(x_le_in[k] + frac * chord_in[k], x_mesh[:, k], z_mesh[:, k])
        return out

    z_qc_in = _z_at(0.25)

    # --- spars. Arc A's box is 0.120c to a straight 0.750c aft spar; both come
    #     from config, which arc_aerostruct sets from the arc's own schedule, so
    #     a future kinking spar arrives here without a change.
    fwd_ratio = np.full(ny, float(config.WINGBOX_FRONT_PCT))
    sched = config.WINGBOX_REAR_SCHEDULE
    aft_ratio = np.array([float(rear_spar_fraction(v, sched)) for v in y_in])

    # ---------------- structural arrays: un-mirror once, here ---------------
    # Everything below the MirrorSpanwise components is TIP -> ROOT. See note 1.
    def elem(name, scale=1.0, **kw):
        return gv(name, **kw).ravel()[::-1] * scale

    struct = {
        "A_in2": elem("A", 1.0 / IN_M**2, units="m**2"),
        "Iy_in4": elem("Iy", 1.0 / IN_M**4, units="m**4"),
        "Iz_in4": elem("Iz", 1.0 / IN_M**4, units="m**4"),
        "J_in4": elem("J", 1.0 / IN_M**4, units="m**4"),
        "Qz_in3": elem("Qz", 1.0 / IN_M**3, units="m**3"),
        "A_enc_in2": elem("A_enc", 1.0 / IN_M**2, units="m**2"),
        "A_int_in2": elem("A_int", 1.0 / IN_M**2, units="m**2"),
        "htop_in": elem("htop", 1.0 / IN_M, units="m"),
        "hbottom_in": elem("hbottom", 1.0 / IN_M, units="m"),
        "hfront_in": elem("hfront", 1.0 / IN_M, units="m"),
        "hrear_in": elem("hrear", 1.0 / IN_M, units="m"),
        "skin_t_in": elem("skin_thickness", 1.0 / IN_M, units="m"),
        "spar_t_in": elem("spar_thickness", 1.0 / IN_M, units="m"),
        "element_mass_lb": elem("element_mass", 1.0 / LB_KG, units="kg"),
    }
    vonmises_psi = gv("vonmises", units="N/m**2")[::-1, :] / PSI_PA

    nodes_mirror = gv("nodes", units="m")                   # TIP -> ROOT, y <= 0
    x_ea_in = (nodes_mirror[::-1, 0]) / IN_M                # ROOT -> TIP
    z_ea_in = (nodes_mirror[::-1, 2]) / IN_M

    # ---------------- loads, kept as three separable components -------------
    # The 1 g case is the 2.5 g one with each component divided by its own
    # factor -- exactly what the deck export does -- so keeping them apart is
    # what lets the Loading tab show both without a second solve. See note 4.
    aero_nodal = gv("loads", units="N")                     # already x maneuver scale
    swl = gv("struct_states.struct_weight_loads", units="N")
    pml = gv("struct_states.loads_from_point_masses", units="N")
    # The maneuver spanload is the cruise one scaled by Nz * W_man / W_cruise --
    # arc_aerostruct._solve's own definition, reused verbatim so the 1 g case this
    # divides back out is the one the deck exported.
    aero_scale = aa.NZ * aa.W_MAN_LB / (aa.w2.W / LB_N)

    sec = gv(aa.POINT + ".aero_states.wing_sec_forces", units="N")   # (nx-1, ny-1, 3)
    panel_fz_lb = sec[:, :, 2].sum(axis=0) / LB_N           # per panel, 1 g, root -> tip
    panel_fx_lb = sec[:, :, 0].sum(axis=0) / LB_N
    panel_width_in = np.abs(np.diff(y_in))

    # ---------------- VMT, in the mirrored frame ----------------------------
    ex, ey, ez = _local_axes(nodes_mirror)
    vmt_cases = {}
    for name, (k_aero, k_inertia) in (
        (SIZING_CASE, (1.0, 1.0)),
        (CRUISE_CASE, (1.0 / aero_scale, 1.0 / aa.NZ)),
    ):
        total = k_aero * aero_nodal + k_inertia * (swl + pml)
        v = _vmt(nodes_mirror, total, ex, ey, ez)
        v[:, :3] /= LB_N
        v[:, 3:] /= INLB_NM
        v = v[::-1]                                          # ROOT -> TIP
        vmt_cases[name] = {c: _f(v[:n_node, i])
                           for i, c in enumerate(("Vx", "Vy", "Vz", "Mx", "My", "Mz"))}

    # ---------------- per-element section geometry --------------------------
    blend = _design_blend(J, aot)
    semi_in = float(y_in[-1])
    # One call, not one per element: the section is the same at every station
    # (see _blend_weight) and each aerosandbox evaluation is 121 thickness
    # lookups.
    xs, up, lo = _airfoil_curves(blend[0])
    ref_tc = float(np.max(up - lo))
    elems = []
    for i in range(n_elem):
        ym = float(y_mid_in[i])
        w_blend = _blend_weight(blend, ym, semi_in)
        c = float(0.5 * (chord_in[i] + chord_in[i + 1]))
        tc = float(toc[i])
        tc_scale = tc / ref_tc
        fr = float(0.5 * (fwd_ratio[i] + fwd_ratio[i + 1]))
        ar = float(0.5 * (aft_ratio[i] + aft_ratio[i + 1]))

        # Box extremes in SECTION coordinates (x from the LE, z from the chord
        # line), which is the frame OAS's htop / hfront are measured in.
        in_box = (xs >= fr) & (xs <= ar)
        z_up = up * c * tc_scale
        z_lo = lo * c * tc_scale
        z_top_box = float(np.max(z_up[in_box])) if in_box.any() else float(np.max(z_up))
        z_bot_box = float(np.min(z_lo[in_box])) if in_box.any() else float(np.min(z_lo))

        # The neutral axis, from OAS's own h-distances. htop is measured from
        # the neutral axis to the extreme top fibre and hfront from the front
        # spar face, so both invert directly.
        z_na_sec = z_top_box - struct["htop_in"][i]
        y_na_sec = c * fr + struct["hfront_in"][i]

        # The chord line's global z at this station: the camber surface less the
        # section's own camber, averaged across the box so a single bad
        # interpolation at one chord station cannot move it.
        f_box = np.linspace(fr, ar, 9)
        cam_sec = np.interp(f_box, xs, 0.5 * (z_up + z_lo))
        cam_glob = np.array([
            np.interp(0.5 * (x_le_in[i] + x_le_in[i + 1]) + f * c,
                      0.5 * (x_mesh[:, i] + x_mesh[:, i + 1]),
                      0.5 * (z_mesh[:, i] + z_mesh[:, i + 1]))
            for f in f_box
        ])
        z_chordline = float(np.mean(cam_glob - cam_sec))

        # Skin and spar areas, split the way section_properties_wingbox builds
        # them: each web is its full local depth times the spar thickness, and
        # the skins are what is left of the total area. That is a split of OAS's
        # OWN area, not a re-derivation of it, so the two always add up.
        t_sk = struct["skin_t_in"][i]
        t_sp = struct["spar_t_in"][i]
        # z_up / z_lo are already in inches -- scaled by chord and by the t/c
        # ratio -- so the spar depths are a plain difference. (They were scaled
        # a second time here at first, which put a 2,946 in spar in the report
        # and is exactly the kind of error a side-by-side against WingCalc is
        # meant to catch.)
        h_fwd = float(np.interp(fr, xs, z_up - z_lo))
        h_aft = float(np.interp(ar, xs, z_up - z_lo))
        a_spar = (h_fwd + h_aft) * t_sp
        a_skin = max(struct["A_in2"][i] - a_spar, 0.0)

        # Developed skin length and its chordwise centroid, for the panel the
        # cross-section draws and for the Weight Visuals cg.
        def _surface(z):
            """Developed length, its centroid, and the mid-panel radius of curvature.

            ``R`` is what a buckling method would call the panel's chordwise
            radius; nothing in OAS uses it -- there is no buckling check -- but
            the cross-section's own hover reads it off the panel, so it is
            measured from the geometry rather than left blank. Taken at the
            middle of the box, where the surface is flattest and the number is
            representative of the panel as a whole.
            """
            xb = xs[in_box] * c
            zb = z[in_box]
            ds = np.hypot(np.diff(xb), np.diff(zb))
            mid_x = 0.5 * (xb[:-1] + xb[1:])
            mid_z = 0.5 * (zb[:-1] + zb[1:])
            L = float(ds.sum())
            d1 = np.gradient(zb, xb)
            d2 = np.gradient(d1, xb)
            k = np.abs(d2) / (1.0 + d1**2) ** 1.5
            k_mid = float(k[len(k) // 2])
            R = 1.0 / k_mid if k_mid > 1e-12 else float("inf")
            return (L, float((mid_x * ds).sum() / L), float((mid_z * ds).sum() / L),
                    R if np.isfinite(R) else None)

        len_up, cx_up, cz_up, r_up = _surface(z_up)
        len_lo, cx_lo, cz_lo, r_lo = _surface(z_lo)

        elems.append({
            "index": i,
            "ws_in": ym,
            "ws_left_in": float(y_in[i]),
            "ws_right_in": float(y_in[i + 1]),
            "width_in": float(panel_width_in[i]),
            "chord_in": c,
            "t_c": tc,
            "blend_weight": w_blend,
            "fwd_ratio": fr,
            "aft_ratio": ar,
            "box_chord_in": c * (ar - fr),
            "spar_h_fwd_in": h_fwd,
            "spar_h_aft_in": h_aft,
            "airfoil": {"x": _f(xs), "upper": _f(up), "lower": _f(lo), "ref_tc": ref_tc},
            "z_chordline_in": z_chordline,
            "z_na_sec_in": z_na_sec,
            "y_na_sec_in": y_na_sec,
            "z_top_box_sec_in": z_top_box,
            "z_bot_box_sec_in": z_bot_box,
            "x_le_in": float(0.5 * (x_le_in[i] + x_le_in[i + 1])),
            "x_ea_in": float(0.5 * (x_ea_in[i] + x_ea_in[i + 1])),
            "a_skin_in2": a_skin,
            "a_spar_in2": a_spar,
            "skin_upper": {"length_in": len_up, "cx_in": cx_up, "cz_sec_in": cz_up,
                           "r_chord_in": r_up},
            "skin_lower": {"length_in": len_lo, "cx_in": cx_lo, "cz_sec_in": cz_lo,
                           "r_chord_in": r_lo},
            "z_up_fwd_in": float(np.interp(fr, xs, z_up)),
            "z_lo_fwd_in": float(np.interp(fr, xs, z_lo)),
            "z_up_aft_in": float(np.interp(ar, xs, z_up)),
            "z_lo_aft_in": float(np.interp(ar, xs, z_lo)),
            "z_qc_in": float(np.interp(ym, y_in, z_qc_in)),
            "t_skin_in": float(t_sk),
            "t_spar_in": float(t_sp),
            "twist_deg": float(0.5 * (twist_abs[i] + twist_abs[i + 1])),
        })

    # ---------------- winglet, drawn but not counted ------------------------
    # Drawn on the planform and on the Weight Visuals span, so the wing reads as
    # the wing; counted nowhere. Its own mass is carried as a single number and
    # a single cg so the report can say what the cut left out instead of just
    # leaving it out -- see note 3.
    w_mass = struct["element_mass_lb"][i_cut:]
    w_ws = y_mid_in[i_cut:]
    w_tot = float(w_mass.sum())
    winglet = {
        "ws_in": _f(y_in[i_cut:]),
        "x_le_in": _f(x_le_in[i_cut:]),
        "x_te_in": _f(x_te_in[i_cut:]),
        "chord_in": _f(chord_in[i_cut:]),
        "box_mass_lb": w_tot,
        "n_elements": int(ny - 1 - i_cut),
        "cg_ws_in": float((w_mass * w_ws).sum() / w_tot) if w_tot > 0 else float(y_in[i_cut]),
        "cg_x_in": float(np.interp(
            (w_mass * w_ws).sum() / w_tot if w_tot > 0 else y_in[i_cut],
            y_in, x_le_in + 0.5 * (fwd_ratio + aft_ratio) * chord_in)),
        "cg_z_in": float(np.interp(
            (w_mass * w_ws).sum() / w_tot if w_tot > 0 else y_in[i_cut], y_in, z_ea_in)),
    }

    # ---------------- the payload ------------------------------------------
    def _cut_e(a):
        return _f(np.asarray(a)[:n_elem])

    snap = {
        "snapshot_version": SNAPSHOT_VERSION,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "source": {
            "converged_json": source_json,
            "case_index": case_index,
            "case_label": cs["label"],
            "objective": cs.get("objective"),
            "twist_floor_deg": cs.get("twist_lower_deg"),
            "converged": bool(cs.get("success")),
            "study": "studies/vsp_planform/scripts/arc_aerostruct.py",
        },
        "design": {
            "alpha_deg": float(gv("alpha", units="deg")[0]),
            "twist_cp_deg": _f(gv("wing.twist_cp", units="deg")),
            "twist_abs_deg": _f(twist_abs),
            "t_over_c_cp": _f(J["t_over_c_cp"]),
            "taper_B": float(J["taper_B"]),
            "arc": J.get("arc", "A"),
            "wingbox_pct": float(J["wingbox_pct"]),
            "airfoil_inboard": blend[0],
            "airfoil_outboard": blend[1],
            "blend_start_frac": float(blend[2]),
            "blend_end_frac": float(blend[3]),
            "skin_cp_m": _f(gv("skin_thickness_cp", units="m")),
            "spar_cp_m": _f(gv("spar_thickness_cp", units="m")),
        },
        "flight": {
            "v_ms": config.V_MS, "rho_kgm3": config.RHO, "mach": config.MACH,
            "re_per_m": config.RE_PER_M, "altitude_ft": config.ALTITUDE_FT,
            "ktas": config.KTAS,
            "CL": float(gv(aa.POINT + ".wing_perf.CL")[0]),
            "CD": float(gv(aa.POINT + ".wing_perf.CD")[0]),
            "lift_lb": float(gv("lift")[0]) / LB_N,
            "drag_N": float(gv("drag")[0]),
            "S_ref_m2": float(gv(aa.POINT + ".wing.S_ref")[0]),
        },
        "sizing": {
            "case_name": SIZING_CASE,
            "cruise_case_name": CRUISE_CASE,
            "Nz": aa.NZ,
            "W_maneuver_lb": aa.W_MAN_LB,
            "safety_factor": aa.SAFETY,
            "aero_scale": float(aero_scale),
            # Each nacelle as the model carries it: WingCalc's mass (the inboard one
            # with its gear) at WingCalc's x/c and delta_Z, and the CG that puts it at.
            "nacelles": [
                {"name": n, "weight_lb": float(m), "ws_in": float(y), "xc": float(xc),
                 "dz_in": float(dz), "cg_in": [float(v) for v in loc]}
                for (n, m, y, xc, dz), loc in zip(
                    aa.NACELLES, aa.nacelle_locations(prob) / IN_M)
            ],
            "E_psi": aa.E_PA / PSI_PA,
            "G_psi": aa.G_PA / PSI_PA,
            "sigma_tension_psi": aa.SIG_T / PSI_PA,
            "sigma_compression_psi": aa.SIG_C / PSI_PA,
            "mrho_lb_in3": aa.MRHO / 27679.9047,
            "failure_ks": float(gv("failure")[0]),
            "material": "UD HARD (Materials.csv), isotropic equivalent",
        },
        "weight": {
            "box_half_lb": float(struct["element_mass_lb"][:n_elem].sum()),
            "box_half_uncut_lb": float(struct["element_mass_lb"].sum()),
            "W_box_lb": float(gv("W_box_lb")[0]),
            "W_wing_lb": float(gv("W_wing_lb")[0]),
            "m_batt_lb": float(gv("m_batt_lb")[0]),
            "R_nmi": float(gv("R_nmi")[0]),
            # WingCalc's build-up, computed by this model from its own box and
            # geometry (studies/vsp_planform/weight). Every term, in WingCalc's
            # names, so the Weight Summary tabs of the two reports read row for row.
            "buildup": _buildup_block(prob.model.weight_buildup.last),
        },
        "grid": {
            "n_nodes_full": int(ny),
            "n_nodes": n_node,
            "n_elements": n_elem,
            "cut_index": i_cut,
            "cut_ws_in": float(y_in[i_cut]),
            "semispan_full_in": semi_in,
            "wc_bay_ws_in": list(WC_BAY_WS_IN),
        },
        "stations": {
            "ws_in": _f(y_in[:n_node]),
            "x_le_in": _f(x_le_in[:n_node]),
            "x_te_in": _f(x_te_in[:n_node]),
            "chord_in": _f(chord_in[:n_node]),
            "x_fwd_in": _f((x_le_in + fwd_ratio * chord_in)[:n_node]),
            "x_aft_in": _f((x_le_in + aft_ratio * chord_in)[:n_node]),
            "x_qc_in": _f((x_le_in + 0.25 * chord_in)[:n_node]),
            "x_ea_in": _f(x_ea_in[:n_node]),
            "z_qc_in": _f(z_qc_in[:n_node]),
            "z_ea_in": _f(z_ea_in[:n_node]),
            "fwd_ratio": _f(fwd_ratio[:n_node]),
            "aft_ratio": _f(aft_ratio[:n_node]),
            "twist_deg": _f(twist_abs[:n_node]),
        },
        "elements": elems,
        "struct": {k: _cut_e(v) for k, v in struct.items()},
        "vonmises_psi": [_f(vonmises_psi[i]) for i in range(n_elem)],
        "aero": {
            "ws_mid_in": _f(y_mid_in[:n_elem]),
            "width_in": _f(panel_width_in[:n_elem]),
            "fz_1g_lb": _f(panel_fz_lb[:n_elem]),
            "fx_1g_lb": _f(panel_fx_lb[:n_elem]),
            # The same arrays over the FULL span, winglet included. The VLM loads
            # the winglet and the FEM carries that load, so the Loading tab draws
            # aero to the tip and marks the structural cut instead of stopping at it.
            "fz_1g_lb_uncut": _f(panel_fz_lb),
            "fx_1g_lb_uncut": _f(panel_fx_lb),
            "ws_mid_in_uncut": _f(y_mid_in),
            "width_in_uncut": _f(panel_width_in),
            "chord_in_uncut": _f(0.5 * (chord_in[:-1] + chord_in[1:])),
        },
        "vmt": vmt_cases,
        "winglet": winglet,
    }
    return snap


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    logs = _STUDY / "out" / "logs"
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=str(logs / "arc_aerostruct_A_optimal_e694.json"))
    ap.add_argument("--case", default="0",
                    help="index into the converged JSON, or a substring of one label")
    ap.add_argument("--seed", default=None,
                    help="arc_optimal_toc JSON the frozen geometry comes from")
    ap.add_argument("--out", default=str(logs / "oas_report_snapshot.json"))
    a = ap.parse_args(argv)

    snap = build_snapshot(a.json, case=a.case, seed_json=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(snap, indent=1))
    print(f"wrote {a.out}  ({os.path.getsize(a.out) / 1e6:.2f} MB)")
    print(f"  case      {snap['source']['case_label']}")
    print(f"  box half  {snap['weight']['box_half_lb']:.2f} lb "
          f"(uncut {snap['weight']['box_half_uncut_lb']:.2f}, "
          f"winglet {snap['winglet']['box_mass_lb']:.2f})")
    print(f"  cut at WS {snap['grid']['cut_ws_in']:.2f} in, "
          f"{snap['grid']['n_elements']} elements")
    return snap


if __name__ == "__main__":
    main()
