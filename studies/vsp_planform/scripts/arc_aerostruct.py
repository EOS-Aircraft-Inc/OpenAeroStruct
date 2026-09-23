"""Arc A with OAS's own wingbox in the loop: minimize DRAG AND WEIGHT, not drag alone.

``arc_optimal_toc.py`` minimizes drag at fixed lift and then hands the converged
planform to WingCalc for a weight. Weight is therefore exogenous: the twist
optimizer sees no penalty for carrying lift outboard, so it converges on the
elliptic spanload, which is the minimum-induced-drag answer and not the
minimum-range-cost one. Putting WingCalc itself inside the SLSQP loop does not
work -- its bay sizer is a differential-evolution search, so its output is not
smooth in the design variables and a gradient method reads its noise as signal.

This script closes the loop with OAS's own wingbox FEM instead, which is smooth
and differentiable throughout. The aero model is untouched: same mesh, same
per-panel ``c_max_t``, same trim. The structure hangs off the geometry group's
existing ``wing.mesh`` and ``wing.t_over_c`` outputs, which is exactly the seam
``AerostructGeometry`` uses (Geometry -> WingboxGroup -> SpatialBeamSetup); only
the stock ``Geometry`` is replaced by this study's region-based one.

Three things here are not obvious and all three were measured, not assumed:

1. THE MESH RUNS THE WRONG WAY FOR THE FEM. ``structures/fem.py`` clamps
   spanwise node ``ny - 1`` under symmetry, because a stock OAS half-wing mesh
   runs TIP -> ROOT with y <= 0. This study's mesh runs ROOT -> TIP with y >= 0.
   The VLM does not care about spanwise ordering, which is why the aero-only
   study never noticed, but the FEM does: fed directly it clamps the winglet tip
   and leaves the root free. The root then carries ~1 MPa, the tip carries
   50 GPa, and the KS failure comes back at +154. :class:`MirrorSpanwise`
   reflects the mesh and the loads into OAS's convention (reverse the span index,
   negate y). A reflection is exact -- the structure is identical and only the
   ordering of per-element output changes, which is flipped back before anything
   is reported.

2. THE NACELLES DOMINATE THE RELIEF, AND THEY TWIST THE BOX. ``wingLoadingIn.csv``
   hangs 8,205 lb (nacelle plus main gear) at WS 176.5 and 7,002 lb at WS 356 --
   over 15,000 lb per half wing against ~1,700 lb of box, worth about 30% of the
   root bending moment at 2.5 g. Sized without them OAS returns 4,321 lb of box
   against WingCalc's 3,379 lb. They sit where WingCalc puts them (``nacelle_spec``):
   x/c 0.177 and 0.205 aft of the local LE, 1.73 and 21.5 in above the elastic axis,
   i.e. FORWARD of the box. An earlier version put them on the axis, which agreed
   with WingCalc to 0.26% -- by zeroing their nose-down torque, so the spar webs sat
   at minimum gauge. Placed properly the box is 3,418 lb, +1.2%: a worse match from
   a better model, because the old agreement was cancellation.

3. THE SIZING SPANLOAD IS THE SCALED CRUISE ONE. That is not a shortcut, it is
   what the coupling already does: ``write_deck`` exports the 1 g MTOW spanload
   and WingCalc scales it by Nz. Scaling here keeps the two models fed the same
   thing, and keeps this comparable to every weight the study has reported.

What is held: ``taper_B``, ``t_over_c_cp`` and ``wingbox_pct`` come from the
converged Arc A run and do not move. Arc A solves its taper in closed form from
the 7 in aileron depth requirement, and depth depends on t/c and chord but not on
twist, so freezing them holds the depth at its delivered 7.00 in and isolates the
trade this script exists to measure: twist and skin/spar thickness against range.
"""

import argparse
import json
import os
import sys

import numpy as np
import openmdao.api as om

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

from studies.vsp_planform import config, param                      # noqa: E402
import studies.vsp_planform.run_opt as ro                           # noqa: E402
from studies.vsp_planform.run_opt import POINT, trim_alpha          # noqa: E402
from studies.vsp_planform.param import twist_cp_bounds              # noqa: E402
from studies.vsp_planform.coupling import mission                   # noqa: E402
from studies.vsp_planform.coupling import geometry as wgeom         # noqa: E402
from studies.vsp_planform.weight.deck_inputs import read_weight_deck   # noqa: E402
from studies.vsp_planform.weight.component import WingWeightBuildUp, fem_origin  # noqa: E402
import arc_optimal_toc as aot                                       # noqa: E402
import wing2_oas as w2                                              # noqa: E402

from openaerostruct.structures.wingbox_group import WingboxGroup                       # noqa: E402
from openaerostruct.structures.spatial_beam_setup import SpatialBeamSetup              # noqa: E402
from openaerostruct.structures.spatial_beam_states import SpatialBeamStates            # noqa: E402
from openaerostruct.structures.spatial_beam_functionals import SpatialBeamFunctionals  # noqa: E402
from openaerostruct.transfer.load_transfer import LoadTransfer                         # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")
IN_M, LB_N, LB_KG = 0.0254, 4.4482216, 0.45359237
PSI_PA, LBIN3_KGM3 = 6894.757293, 27679.9047

# --- sizing case: "2.5g Upbend1 flight" from the deck's loadCasesIn.csv. Nz 2.5 at
#     81,920 lb (Fuel_fraction 0, so no wing-fuel relief), ultimate factor 1.5. The
#     load applied here is the LIMIT load; ultimate comes from `safety_factor`,
#     because failure_ks.py tests against yield / safety_factor.
NZ, W_MAN_LB, SAFETY = 2.5, 81_920.0, 1.5

# --- the deck WingCalc reads, with the overrides write_deck applies. The nacelles
#     and the whole non-box weight come out of it, so both tools see one installation.
WEIGHT_DECK = read_weight_deck()


def nacelle_spec(deck=WEIGHT_DECK, lg_state="Retracted"):
    """(name, mass lb, WS in, x/c, delta_Z in) per nacelle -- WingCalc's own reading.

    ``loading/wing_loading.py`` hangs each nacelle at ``x/c`` aft of the LOCAL
    leading edge and ``delta_Z`` above the elastic axis, and the gear rides inside
    the inboard nacelle, so the inboard mass there is nacelle PLUS gear
    (``io/inputs.inboard_nacelle_weight``). The gear state picks the inboard x/c and
    delta_Z; the sizing case (2.5g Upbend1) flies with it retracted.

    Both are deliberately the flight values, not the ``_no_mlg`` ones: those describe
    the gear-free fitting condition the nacelle REINFORCEMENT is sized on, which the
    weight build-up reads separately.
    """
    wl, dep = deck.wl, lg_state == "Deployed"
    return (
        ("Inboard", wl("Inboard nacelle weight wo mlg") + wl("Mlg weight"),
         deck.inboard_nacelle_Y,
         wl("x/c ncl ib deployed" if dep else "x/c ncl ib retracted"),
         wl("delta_Z NCL ib deployed" if dep else "delta_Z NCL ib retracted")),
        ("Outboard", wl("Outboard nacelle weight"), deck.outboard_nacelle_Y,
         wl("x/c ncl ob"), wl("delta_Z NCL ob")),
    )


NACELLES = nacelle_spec()

# --- material: UD HARD from the deck's Materials.csv, as an isotropic equivalent.
#     The card gives smeared LAMINATE properties for three layups, not lamina data,
#     so the Tsai-Wu path (which needs E2, sigma_t2, ...) cannot be fed from it.
#     Tension (OHT) is the `yield`; the compression-critical upper skin gets
#     min(CAI/OHC) through `strength_factor_for_upper_skin`.
E_PA, G_PA = 11_799_435.16 * PSI_PA, 2_302_535.824 * PSI_PA
SIG_T, SIG_C = 70_128.79237 * PSI_PA, 35_681.82745 * PSI_PA
MRHO = 0.056 * LBIN3_KGM3

# --- weight. The wing is WingCalc's build-up, computed from THIS model: the sized
#     OAS box, plus ribs, fasteners, splices, nacelle and gear reinforcement, ply
#     drops, Torenbeek secondary structure, lightning mesh and paint, each from the
#     OAS geometry by WingCalc's rule (studies/vsp_planform/weight). There is no
#     calibration factor and no WingCalc number in it. An earlier version carried the
#     3,437 lb of non-box weight as a constant held from one WingCalc run and scaled
#     the box by WingCalc's W_bays; that was right only while the planform was frozen,
#     and it made WingCalc a hidden input to every OAS answer.

N_CP_T = 5                     # skin / spar thickness control points
T_BOUNDS_M = (0.0015, 0.10)     # 0.059 in is about eight plies at 0.0075 in


class MirrorSpanwise(om.ExplicitComponent):
    """Reflect an (n0, ny, 3) field into OAS's structural span convention.

    Reverses the spanwise index and negates y, turning this study's ROOT -> TIP,
    y >= 0 arrays into OAS's TIP -> ROOT, y <= 0 ones. ``scale`` folds the
    maneuver load factor into the same pass, for ``sec_forces``.
    """

    def initialize(self):
        self.options.declare("shape", types=tuple)
        self.options.declare("units", types=str, default="m")
        self.options.declare("scale", types=float, default=1.0)

    def setup(self):
        n0, n1 = self.options["shape"]
        u, k = self.options["units"], self.options["scale"]
        self.add_input("mirror_in", shape=(n0, n1, 3), units=u)
        self.add_output("mirror_out", shape=(n0, n1, 3), units=u)

        idx = np.arange(n0 * n1 * 3).reshape(n0, n1, 3)
        self.declare_partials(
            "mirror_out", "mirror_in",
            rows=np.arange(n0 * n1 * 3), cols=idx[:, ::-1, :].ravel(),
            val=np.tile(np.array([1.0, -1.0, 1.0]), n0 * n1) * k)
        self._k, self._sgn = k, np.array([1.0, -1.0, 1.0])

    def compute(self, inputs, outputs):
        outputs["mirror_out"] = self._k * inputs["mirror_in"][:, ::-1, :] * self._sgn


class ReverseVec(om.ExplicitComponent):
    """Reverse a per-panel vector, so it matches the mirrored span ordering."""

    def initialize(self):
        self.options.declare("size", types=int)

    def setup(self):
        n = self.options["size"]
        self.add_input("rev_in", shape=n)
        self.add_output("rev_out", shape=n)
        self.declare_partials("rev_out", "rev_in", rows=np.arange(n),
                              cols=np.arange(n)[::-1], val=np.ones(n))

    def compute(self, inputs, outputs):
        outputs["rev_out"] = inputs["rev_in"][::-1]


def wingbox_section(name="e694", front=0.120, rear=0.750, n=40):
    """Normalized wingbox skin coordinates between the two spar stations.

    OAS scales these by chord and by ``t_over_c / original_wingbox_airfoil_t_over_c``,
    so they have to come from the section the t/c distribution is quoted against.
    Upper and lower must share their first and last x. Arc A blends to goe16k
    outboard of 524 in, but the box mass is overwhelmingly inboard of that, so the
    inboard section is the one that matters here.
    """
    import aerosandbox as asb
    af = asb.Airfoil(name)
    xs = np.linspace(front, rear, n)

    def surf(x, upper):
        t = float(af.local_thickness(x_over_c=x))
        c = float(af.local_camber(x_over_c=x))
        return c + 0.5 * t if upper else c - 0.5 * t

    toc = float(max(af.local_thickness(x_over_c=x) for x in np.linspace(0.02, 0.98, 400)))
    return (xs.astype("complex128"), xs.astype("complex128"),
            np.array([surf(x, True) for x in xs], dtype="complex128"),
            np.array([surf(x, False) for x in xs], dtype="complex128"), toc)


def struct_surface(mesh_rt, toc_cp, airfoil="e694", front=0.120, rear=0.750):
    """The surface dict the FEM sees: the MIRRORED mesh, plus box and material.

    OAS takes ONE box section for the whole beam, so a rear spar that moves across
    the span (Arc B's 0.750c -> 0.550c) cannot be represented: the caller passes
    the INBOARD fraction, where the box mass is. The weight build-up is not bound by
    this -- it follows the real schedule station by station.
    """
    xu, xl, yu, yl, toc_orig = wingbox_section(airfoil, front=front, rear=rear)
    mirrored = mesh_rt[:, ::-1, :].copy()
    mirrored[:, :, 1] *= -1.0
    return {
        "name": "wing", "symmetry": True, "S_ref_type": "projected",
        "mesh": mirrored, "fem_model_type": "wingbox",
        "data_x_upper": xu, "data_x_lower": xl, "data_y_upper": yu, "data_y_lower": yl,
        "original_wingbox_airfoil_t_over_c": toc_orig,
        # Seeded from the sized datum, in the MIRRORED ordering: cp[0] is the tip
        # and cp[-1] is the root. Starting at a feasible box costs nothing and
        # keeps SLSQP off an infeasible first arc.
        "spar_thickness_cp": np.array([0.0015, 0.0015, 0.0018, 0.0022, 0.0027]),
        "skin_thickness_cp": np.array([0.0015, 0.0022, 0.0045, 0.0095, 0.0183]),
        "t_over_c_cp": np.asarray(toc_cp)[::-1].copy(),
        "twist_cp": np.zeros(N_CP_T),
        "E": E_PA, "G": G_PA, "yield": SIG_T, "mrho": MRHO,
        "safety_factor": SAFETY,
        "strength_factor_for_upper_skin": SIG_C / SIG_T,
        "wing_weight_ratio": 1.0,
        "exact_failure_constraint": False,
        "struct_weight_relief": True,
        "distributed_fuel_weight": False,
        "n_point_masses": 2,
        "with_viscous": config.WITH_VISCOUS, "with_wave": config.WITH_WAVE,
        "k_lam": config.K_LAM, "c_max_t": 0.405, "CL0": 0.0, "CD0": 0.0,
    }


def add_structure(model, mesh, regions, surf_s, scale, schedule, blend):
    """Hang the wingbox FEM off the geometry group. Called by ``build_problem``.

    ``extra`` is the last thing before ``setup()``, so the aero point already
    exists and its ``sec_forces`` can be connected here. ``schedule`` is the rear
    spar and ``blend`` the design section, both from the seed JSON: the weight
    build-up measures the box and the airfoil on the wing's OWN section, the same
    one ``write_deck`` exports to WingCalc.
    """
    nx, ny = surf_s["mesh"].shape[0], surf_s["mesh"].shape[1]

    ivc = om.IndepVarComp()
    ivc.add_output("load_factor", val=NZ)
    ivc.add_output("point_masses", val=np.array([m * LB_KG for _, m, *_ in NACELLES]), units="kg")
    ivc.add_output("point_mass_locations", val=np.zeros((2, 3)), units="m")
    ivc.add_output("engine_thrusts", val=np.zeros(2), units="N")
    model.add_subsystem("struct_vars", ivc, promotes=["*"])

    model.add_subsystem("mirror_mesh", MirrorSpanwise(shape=(nx, ny), units="m"),
                        promotes_outputs=[("mirror_out", "mesh_s")])
    model.add_subsystem("mirror_forces",
                        MirrorSpanwise(shape=(nx - 1, ny - 1), units="N", scale=scale),
                        promotes_outputs=[("mirror_out", "sec_forces_s")])
    model.add_subsystem("mirror_toc", ReverseVec(size=ny - 1),
                        promotes_outputs=[("rev_out", "t_over_c_s")])

    model.connect("wing.mesh", ["mirror_mesh.mirror_in"])
    model.connect(f"{POINT}.aero_states.wing_sec_forces", "mirror_forces.mirror_in")
    model.connect("wing.t_over_c", "mirror_toc.rev_in")

    model.add_subsystem("load_transfer", LoadTransfer(surface=surf_s),
                        promotes_inputs=[("def_mesh", "mesh_s"), ("sec_forces", "sec_forces_s")],
                        promotes_outputs=["loads"])
    model.add_subsystem("wingbox_group", WingboxGroup(surface=surf_s),
                        promotes_inputs=[("mesh", "mesh_s"), ("t_over_c", "t_over_c_s"),
                                         "skin_thickness_cp", "spar_thickness_cp"],
                        promotes_outputs=["A", "Iy", "Iz", "J", "Qz", "A_enc", "A_int",
                                          "htop", "hbottom", "hfront", "hrear",
                                          "skin_thickness", "spar_thickness"])
    model.add_subsystem("struct_setup", SpatialBeamSetup(surface=surf_s),
                        promotes_inputs=[("mesh", "mesh_s"), "A", "Iy", "Iz", "J", "A_int"],
                        promotes_outputs=["nodes", "local_stiff_transformed",
                                          "structural_mass", "cg_location", "element_mass"])
    model.add_subsystem("struct_states", SpatialBeamStates(surface=surf_s),
                        promotes_inputs=["local_stiff_transformed", "loads", "nodes",
                                         "element_mass", "load_factor",
                                         "point_mass_locations", "point_masses",
                                         "engine_thrusts"],
                        promotes_outputs=["disp"])
    model.add_subsystem("struct_funcs", SpatialBeamFunctionals(surface=surf_s),
                        promotes_inputs=["spar_thickness", "disp", "Qz", "J", "A_enc",
                                         "htop", "hbottom", "hfront", "hrear", "nodes"],
                        promotes_outputs=["vonmises", "failure"])

    # The wing weight: WingCalc's build-up on this model's box and geometry. Only the
    # LE and TE rows of the mesh are geometry to it, so only those are connected --
    # the partials against them are differenced, and 210 entries cost a third of 630.
    x_grid = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 201)))   # the export's grid
    profile_at = (wgeom.blended_profile(blend, x_grid) if isinstance(blend, dict)
                  else (lambda _y, p=wgeom.database_profile(blend or "e694", x_grid): p))
    bp = np.array(schedule, dtype=float)
    model.add_subsystem(
        "weight_buildup",
        WingWeightBuildUp(surface=surf_s, deck=WEIGHT_DECK,
                          aft_at=lambda ws: float(np.interp(ws, bp[:, 0], bp[:, 1])),
                          fwd_ratio=float(config.WINGBOX_FRONT_PCT),
                          profile_at=profile_at, x_grid=x_grid),
        promotes_inputs=["element_mass", "A_enc", "A"],
        promotes_outputs=["W_wing_lb", "W_box_lb"])
    # FLAT indices, one dimension. A shaped index array is read by OpenMDAO as a
    # multi-dimensional index sequence, and the input then silently keeps its
    # default of 1.0 everywhere -- a wing with zero chord.
    idx = np.arange(nx * ny * 3).reshape(nx, ny, 3)[[0, -1]].ravel()
    model.connect("mesh_s", "weight_buildup.le_te", src_indices=idx, flat_src_indices=True)
    model.connect("t_over_c_s", "weight_buildup.t_over_c")

    # Range is the objective because it IS the trade: R = eta * e_star * m_batt / D
    # with m_batt = MTOW - K - payload - fuel - W_wing, so drag and weight enter on
    # one currency and nothing has to be weighted by hand.
    k_const = mission.ETA_PROP * mission.E_STAR_WH_KG * 3600.0 * mission.LB_KG / mission.NMI_M
    m_fixed = mission.MTOW_LB - mission.K_EX_BATT_LB - mission.PAYLOAD_LB - mission.FUEL_LB
    model.add_subsystem(
        "perf",
        om.ExecComp(
            ["m_batt_lb = m_fixed - W_wing_lb",
             "R_nmi = k * (m_fixed - W_wing_lb) / drag"],
            W_wing_lb={"val": 6.8e3}, drag={"units": "N", "val": 1.1e4},
            m_batt_lb={"val": 1.7e4}, R_nmi={"val": 3.4e2},
            m_fixed=m_fixed, k=k_const,
        ),
        promotes_inputs=["W_wing_lb"],
        promotes_outputs=["m_batt_lb", "R_nmi"],
    )
    model.connect("drag", "perf.drag")


def build(json_path, twist_bounds, aero_dv=True, thick_dv=True):
    """Rebuild the shipped Arc A geometry, with the wingbox FEM attached.

    ``aero_dv`` / ``thick_dv`` select which half of the design space is free. A
    quantity that is held is left OUT of the design variables and its constraint
    is dropped with it, rather than carried at fixed bounds: a constraint no
    design variable can move is an all-zero Jacobian row, which is what puts
    SLSQP into a degenerate active set (see ``run_opt.add_optimization``).
    """
    with open(json_path) as f:
        J = json.load(f)

    # EVERY ARC, from the design's own JSON. This used to hard-code Arc A -- its
    # spar schedule, its straight-aft-spar fraction, its e694/goe16k blend -- which
    # is right for this script's own runs but wrong for anything else handed to it:
    # arc_optimal_toc's report hook runs for B and C too, and would have drawn Arc
    # A's geometry under Arc B's twist without a word. The JSON records the arc, the
    # schedule, the spar fraction and the section; the arc tables only supply the
    # region-A rule and pin, which are the arc's definition rather than a result.
    arc = J.get("arc", "A")
    y_a, rule, pin_p, _ = aot.ARCHS[arc]
    rule = J.get("region_a_rule") or rule          # a run that overrode it says so
    schedule = tuple(tuple(float(v) for v in bp) for bp in J["rear_schedule"])
    aot.RET, aot.C_MAX_T = aot.section(J["airfoil"])
    blend = aot.SECTION_BLEND.get(arc)
    if blend is not None:
        aot.RET_AT, aot.CMT_AT, _ = aot.blended_section(*blend)
    else:
        aot.RET_AT = aot.CMT_AT = None

    spar_ail = aot.spar_at_aileron(schedule)
    stations = aot.width_stations(spar_ail, J["chord_at_aileron_in"])
    w2.REAR_SCHEDULE, w2.WIDTH_STATIONS = schedule, stations
    config.WINGBOX_FRONT_PCT = w2.FRONT_PCT
    config.WINGBOX_REAR_SCHEDULE = schedule
    config.WINGBOX_WIDTH_STATIONS = stations
    saved_rule = param.REGION_A_RULE[w2.BASELINE]
    param.REGION_A_RULE[w2.BASELINE] = rule

    try:
        mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, y_a)
        if pin_p is not None:
            # A pinned straight line rebuilds the planform around it, exactly as
            # arc_optimal_toc.optimize does (Arc B's straight forward spar).
            planform0 = param.baseline_planform(stick, regions, rule=rule)
        # The FEM's box section is the design's own INBOARD section: the box mass
        # is overwhelmingly inboard, and OAS takes one section for the whole beam.
        sb = J.get("section_blend")
        box_af = sb["inboard"] if isinstance(sb, dict) else (J.get("airfoil") or "e694")
        box_af = "e694" if box_af in (None, "", "as-built") else box_af
        surf_s = struct_surface(mesh, J["t_over_c_cp"], airfoil=box_af,
                                front=float(w2.FRONT_PCT), rear=float(schedule[0][1]))
        # The maneuver spanload is the cruise one scaled by Nz * W_man / W_cruise,
        # which is how the deck already feeds WingCalc.
        scale = NZ * W_MAN_LB / (w2.W / LB_N)

        # c_max_t is read once when the viscous component is built, so the blended
        # section has to be injected before the problem is. Same wrapper as
        # arc_optimal_toc.optimize uses.
        orig_build_surface = ro.build_surface

        def _surface(mesh_, stick_, regions_, **kw):
            sd = orig_build_surface(mesh_, stick_, regions_, **kw)
            ym = np.abs(np.asarray(mesh_)[0, :, 1]) / config.SCALE
            yp = 0.5 * (ym[:-1] + ym[1:])
            sd["c_max_t"] = (aot.C_MAX_T if aot.CMT_AT is None
                             else np.array([aot.CMT_AT(v) for v in yp]))
            return sd

        ro.build_surface = _surface
        try:
            prob, _ = ro.build_problem(
                w2.BASELINE, mesh, stick, regions, planform0,
                extra=lambda m, ms, rg: add_structure(
                    m, ms, rg, surf_s, scale, schedule,
                    J.get("section_blend") or J.get("airfoil")))
        finally:
            ro.build_surface = orig_build_surface
    finally:
        param.REGION_A_RULE[w2.BASELINE] = saved_rule

    model = prob.model
    if aero_dv:
        tw_lo, tw_up = twist_cp_bounds(mesh, config.N_TWIST_CP, bounds=twist_bounds)
        model.add_design_var("wing.twist_cp", lower=tw_lo, upper=tw_up, units="deg")
        model.add_design_var("alpha", lower=-5.0, upper=12.0, units="deg")
        model.add_constraint("twist_abs", lower=twist_bounds[0], upper=twist_bounds[1],
                             units="deg", ref=twist_bounds[1])
        model.add_constraint("lift", equals=w2.W, ref=w2.W)
    if thick_dv:
        model.add_design_var("skin_thickness_cp", lower=T_BOUNDS_M[0],
                             upper=T_BOUNDS_M[1], scaler=1e2)
        model.add_design_var("spar_thickness_cp", lower=T_BOUNDS_M[0],
                             upper=T_BOUNDS_M[1], scaler=1e2)
        model.add_constraint("failure", upper=0.0)
    return prob, mesh, J, surf_s


def apply_frozen(prob, J, seed=None):
    """Push back everything setup() resets: the held geometry and the seed.

    ``seed`` carries a converged aero state forward from an earlier stage, so the
    sizing pass sees the spanload the drag optimizer actually produced.
    """
    prob.set_val("wing.wingbox_pct", float(J["wingbox_pct"]))
    prob.set_val("wing.taper_B", J["taper_B"])
    prob.set_val("wing.t_over_c_cp", np.array(J["t_over_c_cp"]))
    s = seed or {}
    prob.set_val("wing.twist_cp", np.array(s.get("twist_cp_deg", J["twist_cp"])), units="deg")
    prob.set_val("alpha", s.get("alpha_deg", J["alpha"]), units="deg")
    if "skin_cp_m" in s:
        prob.set_val("skin_thickness_cp", np.array(s["skin_cp_m"]), units="m")
        prob.set_val("spar_thickness_cp", np.array(s["spar_cp_m"]), units="m")


def nacelle_locations(prob):
    """Each nacelle CG where WingCalc puts it, in OAS's mirrored frame, m.

    ``x/c`` aft of the local leading edge along the TRUE chord, and ``delta_Z`` above
    the elastic axis -- ``loading/wing_loading.py``'s reading, which is the whole
    point. The elastic axis here is OAS's own (the FEM node line), since that is the
    axis the beam reacts the offset about.

    An earlier version put both CGs ON that axis, for pure vertical relief. That zeroed
    their torque outright: at 2.5 g two nacelles forward of the box are over a
    million in-lb of nose-down torsion, several times everything else on the wing,
    and the spar webs were being sized as if it were not there.
    """
    m = np.asarray(prob.get_val("wing.mesh", units="m")) / IN_M      # root -> tip, y >= 0
    le, te = m[0], m[-1]
    ws = le[:, 1]
    chord = np.linalg.norm(te - le, axis=1)
    w_fem = fem_origin(prob.model.weight_buildup.options["surface"])
    z_ea = le[:, 2] + w_fem * (te[:, 2] - le[:, 2])
    pml = np.zeros((len(NACELLES), 3))
    for k, (_n, _m, y, xc, dz) in enumerate(NACELLES):
        pml[k] = (np.interp(y, ws, le[:, 0]) + xc * np.interp(y, ws, chord),
                  -y, np.interp(y, ws, z_ea) + dz)
    return pml * IN_M


def place_nacelles(prob):
    """Set the nacelle CGs. The mesh must exist, so this follows a run_model.

    Placed once, from the seed geometry, and then held: a nacelle is installed at a
    station, and the few millimetres its CG would drift as twist rotates the section
    under it are not a design freedom.
    """
    prob.set_val("point_mass_locations", nacelle_locations(prob), units="m")


def report(prob, mesh, label):
    """Everything the comparison needs, including where the lift actually sits."""
    def gv(n, **k):
        return np.asarray(prob.get_val(n, **k))

    sec = gv(POINT + ".aero_states.wing_sec_forces", units="N")
    y_in = np.abs(mesh[0, :, 1]) / config.SCALE
    y_mid = 0.5 * (y_in[:-1] + y_in[1:])
    fz = sec[:, :, 2].sum(axis=0)
    y_cl = float(np.sum(fz * y_mid) / np.sum(fz))
    semi = float(y_in[-1])
    em = gv("element_mass", units="kg").ravel()[::-1]
    tw = gv("twist_abs", units="deg").ravel()
    return {
        "label": label,
        "drag_N": float(gv("drag")[0]),
        "lift_lb": float(gv("lift")[0]) / LB_N,
        "CL": float(gv(POINT + ".wing_perf.CL")[0]),
        "CD": float(gv(POINT + ".wing_perf.CD")[0]),
        "alpha_deg": float(gv("alpha", units="deg")[0]),
        "W_box_lb": float(gv("W_box_lb")[0]),
        "W_wing_lb": float(gv("W_wing_lb")[0]),
        "m_batt_lb": float(gv("m_batt_lb")[0]),
        "R_nmi": float(gv("R_nmi")[0]),
        "failure": float(gv("failure")[0]),
        "twist_root_deg": float(tw[0]),
        "twist_tip_deg": float(tw[-1]),
        "twist_abs_deg": tw.tolist(),
        "twist_cp_deg": gv("wing.twist_cp", units="deg").ravel().tolist(),
        "y_cl_in": y_cl,
        "y_cl_frac_semi": y_cl / semi,
        "skin_root_in": float(gv("skin_thickness", units="m").ravel()[-1] / IN_M),
        "spar_root_in": float(gv("spar_thickness", units="m").ravel()[-1] / IN_M),
        # The sized box itself, not just its root. Without these the design point
        # cannot be REBUILT -- twist and alpha alone reproduce the aero but leave
        # the box wherever the seed put it, which is a wing that weighs 2,703 lb
        # and fails at +0.34. studies/vsp_planform/viewer needs exactly this to
        # draw a report without re-running the sizing stage. Note the ordering is
        # the MIRRORED one, tip -> root, because that is the ordering
        # `struct_surface` builds the surface dict in and the ordering
        # `apply_frozen` pushes them back in.
        "skin_cp_m": gv("skin_thickness_cp", units="m").ravel().tolist(),
        "spar_cp_m": gv("spar_thickness_cp", units="m").ravel().tolist(),
        "box_mass_half_lb": float(em.sum() / LB_KG),
        # Every term of the build-up, so a converged point carries its own weight
        # statement and the report can show it without re-running anything.
        "weight_buildup": {k: float(v) for k, v in prob.model.weight_buildup.last.items()
                           if k != "detail"},
        "nacelle_locations_in": (nacelle_locations(prob) / IN_M).tolist(),
        "spanload_lb": (fz / LB_N).tolist(),
        "y_mid_in": y_mid.tolist(),
        "success": bool(prob.driver.result.success),
    }


def _solve(json_path, twist_lo, objective, aero_dv, thick_dv, seed=None):
    """One SLSQP stage. Returns the converged problem, its mesh and its state."""
    prob, mesh, J, _ = build(json_path, (twist_lo, config.TWIST_BOUNDS[1]),
                             aero_dv=aero_dv, thick_dv=thick_dv)
    obj = {"drag": ("drag", dict(ref=1.0e4)),
           # Maximize: a negative scaler flips the sense. The reference is the
           # shipped 337.6 nmi, so the scaled objective starts at about -1.
           "range": ("R_nmi", dict(scaler=-1.0 / 337.6)),
           "mass": ("structural_mass", dict(ref=1.5e3))}[objective]
    prob.model.add_objective(obj[0], **obj[1])

    prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", tol=1e-7, maxiter=200, disp=True)
    prob.setup()
    apply_frozen(prob, J, seed)
    prob.run_model()
    place_nacelles(prob)          # needs `nodes`, so it follows the first run_model
    prob.run_model()
    print("  seed: drag {:.1f} N, W_box {:.1f} lb, failure {:+.4f}".format(
        float(prob.get_val("drag")[0]), float(prob.get_val("W_box_lb")[0]),
        float(prob.get_val("failure")[0])), flush=True)
    prob.run_driver()
    return prob, mesh, J


def run_case(objective, twist_lo, json_path, label):
    """One design point.

    The drag objective runs in TWO stages, because with drag as the objective the
    thickness variables do not appear in it at all: SLSQP has no reason to move
    them and leaves the box wherever feasibility put it, which is not a weight
    anyone should quote. Stage 1 optimizes the aero exactly as the shipped study
    does; stage 2 then sizes the box to minimum mass against that spanload, which
    is the job WingCalc does today. The range objective needs only one stage --
    it already prices the box, so minimum mass is implied.
    """
    if objective == "drag":
        p1, _, _ = _solve(json_path, twist_lo, "drag", aero_dv=True, thick_dv=False)
        seed = {"twist_cp_deg": np.asarray(p1.get_val("wing.twist_cp", units="deg")).ravel().tolist(),
                "alpha_deg": float(p1.get_val("alpha", units="deg")[0])}
        ok1 = bool(p1.driver.result.success)
        print("  stage 2: sizing the box against the drag-optimal spanload", flush=True)
        prob, mesh, _ = _solve(json_path, twist_lo, "mass",
                               aero_dv=False, thick_dv=True, seed=seed)
        r = report(prob, mesh, label)
        r["success"] = bool(r["success"]) and ok1
    else:
        prob, mesh, _ = _solve(json_path, twist_lo, "range", aero_dv=True, thick_dv=True)
        r = report(prob, mesh, label)
    r["objective"], r["twist_lower_deg"] = objective, twist_lo
    return r


def _case_slug(r, i):
    return "case{}_{}_twist{:+.0f}".format(i, r["objective"], r["twist_lower_deg"])


def results_dir(out_json):
    """Where a study's reports go: a folder named after its JSON, beside it."""
    stem = os.path.splitext(os.path.basename(out_json))[0]
    return os.path.join(os.path.dirname(out_json), stem.replace("arc_aerostruct_", "aerostruct_arc"))


def write_study_reports(out_json, seed):
    """End of every run: WingCalc + OAS reports and the comparison, per design.

    For each converged design point: rebuild it in OAS, export it to WingCalc
    exactly as arc_optimal_toc does (``aot.wingcalc_payload`` -> ``write_deck``),
    size it there, then write the OAS report and the side-by-side page into the
    same folder. One index page lists every design with the results table and links
    to all three reports, so the whole study is one file to open.

    Each design is wrapped on its own: a WingCalc sizing failure costs that design's
    WingCalc pane, never the others, and never the JSON already written.
    """
    from pathlib import Path

    from studies.vsp_planform.coupling import deck as wcdeck
    from studies.vsp_planform.viewer.compare import write_study_index
    from studies.vsp_planform.viewer.pair import write_design_reports

    with open(out_json) as f:
        cases = json.load(f)
    root = Path(results_dir(out_json))
    rows = []
    for i, r in enumerate(cases):
        slug = _case_slug(r, i)
        case_dir = root / slug
        deck_dir = root / "_decks" / f"deck_{slug}"
        print(f"\n  reports for '{r['label']}' -> {case_dir}", flush=True)
        row = {"label": r["label"], "slug": slug, "result": r, "error": None,
               "wingcalc_html": None, "oas_html": None, "compare_html": None,
               "wc_w_wing_lb": None}
        try:
            prob, _mesh, J, _ = build(seed, (r["twist_lower_deg"], config.TWIST_BOUNDS[1]),
                                      aero_dv=False, thick_dv=False)
            prob.setup()
            apply_frozen(prob, J, {"twist_cp_deg": r["twist_cp_deg"], "alpha_deg": r["alpha_deg"],
                                   "skin_cp_m": r["skin_cp_m"], "spar_cp_m": r["spar_cp_m"]})
            prob.run_model()
            place_nacelles(prob)
            prob.run_model()
            airfoil = J.get("section_blend") or J.get("airfoil")
            payload = aot.wingcalc_payload(prob, airfoil, r["label"])
            wc_html = None
            try:
                wcdeck.write_deck(wcdeck.WC_DECK, deck_dir, mission.MTOW_LB, oas=payload)
                row["wc_w_wing_lb"] = wcdeck.run_wingcalc(deck_dir, case_dir)
                found = sorted(case_dir.glob("Wing_Report*.html"))
                wc_html = found[-1] if found else None
            except Exception as exc:                     # the OAS side still stands
                row["error"] = f"WingCalc: {type(exc).__name__}: {exc}"
                print(f"  !!! {row['error']}", flush=True)
            oas_html, cmp_html, _snap = write_design_reports(
                out_json, i, seed, case_dir, wingcalc_html=wc_html,
                label=f"arc A aerostruct / {r['label']}", expect_drag_N=r["drag_N"])
            row.update(wingcalc_html=wc_html, oas_html=oas_html, compare_html=cmp_html)
        except Exception as exc:
            row["error"] = ((row["error"] + "; ") if row["error"] else "") + \
                f"OAS: {type(exc).__name__}: {exc}"
            print(f"  !!! {row['error']}", flush=True)
        rows.append(row)
    return write_study_index(root / "index.html", "Arc A aerostructural study", rows,
                             source_json=Path(out_json))


def print_table(out):
    print("\n" + "=" * 104)
    print("{:<40}{:>10}{:>11}{:>9}{:>12}{:>9}{:>7}".format(
        "case", "drag N", "W_wing lb", "R nmi", "y_cl %semi", "tw tip", "ok"))
    print("-" * 104)
    for r in out:
        print("{:<40}{:>10.1f}{:>11.1f}{:>9.2f}{:>12.3f}{:>9.2f}{:>7}".format(
            r["label"], r["drag_N"], r["W_wing_lb"], r["R_nmi"],
            100 * r["y_cl_frac_semi"], r["twist_tip_deg"], str(r["success"])))
    print("=" * 104)
    if len(out) > 1:
        b = out[0]
        print("\nagainst '{}':".format(b["label"]))
        for r in out[1:]:
            print("  {:<40} dR {:+7.2f} nmi ({:+.2f}%),  dD {:+8.1f} N,  dW {:+7.1f} lb,"
                  "  dy_cl {:+.3f} %semi".format(
                      r["label"], r["R_nmi"] - b["R_nmi"],
                      100 * (r["R_nmi"] / b["R_nmi"] - 1),
                      r["drag_N"] - b["drag_N"], r["W_wing_lb"] - b["W_wing_lb"],
                      100 * (r["y_cl_frac_semi"] - b["y_cl_frac_semi"])))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=os.path.join(LOGS, "arc_optimal_toc_A_optimal_e694.json"),
                    help="converged arc_optimal_toc JSON to take the frozen geometry from")
    ap.add_argument("--twist-floor", type=float, action="append", default=None,
                    help="absolute twist lower bound in degrees; repeatable")
    ap.add_argument("--objective", choices=("drag", "range", "both"), default="both")
    ap.add_argument("--out", default=os.path.join(LOGS, "arc_aerostruct_A_optimal_e694.json"))
    ap.add_argument("--reports-only", action="store_true",
                    help="skip the optimization; write the reports from the existing --out JSON")
    ap.add_argument("--table-only", action="store_true",
                    help="print the results table from the existing --out JSON and stop")
    ap.add_argument("--no-reports", action="store_true",
                    help="skip the WingCalc sizing and the reports at the end")
    a = ap.parse_args(argv)

    if a.table_only or a.reports_only:
        with open(a.out) as f:
            out = json.load(f)
        print_table(out)
        if a.table_only:
            return
    else:
        floors = a.twist_floor if a.twist_floor else [-1.0, -5.0]
        objs = ("drag", "range") if a.objective == "both" else (a.objective,)
        cases = [(o, f, "{} objective, twist floor {:+.1f} deg".format(o, f))
                 for f in floors for o in objs]

        print("weight model: WingCalc's build-up on the OAS box and geometry, deck "
              f"{WEIGHT_DECK.source}", flush=True)
        for name, m, y, xc, dz in NACELLES:
            print(f"  {name} nacelle {m:,.1f} lb at WS {y:.1f}, x/c {xc:.3f}, "
                  f"{dz:+.2f} in above the EA", flush=True)

        out = []
        for objective, floor, label in cases:
            print("\n" + "=" * 74 + "\n" + label + "\n" + "=" * 74, flush=True)
            out.append(run_case(objective, floor, a.seed, label))
        print_table(out)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
        print("\nwrote " + a.out)

    if not a.no_reports:
        try:
            index = write_study_reports(a.out, a.seed)
        except Exception as exc:
            print(f"\n!!! reports FAILED ({type(exc).__name__}: {exc}). The results in "
                  f"{a.out} stand; retry with --reports-only.", flush=True)
        else:
            print("\n" + "#" * 78)
            print("  RESULTS -- open this one file:")
            print(f"  {index}")
            print("#" * 78)


if __name__ == "__main__":
    main()
