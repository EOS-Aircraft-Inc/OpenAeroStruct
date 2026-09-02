"""ONE OpenMDAO problem: drag at two operating points, under the field-length and
climb-gradient requirements, with CLmax computed from the flap planform.

WHAT IS COUPLED. All five pieces are OpenMDAO components in ONE problem:

  OpenAeroStruct    the wing, and an AeroPoint at EACH operating point
  WingCLmaxEstimateGroup   flap span and chord fractions -> flap area -> CLmax
  TOFL / LFL        Atlas's MetaModelStructuredComp tables
  climb metrics     LiveClimbComp, the REAL Atlas fixed-point WATLIM segment,
                    built once in setup() and re-run in compute() (--live-climb),
                    with finite-difference partials on (S_ref, CLmax). About
                    0.05 s per solve. The five metrics are the two WATLIM
                    gradients, the landing and approach gradients, and the AEO
                    rate of climb.

The earlier surrogate -- a MetaModelStructuredComp trained on 45 samples of the
same analysis, watlim_surrogate.py -- is kept as the default for quick runs. It
refuses to extrapolate off its trained box. With --live-climb the box is replaced
by a wide S_ref bound and the analysis is exact. Both paths converge to the same
wing to within 0.1 m2 of S_ref.

TWO OPERATING POINTS. Drag is evaluated at both, each with its own atmosphere and
its own trim angle of attack:

  cruise   25,000 ft, 260 KTAS   Mach 0.4319, rho 0.5489
  low      10,000 ft, 245 KTAS   Mach 0.3838, rho 0.9046

The objective is a WEIGHTED SUM of the two drags, --w-cruise and --w-low, because
one wing cannot be optimal at both and the split is a programme choice rather than
a result. Set a weight to zero to optimize one point and merely report the other.
Both points fly the same mesh and the same thickness distribution; only the
atmosphere and alpha differ.

WHAT IS HELD. Span at 118 ft, MTOW 86,000 lb, commanded power 1400 kW. The AEO
rate-of-climb requirement is AEO_FPM, 1300 ft/min. The climb surrogate is trained
at that power and weight and is not valid away from them.

THE APPROXIMATION WORTH KNOWING. WingCLmaxEstimateGroup builds its flap area on a
SINGLE TRAPEZOID wing, root_chord = 2*S_ref/(span*(1+taper)). These wings are
three-region planforms with a constant-chord inboard section, so the flap area, and
therefore CLmax, is an approximation. taper_B is passed as the trapezoid taper.
"""

import argparse
import contextlib
import os
import sys

import numpy as np
import openmdao.api as om

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))

from openaerostruct.aerodynamics.aero_groups import AeroPoint          # noqa: E402
from studies.vsp_planform import config                                # noqa: E402
from studies.vsp_planform.atmosphere import flight_condition           # noqa: E402
from studies.vsp_planform.param import (build_geometry_group, build_surface,  # noqa: E402
                                        mac_quarter_chord)
import wing2_oas as w2                                                 # noqa: E402
import arc_optimal_toc as A                                            # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs")
M2_FT2 = 10.7639104
LB_TO_KG = 0.45359237
SPAN_FT = 118.0

POINTS = {
    "cruise": {"ktas": 260.0, "alt_ft": 25000.0},
    "low":    {"ktas": 245.0, "alt_ft": 10000.0},
}
MTOW_LB = 86000.0
MTOP_KW = 1400.0
# AEO rate-of-climb requirement. Lowered from 1400 on 2026-09-01: at 1400 kW and
# 86,000 lb no wing area meets 1400 fpm (1314 as built, 1330 at the drag optimum),
# so the wing study holds 1300 while the power question is settled elsewhere.
AEO_FPM = 1300.0
TOFL_LIMIT_FT = 6000.0
LFL_LIMIT_FT = 6000.0


@contextlib.contextmanager
def _atlas_cwd():
    import atlas
    root = os.path.dirname(os.path.dirname(os.path.abspath(atlas.__file__)))
    prev = os.getcwd()
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(prev)


def _field_length_comps():
    """TOFL and LFL as MetaModelStructuredComp, fed by OUR CLmax.

    Atlas's TOFLLookupTable wires its own flap-schedule CLmax into the interpolator
    internally, so the group cannot accept a computed one. The interpolators are
    rebuilt here from the SAME cached training arrays that Atlas's loader prepares,
    which reuses all of its grid completion and axis padding and adds no new data.
    """
    from atlas.mission.takeoff_landing_lookup import TOFLLookupTable, LFLLookupTable
    with _atlas_cwd():
        TOFLLookupTable._load_training_data()
        LFLLookupTable._load_training_data()
    T, L = TOFLLookupTable, LFLLookupTable

    tofl = om.MetaModelStructuredComp(vec_size=1, method="slinear", extrapolate=True)
    tofl.add_input("TOW", MTOW_LB * LB_TO_KG, training_data=T._mtow_kg, units="kg")
    tofl.add_input("S_ref", 85.0, training_data=T._Sref_m2, units="m**2")
    tofl.add_input("MTOP", MTOP_KW, training_data=T._mtop_kW, units="kW")
    tofl.add_input("CLmax", float(T._clmax.max()), training_data=T._clmax)
    tofl.add_output("TOFL_ft", 4000.0, training_data=T._tofl_matrix, units="ft")

    lfl = om.MetaModelStructuredComp(vec_size=1, method="slinear", extrapolate=True)
    lfl.add_input("TOW", MTOW_LB * LB_TO_KG, training_data=L._mtow_kg, units="kg")
    lfl.add_input("S_ref", 85.0, training_data=L._Sref_m2, units="m**2")
    lfl.add_input("CLmax", float(np.median(L._clmax)), training_data=L._clmax)
    lfl.add_output("LFL_ft", 4000.0, training_data=L._lfl_matrix, units="ft")
    # The LFL CLmax axis spans 3.0144 to 3.0164 -- one landing CLmax padded to three
    # points. It is therefore NOT driven by the computed CLmax, which is a take-off
    # value in any case; it is pinned at the landing value the table carries.
    return tofl, lfl, {"tofl_clmax_axis": (float(T._clmax.min()), float(T._clmax.max())),
                       "lfl_clmax_fixed": float(np.median(L._clmax)),
                       "sref_axis": (float(T._Sref_m2.min()), float(T._Sref_m2.max()))}


def _climb_comp(npz):
    """The five climb metrics, from the surrogate trained on the real analysis."""
    d = np.load(npz, allow_pickle=True)
    c = om.MetaModelStructuredComp(vec_size=1, method="slinear", extrapolate=False)
    c.add_input("S_ref", 85.0, training_data=d["s_ref_m2"], units="m**2")
    c.add_input("CLmax", 2.78, training_data=d["clmax25"])
    labels = [str(x) for x in d["labels"]]
    targets = [float(v) for v in d["targets"]]
    names = ["watlim_2nd", "watlim_4th", "landing", "approach", "aeo"]
    for i, nm in enumerate(names):
        c.add_output(nm, float(np.nanmean(d[f"phase_{i+1}"])),
                     training_data=d[f"phase_{i+1}"])
    meta = {"names": names, "labels": labels, "targets": targets,
            "sref_axis": (float(d["s_ref_m2"].min()), float(d["s_ref_m2"].max())),
            "clmax_axis": (float(d["clmax25"].min()), float(d["clmax25"].max())),
            "power_kw": float(d["power_kw"]), "tow_lbm": float(d["tow_lbm"])}
    return c, meta


class LiveClimbComp(om.ExplicitComponent):
    """The five climb metrics from the REAL Atlas segment, BUILT ONCE and re-run.

    Atlas is built to run a standalone segment repeatedly, and this uses it that
    way. Two earlier versions of this file did not:

      a surrogate            45 samples, justified by an assertion that the direct
                             model was too slow. It was never timed.
      a black-box wrapper    called run_fixed_point_watlim per evaluation, which
                             reloads the Excel and re-runs setup every time.

    Timed: the whole black-box call is 1.155 s, prepare_analysis inside it is
    0.008 s, and run_model on the ALREADY BUILT problem is 0.0488 s. Rebuilding cost
    24x, and all of it was the Excel and setup. The segment is therefore built once
    in setup() and only re-run in compute().

    The wing and the speed schedule reach it through set_val on IVC outputs that the
    built problem already exposes -- S_plan, S_trap, span_plan, span_trap, taper and
    seg|fltcond|Ueas -- so nothing is patched and nothing is suppressed.

    Partials are finite-differenced because the segment returns floats. With two
    inputs that is three solves, about 0.15 s per gradient.
    """

    def initialize(self):
        self.options.declare("power_kw", default=MTOP_KW)
        self.options.declare("tow_lbm", default=MTOW_LB)
        self.options.declare("span_ft", default=SPAN_FT)
        self.options.declare("altitude_ft", default=2000.0)
        self.options.declare("disa_degC", default=20.0)
        self.options.declare("aeo_fpm", default=AEO_FPM)

    def setup(self):
        self._build_once()
        self.add_input("S_ref", val=80.0, units="m**2")
        self.add_input("CLmax", val=2.78)
        self._names = ("watlim_2nd", "watlim_4th", "landing", "approach", "aeo")
        for nm in self._names:
            self.add_output(nm, val=3.0)
        self.declare_partials("*", "S_ref", method="fd", step=0.5)
        self.declare_partials("*", "CLmax", method="fd", step=0.02)
        self.n_solves = 0

    def _build_once(self):
        """Build the Atlas segment a single time and keep it."""
        from atlas.scenarios.runs.emotor_sizing import run_emotor_sizing_fixed_point as FP
        o = self.options
        cap = {}
        orig_prep = FP.prepare_analysis
        orig_sched = FP.build_watlim_speed_schedule_mps

        def spy(*a, **kw):
            out = orig_prep(*a, **kw)
            cap["prob"] = out[0]
            return out

        def sched_spy(mass_kg, s_ref, flap, aeo_speed_kias=190.0):
            r = orig_sched(mass_kg, s_ref, flap, aeo_speed_kias=aeo_speed_kias)
            cap["sched0"] = dict(r)
            cap["s_ref0"] = float(s_ref)
            return r

        FP.prepare_analysis, FP.build_watlim_speed_schedule_mps = spy, sched_spy
        try:
            with _atlas_cwd():
                FP.run_fixed_point_watlim(
                    altitude_ft=o["altitude_ft"], disa_degC=o["disa_degC"],
                    mode="fixed_power", fixed_power_hp=o["power_kw"] * 1.341,
                    tow_lbm_override=o["tow_lbm"],
                    bypass_motor_shaft_power_limit=True,
                    aeo_climb_rate_target_fpm=o["aeo_fpm"])
        finally:
            FP.prepare_analysis = orig_prep
            FP.build_watlim_speed_schedule_mps = orig_sched
        self._FP = FP
        self._prob = cap["prob"]
        self._sched0 = cap["sched0"]
        self._s_ref0 = cap["s_ref0"]
        self._reqs = FP._phase_requirements(o["aeo_fpm"])
        self._w = self._prob.model.get_val("ac|geom|wing|S_plan", units="m**2")
        self._wing0 = {k: float(self._prob.get_val(f"ac|geom|wing|{k}",
                                                   units=("m**2" if k.startswith("S") else "m"))[0])
                       for k in ("S_plan", "S_trap", "span_plan", "span_trap")}
        self._phase_ids = None

    def compute(self, inputs, outputs):
        o = self.options
        s_ref = float(inputs["S_ref"][0])
        clmax = float(inputs["CLmax"][0])
        p = self._prob

        # The wing: scale every area and every span, as watlim_area_bound does, so
        # span_trap keeps its own value rather than being set equal to span.
        span_m = o["span_ft"] * 0.3048
        ks = s_ref / self._wing0["S_plan"]
        kb = span_m / self._wing0["span_plan"]
        for k in ("S_plan", "S_trap"):
            p.set_val(f"ac|geom|wing|{k}", self._wing0[k] * ks, units="m**2")
        for k in ("span_plan", "span_trap"):
            p.set_val(f"ac|geom|wing|{k}", self._wing0[k] * kb, units="m")

        # The speed schedule follows S_ref and CLmax, so it is recomputed and set.
        mass_kg = o["tow_lbm"] / 2.20462
        k = clmax / 2.7836
        vs = lambda cl: self._FP._stall_speed_mps(cl * k, mass_kg, s_ref)
        sched = {"v_watlim_2nd_mps": 1.13 * vs(2.7836),
                 "v_watlim_4th_mps": 1.18 * vs(1.4651),
                 "v_approach_mps": 1.13 * vs(2.7836),
                 "v_ldg_mps": 1.23 * vs(2.9591),
                 "v_aeo_mps": 190.0 * 0.514}
        sw = self._FP.build_fixed_point_sweep(
            o["altitude_ft"], o["disa_degC"],
            np.asarray([o["power_kw"] * 1.341]), 25, sched)
        p.set_val("seg|fltcond|Ueas", np.asarray(sw["speed_sweep_vals"]).ravel(),
                  units="m/s")
        self._phase_ids = np.asarray(sw["phase_id_vals"])

        # The cwd is needed at COMPUTE time, not only at build time: some Atlas
        # components read their empirical CSVs lazily on the first solve, by a path
        # relative to the repo root (motor_empirical_mm.py, for one).
        with _atlas_cwd():
            p.run_model()
        self.n_solves += 1

        v_s = p.get_val("seg.fltcond|vs", units="m/s").flatten()
        utrue = p.get_val("seg.fltcond|Utrue", units="m/s").flatten()
        alt = p.get_val("seg.fltcond|h", units="ft").flatten()
        disa = p.get_val("seg.fltcond|TempIncrement", units="degC").flatten()
        grad = np.tan(np.arcsin(np.clip(v_s / utrue, -1.0, 1.0))) * 100.0
        roc = v_s * 60.0 / 0.3048
        res = self._FP._phase_metrics_at_power(
            self._phase_ids, alt, disa, grad, roc,
            o["altitude_ft"], o["disa_degC"], self._reqs)
        for r, nm in zip(res, self._names):
            outputs[nm] = (r.metric_value if r.metric_value is not None else -1e3)


def build(case, npz, flap_span=0.70, flap_chord=0.35, flap_angle=25.0, live_climb=False,
          aeo_fpm=AEO_FPM):
    """Assemble the whole thing. ``case`` is a design point dict."""
    from atlas.aerodynamics.CL_max_est import WingCLmaxEstimateGroup

    y_a = A.REGION_A_AS_BUILT_IN
    import studies.vsp_planform.param as param
    saved = param.REGION_A_RULE[w2.BASELINE]
    param.REGION_A_RULE[w2.BASELINE] = case.get("region_a_rule") or "root_le_fixed"
    try:
        mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, y_a)
        surface = build_surface(mesh, stick, regions)
        # THE SECTION BLEND MUST TRAVEL WITH THE DESIGN. A design point built on a
        # spanwise section carries a per-panel c_max_t, and c_max_t sets the Raymer
        # form factor. Building the surface without it gives the wing the AS-BUILT
        # section's form factor instead: measured here as 11112.6 N against Arc A's
        # own 10836.4 N, a 276 N error, and compare_classes.replay records the same
        # failure as 252 N on Arc C. Every drag number in this problem depends on it.
        blend = case.get("section_blend")
        if blend:
            _, cmt_at, _ = A.blended_section(blend["inboard"], blend["outboard"],
                                             blend["f_start"], blend["f_end"])
            ym = np.abs(np.asarray(mesh)[0, :, 1]) / config.SCALE
            yp = 0.5 * (ym[:-1] + ym[1:])
            surface["c_max_t"] = np.array([cmt_at(v) for v in yp])
        elif case.get("c_max_t") is not None:
            surface["c_max_t"] = float(case["c_max_t"])
    finally:
        param.REGION_A_RULE[w2.BASELINE] = saved

    prob = om.Problem(reports=False)
    model = prob.model
    cg, mac, y_mac = mac_quarter_chord(mesh)

    model.add_subsystem("wing", build_geometry_group(surface, regions, planform0),
                        promotes_outputs=["sweep_B", "station_chord", "wingbox_width",
                                          "twist_abs"])

    # ---- one AeroPoint per operating point, each with its own atmosphere and alpha
    for tag, spec in POINTS.items():
        fc = flight_condition(spec["ktas"], spec["alt_ft"])
        ivc = om.IndepVarComp()
        ivc.add_output("v", val=fc["v"], units="m/s")
        ivc.add_output("alpha", val=1.0, units="deg")
        ivc.add_output("Mach_number", val=fc["Mach_number"])
        ivc.add_output("re", val=fc["re"], units="1/m")
        ivc.add_output("rho", val=fc["rho"], units="kg/m**3")
        ivc.add_output("cg", val=cg, units="m")
        model.add_subsystem(f"vars_{tag}", ivc)
        model.add_subsystem(f"pt_{tag}", AeroPoint(surfaces=[surface]))
        for v in ("v", "alpha", "Mach_number", "re", "rho", "cg"):
            model.connect(f"vars_{tag}.{v}", f"pt_{tag}.{v}")
        model.connect("wing.mesh", f"pt_{tag}.wing.def_mesh")
        model.connect("wing.mesh", f"pt_{tag}.aero_states.wing_def_mesh")
        model.connect("wing.t_over_c", f"pt_{tag}.wing_perf.t_over_c")
        model.add_subsystem(f"forces_{tag}", om.ExecComp(
            ["lift = 0.5 * rho * v**2 * S_ref * CL",
             "drag = 0.5 * rho * v**2 * S_ref * CD"],
            lift={"units": "N"}, drag={"units": "N"},
            rho={"units": "kg/m**3", "val": fc["rho"]},
            v={"units": "m/s", "val": fc["v"]},
            S_ref={"units": "m**2", "val": 85.0}, CL={"val": 0.9}, CD={"val": 0.03}))
        model.connect(f"pt_{tag}.wing.S_ref", f"forces_{tag}.S_ref")
        model.connect(f"pt_{tag}.wing_perf.CL", f"forces_{tag}.CL")
        model.connect(f"pt_{tag}.wing_perf.CD", f"forces_{tag}.CD")

    # ---- CLmax from the flap planform -------------------------------------
    semi_m = 0.5 * SPAN_FT * 0.3048
    fivc = om.IndepVarComp()
    fivc.add_output("ac|geom|wing|span", SPAN_FT * 0.3048, units="m")
    fivc.add_output("ac|geom|wing|taper", float(case.get("taper_B", 0.44)))
    fivc.add_output("ac|geom|wing|y_inbd_flp_inbd", 0.10 * semi_m, units="m")
    fivc.add_output("ac|geom|wing|y_inbd_flp_outbd", 0.38 * semi_m, units="m")
    fivc.add_output("ac|geom|wing|y_outbd_flp_inbd", 0.41 * semi_m, units="m")
    fivc.add_output("ac|geom|wing|outbd_span_ratio", flap_span)
    fivc.add_output("flap_chord_frac", flap_chord)
    fivc.add_output("toverc", float(case.get("toc_root", 0.24)) * 100.0)
    fivc.add_output("flap_angle", flap_angle, units="deg")
    fivc.add_output("cl_max_clean", 2.05)
    fivc.add_output("delta_y", 5.1)
    fivc.add_output("mach", 0.2)
    fivc.add_output("delta_CLmax_s", 0.0)
    model.add_subsystem("flap_vars", fivc, promotes=["*"])
    model.add_subsystem("clmax", WingCLmaxEstimateGroup(), promotes=["*"])
    model.connect("pt_cruise.wing.S_ref", "ac|geom|wing|S_ref")

    # ---- field length and climb -------------------------------------------
    tofl, lfl, fmeta = _field_length_comps()
    model.add_subsystem("tofl", tofl)
    model.add_subsystem("lfl", lfl)
    model.connect("pt_cruise.wing.S_ref", ["tofl.S_ref", "lfl.S_ref"])
    model.connect("CL_max", "tofl.CLmax")
    if live_climb:
        climb = LiveClimbComp(aeo_fpm=aeo_fpm)
        model.add_subsystem("climb", climb)
        # The real analysis has no trained box, so no S_ref bound is imposed by it.
        cmeta = {"names": list(climb._names) if hasattr(climb, "_names")
                 else ["watlim_2nd", "watlim_4th", "landing", "approach", "aeo"],
                 "labels": ["WATLIM 2nd Segment", "WATLIM 4th Segment", "Landing",
                            "Approach", "AEO"],
                 "targets": [3.0, 1.7, 3.2, 2.7, aeo_fpm],
                 "sref_axis": (40.0, 140.0), "clmax_axis": (2.0, 3.4),
                 "power_kw": MTOP_KW, "tow_lbm": MTOW_LB, "live": True}
    else:
        climb, cmeta = _climb_comp(npz)
        model.add_subsystem("climb", climb)
        # The npz stores the requirement it was trained under; the metric itself
        # does not depend on the requirement, so the target is overridden here.
        cmeta["targets"][cmeta["names"].index("aeo")] = aeo_fpm
        cmeta["live"] = False
    model.connect("pt_cruise.wing.S_ref", "climb.S_ref")
    model.connect("CL_max", "climb.CLmax")
    return prob, surface, planform0, {"field": fmeta, "climb": cmeta}, mesh


def add_optimization(prob, meta, mesh, w_cruise=1.0, w_low=0.0,
                     weight_n=None, flap_dv=False, width_min_in=None,
                     drop_aeo=False, pct_dv=False):
    """Objective, trim, design variables and constraints.

    EACH POINT IS TRIMMED SEPARATELY. alpha is a design variable per point and the
    lift at that point is an equality constraint, so both operating points carry the
    same aircraft weight at their own atmosphere. Without this the two drags are
    read at whatever alpha the model happens to hold, which is not a comparison.
    """
    from studies.vsp_planform.param import twist_cp_bounds
    m = prob.model
    w_n = float(weight_n if weight_n is not None else w2.W)

    m.add_subsystem("obj", om.ExecComp(
        "J = wc*dc + wl*dl", J={"units": "N"}, dc={"units": "N"}, dl={"units": "N"},
        wc={"val": w_cruise}, wl={"val": w_low}), promotes_outputs=["J"])
    m.connect("forces_cruise.drag", "obj.dc")
    m.connect("forces_low.drag", "obj.dl")
    m.add_objective("J", ref=1e4)

    tw_lower, tw_upper = twist_cp_bounds(mesh, config.N_TWIST_CP)
    m.add_design_var("wing.twist_cp", lower=tw_lower, upper=tw_upper, units="deg")
    m.add_design_var("wing.taper_B", lower=config.TAPER_B_BOUNDS[0],
                     upper=config.TAPER_B_BOUNDS[1])
    if pct_dv:
        # run_opt.py records why this is off by default: under a rule that pins the
        # straight line, wingbox_pct is a FIXED quantity and offering it to SLSQP as
        # a design variable stalls the driver on a degenerate direction.
        m.add_design_var("wing.wingbox_pct", lower=config.WINGBOX_CHORD_PCT_BOUNDS[0],
                         upper=config.WINGBOX_CHORD_PCT_BOUNDS[1])
    for tag in POINTS:
        m.add_design_var(f"vars_{tag}.alpha", lower=-5.0, upper=12.0, units="deg")
        m.add_constraint(f"forces_{tag}.lift", equals=w_n, ref=w_n)
    if flap_dv:
        # The flap is a real lever on CLmax and therefore on TOFL and the climb
        # margins. It is off by default because it changes the aircraft, not the wing.
        m.add_design_var("ac|geom|wing|outbd_span_ratio", lower=0.55, upper=0.80)
        m.add_design_var("flap_chord_frac", lower=0.25, upper=0.40)

    if width_min_in is not None:
        m.add_constraint("wingbox_width", lower=np.asarray(width_min_in) * config.SCALE,
                         units="m", ref=float(np.mean(width_min_in)) * config.SCALE)

    m.add_constraint("tofl.TOFL_ft", upper=TOFL_LIMIT_FT, ref=1e3)
    m.add_constraint("lfl.LFL_ft", upper=LFL_LIMIT_FT, ref=1e3)
    # The surrogate REFUSES to extrapolate, so the optimizer has to be told where it
    # may go. Outside this box the climb outputs are not defined.
    lo, hi = meta["climb"]["sref_axis"]
    m.add_constraint("pt_cruise.wing.S_ref", lower=lo, upper=hi, units="m**2", ref=hi)
    for nm, tgt in zip(meta["climb"]["names"], meta["climb"]["targets"]):
        if drop_aeo and nm == "aeo":
            # At 1400 fpm AEO was not met at ANY area at this power and weight --
            # 1373 fpm at the smallest area in the training grid. Leaving it in made
            # the problem infeasible by construction. With AEO_FPM at 1300 it is
            # feasible; the switch stays for studies that want it reported only.
            continue
        m.add_constraint(f"climb.{nm}", lower=tgt, ref=abs(tgt) or 1.0)
    return prob


def pretrim(prob, weight_n=None, tol=1e-8, max_iter=25):
    """Put each point on its lift equality BEFORE the driver starts.

    SLSQP starts from alpha = 1 deg at both points, where the low point carries 560
    kN against a 383 kN target. Handing it that costs iterations on a constraint a
    secant solves in seconds, and a 25-iteration run ended worse than the design
    point it started from. Each point is trimmed independently, because they differ
    only in atmosphere and each must carry the same weight.
    """
    w_n = float(weight_n if weight_n is not None else w2.W)
    for tag in POINTS:
        a0 = float(prob.get_val(f"vars_{tag}.alpha", units="deg")[0])

        def lift_at(al):
            prob.set_val(f"vars_{tag}.alpha", al, units="deg")
            prob.run_model()
            return float(prob.get_val(f"forces_{tag}.lift")[0]) - w_n

        f0 = lift_at(a0)
        a1 = a0 + 1.0
        f1 = lift_at(a1)
        for _ in range(max_iter):
            if abs(f1) < tol * w_n or abs(f1 - f0) < 1e-12:
                break
            a2 = a1 - f1 * (a1 - a0) / (f1 - f0)
            a0, f0 = a1, f1
            a1 = float(np.clip(a2, -5.0, 12.0))
            f1 = lift_at(a1)
        print(f"  pre-trimmed {tag:6s} alpha {a1:7.4f} deg, lift residual {f1:+.2f} N")
    return prob


def report(prob, meta, head=""):
    """Every quantity the problem constrains, with its margin."""
    S = float(prob.get_val("pt_cruise.wing.S_ref")[0])
    if head:
        print(head)
    print(f"  S_ref {S:7.3f} m2 = {S*M2_FT2:7.1f} ft2   "
          f"CL_max {float(prob.get_val('CL_max')[0]):.4f}   "
          f"S_wf {float(prob.get_val('S_wf', units='m**2')[0]):.2f} m2   "
          f"taper_B {float(prob.get_val('wing.taper_B')[0]):.4f}")
    tot = 0.0
    for tag in POINTS:
        d = float(prob.get_val(f"forces_{tag}.drag")[0])
        tot += d
        print(f"  {tag:6s} alpha {float(prob.get_val(f'vars_{tag}.alpha', units='deg')[0]):6.3f} deg"
              f"  CL {float(prob.get_val(f'pt_{tag}.wing_perf.CL')[0]):.4f}"
              f"  lift {float(prob.get_val(f'forces_{tag}.lift')[0]):9.1f} N"
              f"  drag {d:9.1f} N")
    for nm, lim, sense in (("tofl.TOFL_ft", TOFL_LIMIT_FT, "<="),
                           ("lfl.LFL_ft", LFL_LIMIT_FT, "<=")):
        v = float(prob.get_val(nm)[0])
        print(f"  {nm.split('.')[1]:10s} {v:9.1f} ft {sense} {lim:.0f}   "
              f"margin {lim-v:+9.1f}   {'PASS' if v <= lim else 'FAIL'}")
    for nm, tgt, lab in zip(meta["climb"]["names"], meta["climb"]["targets"],
                            meta["climb"]["labels"]):
        v = float(prob.get_val(f"climb.{nm}")[0])
        print(f"  {lab[:20]:20s} {v:9.2f} >= {tgt:8.2f}   margin {v-tgt:+9.2f}   "
              f"{'PASS' if v >= tgt else 'FAIL'}")
    return tot


if __name__ == "__main__":
    import json
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", default="arc_a_constfrac_optimal_e694.json")
    ap.add_argument("--surrogate", default="watlim_surrogate_1400kw_86000lbm.npz")
    ap.add_argument("--w-cruise", type=float, default=1.0)
    ap.add_argument("--w-low", type=float, default=0.0)
    ap.add_argument("--flap-dv", action="store_true", help="let the flap geometry move")
    ap.add_argument("--drop-aeo", action="store_true",
                    help="report AEO but do not constrain it")
    ap.add_argument("--pct-dv", action="store_true", help="let wingbox_pct move")
    ap.add_argument("--live-climb", action="store_true",
                    help="run the REAL Atlas climb analysis every evaluation")
    ap.add_argument("--aeo-fpm", type=float, default=AEO_FPM,
                    help=f"AEO rate-of-climb requirement, ft/min (default {AEO_FPM:.0f})")
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--maxiter", type=int, default=40)
    a = ap.parse_args()

    case = json.load(open(os.path.join(LOGS, a.case)))
    npz = os.path.join(LOGS, a.surrogate)
    prob, surface, planform0, meta, mesh_out = build(case, npz, live_climb=a.live_climb, aeo_fpm=a.aeo_fpm)
    add_optimization(prob, meta, mesh_out, a.w_cruise, a.w_low,
                     flap_dv=a.flap_dv, drop_aeo=a.drop_aeo, pct_dv=a.pct_dv)
    if a.optimize:
        prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", maxiter=a.maxiter,
                                             tol=1e-6)
        prob.driver.options["debug_print"] = ["nl_cons", "objs"]
    prob.setup()
    prob.set_val("wing.taper_B", case["taper_B"])
    prob.set_val("wing.wingbox_pct", case["wingbox_pct"])
    prob.set_val("wing.twist_cp", np.array(case["twist_cp"]), units="deg")
    if case.get("t_over_c_cp") is not None:
        prob.set_val("wing.t_over_c_cp", np.array(case["t_over_c_cp"]))
    prob.run_model()
    pretrim(prob)

    print(f"objective weights: cruise {a.w_cruise:.2f}, low {a.w_low:.2f}   "
          f"trim weight {w2.W:,.0f} N at BOTH points")
    print(f"climb model: {'LIVE Atlas analysis every evaluation' if meta['climb'].get('live') else 'surrogate'}")
    print(f"climb setting: {meta['climb']['power_kw']:.0f} kW, "
          f"{meta['climb']['tow_lbm']:,.0f} lbm, S_ref "
          f"{meta['climb']['sref_axis'][0]:.0f}-{meta['climb']['sref_axis'][1]:.0f} m2, "
          f"CLmax {meta['climb']['clmax_axis'][0]:.2f}-{meta['climb']['clmax_axis'][1]:.2f}")
    report(prob, meta, f"\nAS BUILT ({a.case}, both points trimmed):")

    if a.optimize:
        prob.run_driver()
        report(prob, meta, "\nOPTIMIZED:")
        print(f"  driver success: {prob.driver.result.success}")
