"""Cost of forcing monotonic twist through region B (queued item 4 in HANDOFF.md).

The study's most persistent surprise is that the optimizer wants NON-monotonic
twist: raising the number of control points produced MORE sign changes (5 -> 8),
so it is not a spline artifact -- the VLM genuinely prefers a wavy distribution.
Tip washout is robust in every case; it is the middle of the span that oscillates.

A wavy twist is awkward to build and awkward to justify, so the question is what
monotonicity costs. This constrains the twist to be non-increasing outboard
across region B and re-optimizes in full OAS against the unconstrained wing 2
case C, changing nothing else.

Constraining `twist_abs` rather than `twist_cp` is deliberate: the B-spline of a
monotone set of control points is not itself guaranteed monotone, and it is the
physical twist distribution that has to be manufacturable.
"""

import json
import os
import sys

import numpy as np
import openmdao.api as om

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(_HERE), "..", "..", "..")))
sys.path.insert(0, os.path.dirname(_HERE))

from studies.vsp_planform import config  # noqa: E402
import studies.vsp_planform.run_opt as ro  # noqa: E402
from studies.vsp_planform.run_opt import POINT, trim_alpha  # noqa: E402

import wing2_oas as w2  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "out", "logs", "monotonic_twist.json")

q = 0.5 * config.RHO * config.V_MS**2


class TwistSlope(om.ExplicitComponent):
    """Outboard-going differences of the twist over a chosen station range."""

    def initialize(self):
        self.options.declare("ny", types=int)
        self.options.declare("idx", desc="station indices the constraint covers")

    def setup(self):
        ny = self.options["ny"]
        idx = np.asarray(self.options["idx"], dtype=int)
        self._i0, self._i1 = idx[:-1], idx[1:]
        n = self._i0.size

        self.add_input("twist_abs", val=np.zeros(ny), units="deg")
        self.add_output("dtwist", val=np.zeros(n), units="deg")

        rows = np.concatenate([np.arange(n), np.arange(n)])
        cols = np.concatenate([self._i1, self._i0])
        vals = np.concatenate([np.ones(n), -np.ones(n)])
        self.declare_partials("dtwist", "twist_abs", rows=rows, cols=cols, val=vals)

    def compute(self, inputs, outputs):
        tw = inputs["twist_abs"]
        outputs["dtwist"] = tw[self._i1] - tw[self._i0]


def build(monotonic):
    w2.apply_wing2_box()
    mesh, stick, regions, planform0 = w2.load_relofted(w2.BASELINE, w2.REGION_A_END_IN)
    y_c_in = regions.y_c_start
    def attach(model, mesh_, regions_):
        # Region B: from the A|B breakpoint out to the winglet junction.
        y_in = np.abs(mesh_[0, :, 1]) / config.SCALE
        idx = np.flatnonzero((y_in >= abs(regions_.y_a_end) - 1e-6) & (y_in <= regions_.y_c_start + 1e-6))
        model.add_subsystem(
            "twist_slope",
            TwistSlope(ny=y_in.size, idx=idx),
            promotes_inputs=["twist_abs"],
            promotes_outputs=["dtwist"],
        )
        print(f"    monotonicity applied over {idx.size} stations, y = {y_in[idx[0]]:.1f} to {y_in[idx[-1]]:.1f} in")

    prob, _ = ro.build_problem(w2.BASELINE, mesh, stick, regions, planform0, extra=attach if monotonic else None)

    prob.run_model()
    s0 = float(prob.get_val(f"{POINT}.wing.S_ref")[0])
    alpha0 = trim_alpha(prob, w2.W / (q * s0))
    if monotonic:
        # Non-increasing outboard. `ref` is a degree: the twist range is O(5 deg),
        # so this is already the natural scale.
        #
        # MUST come before add_optimization: that function ends with its own
        # prob.setup(), and a constraint added after a setup() is silently never
        # registered. A first pass added it afterwards and got a "monotonic"
        # answer identical to the free one, down to the last twist control point
        # -- which is the signature of a constraint that is not there at all,
        # not of one that does not bind.
        prob.model.add_constraint("dtwist", upper=0.0, units="deg", ref=1.0)

    ro.add_optimization(prob, "plan_l", mesh, planform0, s0, mode="fixed_lift", weight=w2.W)

    prob.set_val("alpha", alpha0, units="deg")
    prob.run_model()
    prob.run_driver()
    r = w2.evaluate(prob, y_c_in)
    r["success"] = bool(prob.driver.result.success)
    r["exit_status"] = str(prob.driver.result.exit_status)
    r["twist_abs"] = prob.get_val("twist_abs", units="deg").tolist()
    r["y_in"] = (np.abs(mesh[0, :, 1]) / config.SCALE).tolist()
    return r


def sign_changes(tw, y, y_a, y_c):
    """Number of slope sign changes inside region B."""
    m = (np.asarray(y) >= y_a - 1e-6) & (np.asarray(y) <= y_c + 1e-6)
    d = np.diff(np.asarray(tw)[m])
    s = np.sign(d[np.abs(d) > 1e-9])
    return int(np.sum(s[1:] != s[:-1]))


if __name__ == "__main__":
    print("=" * 84)
    print("Cost of monotonic (non-increasing outboard) twist through region B")
    print("=" * 84)

    res = {}
    for key, mono in (("free", False), ("monotonic", True)):
        r = build(mono)
        res[key] = r
        print(f"\n  {key}: S_ref {r['S_ref']:7.3f} m^2  CL {r['CL']:.4f}  drag {r['drag_N']:9.1f} N  [{r['exit_status']}]")
        print(f"    twist root {r['twist_root']:+.3f} -> tip {r['twist_tip']:+.3f} deg")
        print(f"    twist_cp " + "  ".join(f"{v:+.3f}" for v in r["twist_cp"]))

    y = res["free"]["y_in"]
    y_a, y_c = w2.REGION_A_END_IN, res["free"]["y_in"][-1]
    meta = json.load(open(os.path.join(os.path.dirname(OUT), "wing2_oas.json")))["_meta"]
    y_a, y_c = meta["region_a_end_in"], meta["width_stations"][-1][0]

    d = res["monotonic"]["drag_N"] - res["free"]["drag_N"]
    print("\n" + "=" * 84)
    print(f"  free       drag {res['free']['drag_N']:9.1f} N   region-B slope sign changes: "
          f"{sign_changes(res['free']['twist_abs'], y, y_a, y_c)}")
    print(f"  monotonic  drag {res['monotonic']['drag_N']:9.1f} N   region-B slope sign changes: "
          f"{sign_changes(res['monotonic']['twist_abs'], y, y_a, y_c)}")
    print(f"  COST OF MONOTONICITY: {d:+.1f} N  ({d / res['free']['drag_N']:+.3%})")
    print("=" * 84)

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n  wrote {OUT}")
