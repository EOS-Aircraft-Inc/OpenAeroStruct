"""The weight build-up as an OpenMDAO component, fed straight from the wingbox FEM.

Every input is the FEM's own, in OAS's MIRRORED structural convention (TIP -> ROOT,
y <= 0; see ``arc_aerostruct.MirrorSpanwise``), so the component connects to the
FEM without another reversal and does the reflection back to root -> tip inside.

PARTIALS, AND WHY THEY ARE SPLIT
--------------------------------
The sized structure reaches the build-up in exactly three places -- the box
weight, the rib cell area ``Am``, and the splice-bay running weight -- and all
three enter LINEARLY: every one of them lands in ``W_wingbox`` ahead of the lay-up
allowance, so ``d W_wing`` is a fixed multiple of each (``buildup.dW_wing_dW_box``).
Those partials are analytic and exact, which matters because they carry the whole
weight-vs-thickness trade the optimizer is making.

Geometry -- the LE/TE lines and t/c -- is differenced. It moves only through twist
here (a rotation about the quarter chord: the chord LENGTH, which the areas are
built on, does not change), so these partials are small, but they are real: the
nacelle fittings and the CG sit on the rotated section. Two terms are rounded to
whole plies or fastener rows exactly as WingCalc rounds them, so they are piecewise
constant in the geometry and their difference is zero except across a step.
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om

from studies.vsp_planform.weight import buildup as bu

_IN_M = 0.0254
_LB_KG = 0.45359237


def fem_origin(surface):
    """OAS's FEM axis, as a chord fraction (structures/compute_nodes.py)."""
    xu, yu, yl = surface["data_x_upper"], surface["data_y_upper"], surface["data_y_lower"]
    h0, h1 = float((yu[0] - yl[0]).real), float((yu[-1] - yl[-1]).real)
    return float((xu[0].real * h0 + xu[-1].real * h1) / (h0 + h1))


class WingWeightBuildUp(om.ExplicitComponent):
    """WingCalc's wing weight, lb, from the OAS box and the OAS geometry.

    ``W_wing_lb`` is the full build-up; ``W_box_lb`` is the sized box alone, both
    wings, inboard of the wingbox tip (the winglet's own box is not counted: WingCalc
    excludes winglets from the box and weighs them as ``W_wingtip``). The whole
    breakdown of the last evaluation is kept on ``self.last`` for reporting.
    """

    def initialize(self):
        self.options.declare("surface", types=dict, desc="the FEM's (mirrored) surface")
        self.options.declare("deck", desc="weight.deck_inputs.WeightDeck")
        self.options.declare("aft_at", desc="aft spar chord ratio as a function of WS, in")
        self.options.declare("fwd_ratio", types=float)
        self.options.declare("profile_at", desc="WS -> (camber, thickness), normalized")
        self.options.declare("x_grid", types=np.ndarray)

    def setup(self):
        s = self.options["surface"]
        deck = self.options["deck"]
        nx, ny = s["mesh"].shape[:2]
        self._ny = ny
        self._w_fem = fem_origin(s)

        # Flat, because OpenMDAO connects it through flat src_indices off the mesh and
        # requires the indexed shape to be the input's own shape.
        self.add_input("le_te", shape=2 * ny * 3, units="m",
                       desc="rows 0 and -1 of the mirrored mesh, LE then TE, flattened")
        self.add_input("t_over_c", shape=ny - 1, desc="mirrored, per panel")
        self.add_input("element_mass", shape=ny - 1, units="kg", desc="mirrored, half wing")
        self.add_input("A_enc", shape=ny - 1, units="m**2", desc="mirrored mid-line cell area")
        self.add_input("A", shape=ny - 1, units="m**2", desc="mirrored box laminate area")
        self.add_output("W_wing_lb", val=6800.0)
        self.add_output("W_box_lb", val=3400.0)

        # ---- fixed for the life of the problem: the spanwise stations. Span, the
        #      rib rules and the element y positions do not move with any design
        #      variable here, so everything keyed on WS is built once.
        y_rt = -np.asarray(s["mesh"][0, ::-1, 1]).real / _IN_M          # root -> tip
        self._ws_node = y_rt
        ws_mid = 0.5 * (y_rt[:-1] + y_rt[1:])
        self._ws_rib = np.array(bu.rib_stations(deck))
        tip = deck.half_wingbox_span
        # The box runs to the last node at or inboard of the wingbox tip. One element
        # past it is already the winglet (nearest node is 679.7 in on Arc A).
        self._n_box = int(np.searchsorted(y_rt, tip + 1e-6, side="right")) - 1
        self._profiles = {float(w): self.options["profile_at"](float(w)) for w in self._ws_rib}

        # Linear maps from the per-element FEM arrays (root -> tip) to the ribs and to
        # the splice station. Clamped at the ends, as np.interp is.
        def interp_matrix(targets):
            m = np.zeros((len(targets), len(ws_mid)))
            for i, t in enumerate(targets):
                e = np.zeros(len(ws_mid))
                for j in range(len(ws_mid)):
                    e[:] = 0.0
                    e[j] = 1.0
                    m[i, j] = np.interp(t, ws_mid, e)
            return m

        self._m_rib = interp_matrix(self._ws_rib)
        self._m_splice = (interp_matrix([self._ws_rib[deck.splice_bay - 1]])[0]
                          if deck.splice_bay > 0 else np.zeros(len(ws_mid)))
        self._rho_box = deck.materials[deck.skin_material].density   # lb/in3

        # d W_wing / d Am at each rib: the rib plate itself, less what the nacelle
        # reinforcement subtracts for the plain rib it replaces (they cancel there).
        t = np.array([bu.SMEARED_THICKNESS_IN[bu.classify(w, deck)] for w in self._ws_rib])
        rho_rib = deck.materials[bu.RIB_MATERIAL].density
        mult = np.where(np.arange(len(t)) == 0, 1.0, 2.0)
        nac = np.isin([bu.classify(w, deck) for w in self._ws_rib], ("NACELLE_IB", "NACELLE_OB"))
        self._dW_dAm = bu.dW_wing_dW_box() * t * rho_rib * (mult - 2.0 * nac)
        self._dW_dsplice = (bu.dW_wing_dW_box() * max(deck.wing_splices, 0)
                            * max(deck.splice_length, 0.0))

        self.declare_partials("W_wing_lb", ["le_te", "t_over_c"], method="fd", step=1e-7)
        self.declare_partials("W_wing_lb", ["element_mass", "A_enc", "A"])
        self.declare_partials("W_box_lb", "element_mass")

    # -- the mirrored FEM arrays, back in root -> tip ------------------------------
    def _rt(self, v):
        return np.asarray(v)[::-1]

    def evaluate(self, inputs):
        o = self.options
        le_te = np.asarray(inputs["le_te"]).reshape(2, self._ny, 3)
        le = le_te[0, ::-1, :] / _IN_M
        te = le_te[1, ::-1, :] / _IN_M
        le, te = le.copy(), te.copy()
        le[:, 1] *= -1.0
        te[:, 1] *= -1.0
        g = bu.rib_geometry_from_oas(
            o["deck"], self._ws_node, le, te, self._rt(inputs["t_over_c"]),
            o["aft_at"], o["fwd_ratio"], lambda w: self._profiles[float(w)],
            o["x_grid"], self._w_fem)
        em = self._rt(inputs["element_mass"]) / _LB_KG
        W_box = 2.0 * float(np.sum(em[:self._n_box]))
        am = self._m_rib @ (self._rt(inputs["A_enc"]) / _IN_M ** 2)
        splice = float(self._m_splice @ (self._rt(inputs["A"]) / _IN_M ** 2)) * self._rho_box
        return bu.wing_weight(g, o["deck"], W_box, am, splice), g

    def compute(self, inputs, outputs):
        r, g = self.evaluate(inputs)
        outputs["W_wing_lb"] = r["W_wing"]
        outputs["W_box_lb"] = r["W_bays_full"]
        self.last, self.last_geometry = r, g

    def compute_partials(self, inputs, partials):
        k = bu.dW_wing_dW_box()
        n = self._ny - 1
        d_em = np.zeros(n)
        d_em[:self._n_box] = 2.0 / _LB_KG                  # root -> tip
        partials["W_box_lb", "element_mass"] = d_em[::-1][None, :]
        partials["W_wing_lb", "element_mass"] = (k * d_em)[::-1][None, :]
        d_aenc = (self._dW_dAm @ self._m_rib) / _IN_M ** 2
        partials["W_wing_lb", "A_enc"] = d_aenc[::-1][None, :]
        d_a = self._dW_dsplice * self._rho_box * self._m_splice / _IN_M ** 2
        partials["W_wing_lb", "A"] = d_a[::-1][None, :]
