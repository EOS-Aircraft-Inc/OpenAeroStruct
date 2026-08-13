# VSP → OpenAeroStruct planform study

Reads OpenVSP DegenGeom CSV exports, rebuilds the wings inside OpenAeroStruct
under a structurally-meaningful parameterization, verifies the rebuild matches
the original, and optimizes for cruise drag. A separate, deliberately
unconnected AeroSandbox DOE explores airfoil sections.

## The two baselines

Both are symmetric wings exported as two halves (`SurfNdx` 0/1), in **inches**
(`SCALE = 0.0254`), each with three spanwise regions:

| | Plan_L | Plan_L_ConstChord |
|---|---|---|
| Plate grid | 27 sections × 29 camber points | 95 × 29 |
| Half-span | 694.11 in (17.63 m) | 708.00 in (17.98 m) |
| **A** constant chord | y 0→50 in (7.2%), c = 138.90 | y 0→361.70 (51.1%), c = 105.0 |
| **B** tapered | y 50→646, c → 54.23 | y 361.70→674.95, c → 40.0 |
| **C** winglet | y 662→694, c → 20.0 | y 674.95→708, c → 14.0 |
| Twist | 2.12° → 0.13° | 4.00° → −1.00° |
| t/c | 0.1774 constant | 0.178 → 0.100 |
| t/c location | 0.374 | 0.299 |
| Region indices (A end, C start) | (2, 22) | (51, 90) |

Span ≈ 36 m, area ≈ 80 m², AR ≈ 16.

## The straight spar, and why sweep is not a design variable

Both wings are built around a **straight, unswept aft wingbox spar** running from
the root to the start of the winglet. Least-squares fitting the chord fraction
`p` that makes `x_spar = x_LE + p·(x_TE − x_LE)` constant over regions A+B:

| | p | x_spar | max deviation |
|---|---|---|---|
| Plan_L | 0.60065 | 1018.09 in | 0.065 in (0.047% of root chord) |
| ConstChord | 0.68087 | 1006.12 in | 0.090 in (0.086% of root chord) |

With the spar at constant *x*, the leading edge is pinned to the chord
distribution: `x_LE(y) = x_spar − p·c(y)`. In region B that makes sweep a
*consequence*, not a choice:

```
tan(Λ_LE,B) = p · c_A · (1 − λ) / span_B
```

| | predicted | actual |
|---|---|---|
| Plan_L | 4.879° | 4.872° |
| ConstChord | 8.043° | 8.023° |

Both match to 0.02°, so the VSP models were built to exactly this rule. The
parameterization therefore enforces the straight spar **by construction** rather
than by constraint — cheaper and better conditioned — and reports sweep as a
derived output.

## Design variables and constraints

| DV | Bounds | Notes |
|---|---|---|
| `wingbox_pct` (p) | 0.45 – 0.75 | rear-spar chord fraction; drives region B sweep |
| `taper_B` | 0.15 – 1.0 | tip chord of B ÷ chord at the A\|B junction |
| `twist_cp` (5 pts) | −1° to +5° | **absolute**, not relative |
| `alpha` | — | trim |

Fixed by requirement: dihedral (baseline z carried through verbatim), semi-span,
and the winglet's own shape — region C rides rigidly with the moving junction and
scales with B's tip chord.

Per-geometry rules, which differ:

- **ConstChord**: region A's chord is preserved at 105.0 in exactly.
- **Plan_L**: wingbox width `p·c(y)` ≥ 65 in out to y = 100 in. Binding at
  y = 100, where the baseline sits at 79.2 in.

## Flight condition

ISA at **260 KTAS, 25,000 ft**, computed in `atmosphere.py` rather than hardcoded:

```
T = 238.62 K      p = 37600.9 Pa    rho = 0.548946 kg/m^3
a = 309.669 m/s   mu = 1.53974e-5   V = 133.755 m/s
M = 0.431930      Re = 4.76863e6 per metre
```

## Meshing

Native camber meshes are (29, 27, 3) and (29, 95, 3) per half. Both are resampled
onto a common grid — 35 spanwise (cosine-clustered, 20% of stations inside the
winglet) × 9 chordwise — so the two cases are comparable and the VLM stays
affordable: **952 → 272 panels per half, ~12× off the AIC work.**

Resampling is PCHIP (shape-preserving, no overshoot at the winglet knee) in
cumulative leading-edge arc length spanwise and normalized camber-line arc length
chordwise. Plain *y* compresses badly through the 45° winglet, and plain *x*
under-resolves the LE on cambered sections. Residuals at those settings:

| | spanwise max | chordwise max |
|---|---|---|
| Plan_L | 11.4 mm | 1.18 mm (0.042% chord) |
| ConstChord | 19.3 mm | 5.53 mm (0.272% chord) |

on an ~18 m half-span. The two are reported separately, never summed.

### Is nx = 9 enough?

Geometric error is the wrong question — a coarser chordwise mesh changes the VLM
answer even when it loses no geometry. Measured **at trimmed CL = 0.5**, which is
how every number in this study is produced:

| nx | panels/half | Plan_L CD | ΔCD | ConstChord CD | ΔCD | solve |
|---|---|---|---|---|---|---|
| **9** | 272 | 0.0142054 | −0.071% | 0.0141256 | −0.140% | 1.8 s |
| 13 | 408 | 0.0142104 | −0.036% | 0.0141352 | −0.072% | 4–6 s |
| 17 | 544 | 0.0142127 | −0.020% | 0.0141407 | −0.033% | 8–10 s |
| 21 | 680 | 0.0142141 | −0.010% | 0.0141425 | −0.021% | 16–28 s |
| 29 | 952 | 0.0142155 | — | 0.0141454 | — | 38–40 s |

Converging monotonically, 0.1% of CD for a 20× speedup. **Caveat**: the trim
*angle* is not converged at nx=9 — alpha shifts ~0.6° (Plan_L) and ~0.42°
(ConstChord) between nx=9 and nx=29. Drag comparisons are unaffected; an absolute
incidence or rigging angle needs a finer chordwise mesh. At *fixed alpha* rather
than fixed CL the nx=9 error is 8.4%, which is why the trim matters.

### Total resampling cost

Native mesh → the 35 × 9 study mesh, at trimmed CL:

| | dCD trimmed | dCD at fixed alpha |
|---|---|---|
| Plan_L | −0.299% | −8.387% |
| ConstChord | −0.861% | −5.919% |

For ConstChord the spanwise step alone accounts for −0.737% of the −0.861%, so
the **spanwise** coarsening (95 native sections → 35) dominates, not the
chordwise one — the reverse of what the fixed-alpha figures suggest. Raise
`N_SPANWISE_HALF` first if absolute CD accuracy matters.

## Running it

```bash
python studies/vsp_planform/verify_roundtrip.py   # geometry-match proof
python studies/vsp_planform/run_opt.py            # VLM drag optimization
python studies/vsp_planform/airfoil_doe.py        # standalone airfoil DOE
python -m pytest tests/vsp_planform_tests/ -q     # 73 tests
```

The DOE needs `pip install aerosandbox` (pure Python, bundles NeuralFoil; no
XFoil binary). Nothing else needs OpenVSP — the CSV parser replaces it.

## Gotchas worth knowing

- **`t_over_c_cp`, never scalar `t_over_c`.** The scalar key is not in
  `openaerostruct/utils/check_surface_dict.py:16-81`; it is silently dropped with
  a `RuntimeWarning` and the drag model keeps its `np.arange(ny-1)` default.
  `openaerostruct/docs/advanced_features/scripts/run_vsp_777.py` has this bug.
- **`nx = plate.num_pnts`, not `(num_pnts+1)//2`.** The CSV `PLATE` header already
  reports camber points; OAS's halving applies to the live API object, where the
  field is the surface-point count. Halving twice discards the aft half of every
  airfoil.
- **`half_mesh` returns the mesh in metres but the stick in native inches.** Mind
  the boundary, especially for the Plan_L wingbox constraint.
- **OAS's scalar `sweep`/`taper`/`dihedral` keys are unusable here.** They apply
  one linear transform to the whole surface and cannot express a per-region rule
  or a 45° raked winglet. Only the per-section B-splines can.
- **`ruff.toml` extends `~/.config/ruff/ruff.toml`**, which is absent on this
  machine, so `ruff check` fails repo-wide. Workaround:
  `ruff check --isolated --line-length 120 --select E,W,F`.
