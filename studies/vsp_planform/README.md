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

### DOE limitation: neither baseline's section is inside the grid

| | max camber | at x/c | DOE grid |
|---|---|---|---|
| Plan_L | 2.39% | **0.667** | camber position swept only to 0.6 |
| ConstChord | **6.67%** | 0.372 | camber swept only to 4% |

This is why 4% camber won at every thickness and Reynolds number — the grid was
bounded below the real wing, not bracketing an optimum. Consequences:

- **The thickness finding stands.** 2412-vs-2418 isolates thickness at fixed
  camber, and 4- and 5-digit share a thickness distribution, so "18% costs
  25–29% profile drag at outboard Reynolds" carries across.
- **Camber and "best airfoil" rankings do not transfer** — they describe a
  family neither wing belongs to.

Use `asb.KulfanAirfoil` (CST) rather than NACA enumeration for follow-up work: it
represents both real sections, NeuralFoil evaluates it natively, and it is
differentiable, so a requirement like `t@0.75c >= target` becomes a constraint
instead of something hunted for by enumerating families. Note NACA 5-digit shares
the 4-digit *thickness* distribution exactly (only the camber line differs), so a
5-digit sweep cannot answer an aft-spar-depth question.

### Aft-spar depth

What governs depth at a 0.75c spar is *where* max thickness sits, not t/c.
`naca16018` and `naca0018` are both 18% thick, but retain 78.9% and 52.7% of it
at 0.75c respectively — a 24.5 in chord difference for identical thickness.

### The chord growth buys structure, not drag

Decomposed at MTOW, span pinned:

| case | S_ref | drag | vs baseline |
|---|---|---|---|
| A baseline | 79.57 | 10758 N | — |
| **B planform only** (c=57.2) | 83.63 | 10821 N | **+0.58%** |
| **C airfoil only** (c=40) | 79.57 | 10267 N | **−4.57%** |
| D both | 83.63 | 10250 N | −4.72% |

Growing the chord *alone* makes drag worse. The entire gain is the section
change, which can be taken without touching the chord. So a deeper spar costs
about **+0.6%** — not free, as an earlier version of this file claimed.

Further, much of the −4.57% comes from thinning the **root** 17.8% → 10.7%
(root spar depth 9.85 → 7.0 in), where bending moment is highest. Forcing
t/c ≥ as-built everywhere drops the gain to **−1.58%**. Honest range: **−1.6% to
−4.6%**, turning on whether 7 in suffices at the root — a structural question.

### Screen sections on MAX THICKNESS, not t@0.75c

With a kinking spar the spar sits where the section is deepest, so `t_max` and
its location are the correct screen. Ranking on `t@0.75c` selects blunt
aft-loaded sections and therefore selects *against* aerodynamic quality — that
mis-screen produced every bad candidate in this study (`fx2`, `goe625`,
`fx77w270s`).

Chord needed for 7 in of depth at each section's own max-thickness point:

| section | t_max | chord for 7 in | growth vs 40 in | L/D @ Cl 0.93 | Cl_max |
|---|---|---|---|---|---|
| as-built (S12) | 0.1184 | 59.1 in | 1.48× | **175.5** | 1.867 |
| **S9** | 0.1590 | **44.0 in** | **1.10×** | 150.0 | 1.837 |
| S6 | 0.1530 | 45.8 in | 1.14× | 148.8 | 1.609 |
| S11 | 0.1530 | 45.8 in | 1.14× | 144.6 | 1.725 |

Under the old screen these demanded 110–119 in. **S9 needs 10% chord growth
against 48% for the as-built, at essentially equal Cl_max** (1.837 vs 1.867);
the cost is section L/D (150 vs 175.5).

Suggests **S9 outboard, as-built inboard** — ordinary spanwise blending, and the
model already speaks this language (geoms named `S11_t0`, `S11_t2`, `S12_t2`).
The wing-level trade needs an OAS run: S9's L/D penalty applies only to outboard
area, while reduced chord growth saves wetted area there. Rough arithmetic
suggests they roughly cancel, which would make S9 outboard the better answer on
structural grounds alone.

### Within one section, spar *location* matters more than the section

Efficient sections are all thin at 0.75c — as-built 0.0635, S9 0.0589, S6 0.0627,
clustered near 0.06. Sections deep enough there are ~27% thick with blunt aft
loading, and L/D falls monotonically with aft depth:

| section | t@0.75c | L/D @ Cl 0.93 |
|---|---|---|
| as-built | 0.0635 | **175.5** |
| S9 | 0.0589 | 150.0 |
| fx77w270s | 0.1231 | 134.5 |
| goe625 | 0.1035 | 128.6 |

Selecting on aft depth therefore selects *against* aerodynamic quality. Moving
the spar forward is far cheaper. Chord needed at the B|C junction for 7 in:

| spar | as-built | S9 | S6 |
|---|---|---|---|
| **0.55c** | 70 in | 60 in | **56 in** |
| **0.65c** | 84 in | 79 in | **73 in** |
| 0.75c | 110 in | 119 in | 112 in |

0.75c → 0.65c cuts required chord 25–35% **with no section change**. The ranking
inverts across the range: the as-built needs the least chord at 0.75c and the
most at 0.55c, because the S-sections carry thickness further forward.

**This converges with the measured geometry**: the straight structural line in
the VSP model is at **68.1%** chord (box rear edge 74.2%). The 0.75c figure came
in as a stated requirement, but the model's own straight line is at 0.68c —
close to where this analysis says the spar wants to be. Treat spar location as a
design variable, not a given.

### Proposed wing 2: kinked aft spar

Constraints: front spar 0.12c, 6 in spar depth, box ≥ 65 in to y = 176 in
(inboard nacelle), box ≥ 55 in at y = 356 in (outboard nacelle).

| station | box req | min chord | as-built | aft spar | depth | box |
|---|---|---|---|---|---|---|
| y ≤ 176 in | 65 in | 103.2 | **105.0** | 0.750c | 6.55 in | 65.0 ✓ |
| y = 356 in | 55 in | 89.8 | **105.0** | 0.733c | 6.03 in | 55.1 ✓ |
| y = 675 (B\|C) | — | 50.7 | 40.0 | **0.305c** | 6.00 in | 9.4 |

**Only the junction chord changes: 40 → 50.7 in (1.27×).** All inboard stations
are already satisfied by the existing 105 in chord.

The kink does the work: aft spar 0.75c inboard where box *width* binds, sweeping
forward to 0.305c at the junction — at the section's max thickness (0.29c) —
where *depth* binds. A straight 0.75c spar would need a 94.5 in junction chord;
the kink cuts it to 50.7 in (2.4× growth → 1.27×). Most of the kink is outboard
of y = 356.

Corroboration that the constraints are real: inboard stations need 103.2 in
against 105.0 in as-built — a 1.7% margin, too tight to be coincidence. The wing
was sized to this.

**Junction box width: 20 in** (user, 2026-08-15). This was the study's largest
open input for a long time and the table below was the sensitivity to it; it is
kept because it shows how little the answer moves across the plausible range.
Before it was pinned, the solver put the aft spar at 0.305c, leaving only ~9.4 in
of box.

| box req at junction | chord (7 in deep) | aft spar | chord (6 in deep) |
|---|---|---|---|
| 0–10 in | **59.2 in** | 0.313c | **50.7 in** |
| 15 in | 60.1 in | 0.371c | 52.5 in |
| 25 in | 66.0 in | 0.499c | 59.5 in |
| 40 in | 79.5 in | 0.625c | 73.9 in |

**Free below ~10 in** — depth alone sets the chord and ~9–10 in of box comes
along with it. Above ~15 in the constraints fight (spar moves aft for width →
section is thinner there → more chord for depth), but the climb is gentle,
roughly +0.8 in of chord per +1 in of box. Never catastrophic in a plausible
range.

Reference: the as-built junction has ~46 in of box (40 in chord, 0.742c rear
edge). If the outboard box must keep that, junction chord goes to **~85 in**.

**Section choice matters more than any of this** — see the max-thickness screen
below. S9 needs only 1.10× chord growth against 1.48× for the as-built.

**Now buildable.** `param.py` used to encode the straight-spar rule as
`x_LE(y) + p·c(y) = const` with a single scalar `p`, which is what generates
region B's sweep, so a kinking spar could not be expressed at all. The rear spar
is now a spanwise *schedule* (`config.WINGBOX_REAR_SCHEDULE`, piecewise-linear
`(y_in, x/c)` breakpoints) checked at a *vector* of stations
(`config.WINGBOX_WIDTH_STATIONS`). A single constant breakpoint reproduces the
old behaviour exactly. See the drag numbers below.

### Wing 2 in full OAS

The junction box requirement was set to **20 in** (user, 2026-08-15), replacing
the 25 in placeholder the study had carried. The junction chord is *derived* from
it — the box is `(rear − front)·chord` with the spar schedule pinned, so
`20 / (0.499 − 0.12) = 52.77 in`. It was never a free pick.

Three cases, full OAS (CDi + CDv + CDw), MTOW 382 547 N, span pinned at 118 ft,
all trimmed to the same lift. `out/scripts/wing2_oas.py` → `out/logs/wing2_oas.json`:

| case | S_ref m² | cruise CL | drag N | vs as-built |
|---|---|---|---|---|
| A as-built | 79.57 | 0.979 | 10736.1 | — |
| B design point (junction chord 52.77 in) | 76.43 | 1.019 | 10703.1 | **−0.31%** |
| C optimized under the same box constraints | 77.14 | 1.010 | 10465.3 | **−2.52%** |

C converged clean (exit mode 0). These are comparable to each other and to the
full-OAS table under "Aft-spar depth"; they are **not** comparable to any
simplified-model figure in this README.

**Dropping the junction requirement 25 → 20 in moved the binding station
inboard.** At 25 in the junction drove everything (66 in chord, and the design
point cost +0.56%). At 20 in the junction is cheap enough that **y = 356 in
becomes the driver**, and case B as specified no longer closes: one `taper_B`
cannot serve both stations, so putting 52.77 in at the junction leaves y = 356
with a 54.37 in box against 55 required — **short by 0.63 in**. C resolves it by
growing the root chord to 106.3 in and sitting exactly on y = 356.

So the honest reading is that **the feasible wing 2 is case C, −2.52%**, and the
requirement worth pinning down next is the 55 in at y = 356, not the junction.
The 25 in results are kept in `out/logs/wing2_oas_25in.json` for comparison.

### The root chord is not a design variable — it is redundant

Worth knowing before anyone runs this experiment again. Widening the baseline
root chord and re-optimizing (`out/scripts/root_chord_sweep.py`, region A widened
and region B re-lofted so the junction is held) returns **the same wing**:

| root chord | `wingbox_pct` | chord at y = 100 | S_ref m² | drag N |
|---|---|---|---|---|
| 105 in | 0.6732 | 106.338 | 77.136 | 10465.2 |
| 108 in | 0.6924 | 106.338 | 77.135 | 10463.5 |
| 111 in | 0.7117 | 106.339 | 77.135 | 10461.8 |
| 114 in | 0.7309 | 106.339 | 77.134 | 10460.1 |
| 117 in | 0.7500 ← bound | 106.359 | 77.144 | 10458.8 |
| 120 in | 0.7500 ← bound | 109.086 | 78.650 | 10512.3 |

Station chords agree to three decimals and `wingbox_pct` tracks `1/k` to four,
because under the `root_le_fixed` rule `wingbox_pct` scales every chord together
and simply undoes the widening. At 117 in it hits its 0.75 bound; the 120 in row
is the only one where the widening survives, so **its +0.45% is the price of a
design-variable bound, not of chord**. Do not read that row as a drag gradient.

### What inboard box margin actually costs

The optimizer never *buys* margin — it drives the binding stations to zero slack
by construction — so margin has to be priced by requiring it. Raising the y = 100
and y = 176 requirements together, everything else at the wing 2 design point,
re-optimized in full OAS each step (`out/scripts/margin_sweep.py`):

| inboard req | vs as-built | S_ref m² | drag N | vs 65 in | binding stations |
|---|---|---|---|---|---|
| 65 in | — | 77.14 | 10465.3 | — | y=356, junction |
| 67 in | +3.1% | 77.56 | 10480.9 | +0.15% | y=176, junction |
| 69 in | +6.2% | 79.34 | 10546.4 | +0.77% | y=176, junction |
| 71 in | +9.2% | 81.12 | 10612.1 | +1.40% | y=176, junction |
| 73 in | +12.3% | 82.90 | 10677.8 | +2.03% | y=176, junction |
| 75 in | +15.4% | 84.68 | 10743.7 | +2.66% | y=176, junction |

**The first 2 in of margin is nearly free (+0.15%); after that it costs about
0.31% of drag per inch**, essentially linearly. The kink is the reason for the
cheap first step — at 65 in the inboard stations are not binding at all, and the
requirement only starts driving the design once y = 176 takes over from y = 356.

For scale: 75 in of inboard box gives back the entire −2.52% that wing 2 bought.

### THE SPAR DEPTH REQUIREMENT IS WHAT WING 2 ACTUALLY COSTS

Sweeping the kink's outboard end in full OAS (`out/scripts/spar_sweep_oas.py`,
0.750c held inboard, 20 in box, planform re-optimized at each step, depth checked
afterwards against the as-built section's own thickness distribution):

| junction spar | junction chord | S_ref m² | drag N | vs as-built | depth | 7 in? | 6 in? |
|---|---|---|---|---|---|---|---|
| 0.350c | 86.96 in | 87.58 | 10850.7 | +1.07% | 10.21 in | ✓ | ✓ |
| 0.400c | 71.44 in | 82.19 | 10650.5 | −0.80% | 8.21 in | ✓ | ✓ |
| 0.450c | 60.61 in | 78.45 | 10512.9 | −2.08% | 6.74 in | ✗ | ✓ |
| **0.499c** (design point) | 52.77 in | 77.14 | **10465.3** | **−2.52%** | **5.60 in** | ✗ | ✗ |
| 0.550c | 46.51 in | 76.87 | 10459.7 | −2.57% | 4.64 in | ✗ | ✗ |
| 0.600c | 41.66 in | 76.70 | 10453.5 | −2.63% | 3.85 in | ✗ | ✗ |
| 0.650c | 39.92 in | 76.66 | 10459.2 | −2.58% | 3.34 in | ✗ | ✗ |
| 0.700c | 39.92 in | 76.66 | 10459.2 | −2.58% | 2.96 in | ✗ | ✗ |

**The wing 2 design point does not meet its own depth requirement.** It was
specified with 7 in of depth at 0.499c; on the as-built section that station
actually gives **5.60 in**. Worse, at the 52.77 in junction chord the section's
*maximum* thickness is only 6.25 in, so **7 in of depth is unreachable at any
spar station** — the 20 in box and the 7 in depth are in direct conflict once the
box requirement is what sets the chord.

So the headline −2.52% is not available with the as-built section:

- **7 in depth → spar 0.400c, junction chord 71.44 in, −0.80%.**
- **6 in depth → spar 0.450c, junction chord 60.61 in, −2.08%.**

The 7-vs-6 in question, open since the spar work began, is therefore worth
**~1.3% of cruise drag** — much bigger than anything the twist or margin
constraints move, and the single highest-value input still unpinned.

This is also what makes the DOE v3 result actionable. The gap is entirely about
**thickness retained at 0.45–0.50c**; a section holding more of its thickness
there recovers it directly. That retention has still not been measured for any
candidate — it is the obvious next run, and the DOE now has 2048 feasible
sections to draw from.

Two smaller notes. Drag flattens past 0.55c and the junction chord floors at
39.92 in — the box constraint stops binding there and something else takes over,
so those rows are not a continued trend. And the 0.350c row costing *more* than
the as-built baseline shows the kink can be pushed too far forward: an 87 in
junction chord buys depth at a wetted-area price that swamps the planform gain.

### WING 3 — the design that closes

Wing 2's depth requirement existed to house an **aileron actuator**. Moving the
ailerons inboard to **90% semi-span (y = 637.2 in)** and accepting **6 in** of
depth is wing 3. Figures: `out/figures/wing3_planform.png`, `wing3_3d.png`.
Design point: `out/logs/wing3_design_point.json`.

| | value |
|---|---|
| drag | **10463.9 N, −2.54% vs as-built** |
| S_ref | 77.09 m² (as-built 79.57) |
| cruise CL | 1.011 |
| front spar | 0.12c |
| aft spar | 0.750c inboard, kinking to **0.550c** at the junction |
| junction chord | 51.78 in |
| chord at the aileron | 55.98 in, giving exactly 6.00 in of depth |
| region A | ends at the inboard nacelle (176 in, snaps to 179.2 in) |

Moving the actuator inboard is what makes the whole problem cheap. Same sweep at
both depths, at y = 637.2 in:

| junction spar | 7 in depth | 6 in depth |
|---|---|---|
| 0.499c | −2.40% | −2.52% |
| **0.550c** | −1.90% | **−2.54%** |
| 0.600c | −1.25% | −2.47% |
| 0.650c | −0.36% | −1.79% |
| 0.700c | +0.83% | −0.78% |
| 0.750c | +2.48% | +0.65% |

**The 7-vs-6 in question collapsed from ~1.3% to 0.14%.** At the winglet junction
7 in was worth −0.80% against 6 in's −2.08%; at 90% semi-span it is −2.40%
against −2.54%. The same requirement, moved 38 in inboard, stopped mattering.

Two things not to over-read:

- **The spar fraction is flat.** 0.499c, 0.550c and 0.600c give −2.52%, −2.54%
  and −2.47%. The spread between the top two is 1.4 N out of 10464 — below any
  meaningful resolution of this model. **Choose the spar fraction on structural
  grounds, not aerodynamic ones**; the aero is indifferent across that range.
- The 6 in constraint **binds exactly** (6.00 in, zero margin) but costs only
  ~0.09% against the unconstrained optimum. It is binding, not expensive.

Section is still the as-built. The DOE v3 candidates are not in this number, and
the OAS model could not see them if they were.

### Monotonic twist costs 0.30%

The optimizer genuinely wants non-monotonic twist — more control points produced
*more* sign changes (5 → 8), so it is the VLM's preference and not a spline
artifact. A wavy twist is awkward to build, so the question is the price.

Constraining the twist to be non-increasing outboard across region B and
re-optimizing (`out/scripts/monotonic_twist.py`, 23 stations from y = 193.6 to
674.9 in, full OAS, everything else at wing 2 case C):

| | drag N | S_ref m² | region-B slope sign changes |
|---|---|---|---|
| free | 10465.3 | 77.14 | 1 |
| monotonic | 10496.4 | 77.36 | 0 |

**+31.1 N, or +0.297%** — about an eighth of the −2.52% wing 2 buys. Cheap enough
that manufacturability probably wins; the constraint is worth carrying by default
and relaxing only if that eighth is being fought over.

The constraint is applied to `twist_abs`, not `twist_cp`, deliberately: a monotone
set of control points does not produce a monotone spline, and it is the physical
distribution that has to be built.

**Trap, and it bit this run.** `run_opt.add_optimization` ends with its own
`prob.setup()`, and a constraint added *after* a `setup()` is silently discarded —
no error, no warning. The first attempt added it afterwards and returned a
"monotonic" optimum identical to the free one **down to the last twist control
point**, which is the fingerprint of a constraint that is absent rather than one
that does not bind. Add constraints before `add_optimization`, and check that a
constraint you expect to bite actually changed the design vector. (Adding a
*component* has the opposite problem — OpenMDAO refuses `add_subsystem` after
setup, which is what `build_problem(..., extra=)` exists for.)

### ~~The as-built section is better than any replacement found~~ — RETRACTED

**This claim was an artifact of a broken feasibility screen. It does not survive
the corrected one.** Kept here because it was load-bearing for a while and the
retraction matters more than the original.

Measured properly — t/c 0.1184, camber at x/c 0.368 (**not** the 0.50 an early
`naca6512` mapping implied; conclusions drawn from that mapping were built on the
wrong section) — the as-built gives L/D ~168–176 at Cl 0.93, Cl_max ~1.82–1.87.
The claim was that a scan of all 2174 database sections produced nothing better,
the best candidates reaching only L/D 128–134.

That scan pinned the aft spar **at each section's max thickness** and then
demanded the box width there. Sections peaking well forward — which is most of
the good ones — were charged `BOX/(x_t − 0.12)` for chord and screened out. The
survivors were the blunt aft-loaded sections, and *those* are the ones that top
out at L/D 128–134.

Under the corrected screen (`out/scripts/doe_v3.py`, spar position scanned rather
than pinned; 7 in depth, 20 in box), **2048 of 2174 sections are feasible and the
as-built ranks 321st**, at L/D 168.1 with a 62.6 in junction chord. The leaders:

| rank | airfoil | t_max | spar x/c | junction chord | L/D @ Cl 0.93 | Cl_max | conf |
|---|---|---|---|---|---|---|---|
| 1 | fx78k140a20 | 0.1409 | 0.512 | 51.1 in | 252.7 | 1.671 | 0.905 |
| 2 | fx78k150 | 0.1510 | 0.533 | 48.4 in | 245.9 | 1.617 | 0.905 |
| 4 | fx74130wp2 | 0.1299 | 0.491 | 54.2 in | 242.0 | 1.577 | 0.917 |
| 6 | ah81k144 | 0.1446 | 0.517 | 50.6 in | 239.7 | 1.649 | 0.927 |
| 9 | e694 | 0.1543 | 0.544 | 47.2 in | 231.0 | 1.830 | 0.918 |
| — | **as-built** | 0.1184 | 0.443 | **62.6 in** | **168.1** | 1.821 | 0.838 |

These beat the as-built on *both* axes at once — higher 2D L/D **and** a smaller
junction chord (47–54 in against 62.6), which is exactly the chord growth the
whole wing 2 exercise is paying for. The five manual sections all clear the
corrected screen too (they were **all** excluded by v2): S9 ranks 979th, S6
1318th, S11 1466th, S7 1556th — so the manual picks remain worse than the
as-built, which is the one part of the original claim that holds.

**Two caveats before anyone acts on this.** First, the leaders are Wortmann/Eppler
sailplane sections and their L/D comes largely from extensive predicted laminar
flow; NeuralFoil confidence is high (0.90+) but transition on a real transport
wing at this Reynolds number will not behave like the prediction. Second and more
decisively — **the OAS model cannot see any of it.** OAS reduces a section to two
scalars, t/c and c_max_t, through the Raymer form factor
(`aerodynamics/viscous_drag.py:103`). A 50% 2D L/D gain enters the wing-level
drag only through whatever those two numbers change. So this table selects
candidates for an analysis the study does not currently run; its immediately
usable column is the **junction chord**, which is a geometry result and is real.

So prefer **perturbing the original** (thickening its aft half) over replacing it:

| scale | t@0.75c | chord for 7 in | Cd@Cl 0.93 | L/D | Cl_max | NeuralFoil conf |
|---|---|---|---|---|---|---|
| 1.0 (original) | 0.0633 | 110.5 in | 0.00530 | 175.5 | 1.867 | 0.647 |
| 1.8 | 0.0929 | 75.3 in | **0.00497** | **186.9** | 1.855 | 0.422 |
| 2.2 | 0.1077 | 65.0 in | 0.00536 | 173.4 | 1.818 | 0.316 |
| 2.6 | 0.1225 | 57.2 in | 0.00659 | 141.1 | 1.468 | 0.256 |

**UNVALIDATED — treat these as the least reliable numbers here.** NeuralFoil
confidence falls to 0.316 at scale 2.2 (0.89–0.96 for database sections); the
perturbed shapes are outside its training distribution, and the original itself
only scores 0.647. Confirm with XFoil (`asb.XFoil`, needs the binary) before
relying on any of it.

### Tip-stall risk is a symptom of an inadequate section, not of twist

With a section that has real Cl_max margin, twist optimisation drives the **tip
negative** (−3.68°) rather than to the washin that produced the tip-loading
warning. The earlier washin was the optimizer compensating for a section with no
margin to spare — so the fix is section capability, not banning washin.

**Qualifier**: that −3.68° is an endpoint, and the distribution is *not*
monotonic — five sign changes in its gradient (humps at η ≈ 0.17 and 0.73, a dip
at the A|B junction, then a winglet spike). The winglet spike is inherited from
the VSP loft and appears in the frozen case too.

The inboard humps were **hypothesised to be a 5-control-point spline artifact.
That was tested and is wrong** — raising to 15 control points produced *more*
oscillation (5 → 8 sign changes), and bought only ~0.15% drag. The optimizer
genuinely wants non-monotonic twist: it finds local induced-drag benefit from
wiggling incidence, which a VLM rewards and a real wing will not deliver. Hence
a monotonic-twist constraint through region B is a physical necessity, not
cosmetic, and its cost is worth measuring precisely because the benefit it
removes is not real.

**What is robust**: the tip goes strongly negative in every case tested
(−1.65 to −4.08°). Washout at the tip is a real result.

Report twist as a distribution, never as endpoints — the endpoint summary hid
all of this.

### CAUTION: use first-peak Cl_max, never the global maximum

`fx2` appeared to be the best candidate (−4.72%) until its lift curve was read
properly. Its Cl_max on **first peak is 0.946 at α = 4°**; the apparent 1.532 is
a post-reattachment peak after the section has already separated. The wing
operates at **CL ≈ 0.93**, so `fx2` stalls at the cruise point. Disqualified.

Any section that separates and reattaches will report a flattering global
maximum. Always take the first local peak.

Note the as-built section beats both 6-series candidates on this measure
(1.815 vs 1.191 for `naca664221`).

### Meeting a deeper rear spar, if the chord may grow

Requirement: 7 in of depth at the 0.75c spar, at the ConstChord B|C junction
(as-built chord 40.0 in). The as-built section delivers **2.67 in**. No airfoil
in a 2174-section database reaches 7 in at that chord, so the chord must grow.

Measured in the full OAS model at MTOW, span pinned at 118 ft:

| case | c_junc | S_ref | induced | viscous | total | min spar in region B |
|---|---|---|---|---|---|---|
| as-built | 40.0 | 79.57 | 7254 | 3504 | 10758 | 2.57 in |
| 66(4)-221, t/c ≥ as-built | 56.0 | 83.35 | 7169 | 3419 | 10588 (−1.58%) | 6.02 in |
| 66(4)-221, min t/c | 56.0 | 83.35 | 7169 | 3152 | **10321 (−4.06%)** | 6.02 in |
| 66(4)-221, min t/c | 64.0 | 85.24 | 7158 | 3191 | 10348 (−3.81%) | 6.21 in |
| **fx2, min t/c** | 57.2 | 83.63 | 7166 | 3084 | **10250 (−4.72%)** | 6.10 in |

**Chord growth has an optimum near 56–57 in.** The c=64 case is *worse* than
c=56 despite more chord — added wetted area outruns the thinner section it
permits. Growing the chord is not monotonically good, and a back-of-envelope
that ignores wetted area will miss this.

Viscous drag falls 3505 → 3148 N *despite 4.7% more wetted area* — the
`c_max_t` 0.302 → 0.45 shift working through the Raymer form factor. This result
does **not** depend on any NACA mapping of the as-built section: OAS consumes only
t/c and `c_max_t`, both measured from the geometry.

**Parameterization limit — 7 in is not reachable through the spline.** Delivered
t/c at the junction is 0.1776 against 0.2101 requested. OAS's `t_over_c`
`SplineComp` is order-4 and *approximates* its control points rather than
interpolating them, so it cannot hold a peak that climbs at the junction then
drops into the thin winglet. Raising control points 5 → 15 → 35 gave
4.07 → 6.10 → 6.55 in, still converging short.

Inverting the spline to hit a target distribution is delicate — a plain
least-squares inversion produced **negative t/c** and nonsense drag, and a
bounded inversion against evenly-spaced points silently solved the wrong problem
because OAS interpolates at true mid-panel stations
(`utils/interpolation.py`, `mid_panel=True`). The working approach measures the
control-point → t/c map from the live model, sampled at real mid-panel stations;
it reproduces the baseline to 0.5%.

So **6.0–6.2 in is the ceiling for this parameterization, not for the design.**
A true 7 in needs `t_over_c` driven per panel — replacing the `SplineComp` in
`param.py`'s geometry group with a direct input. Straightforward, not yet done.

### Aspect ratio is not a design variable once span is fixed

At fixed lift and fixed span, induced drag is **area-independent**:

```
Di = q·S·CL²/(π·AR·e),  CL = W/(qS),  AR = b²/S   ->   Di = W²/(q·π·b²·e)
```

Verified numerically: 7719.4 N at every candidate area from 77.7 to 84.3 m²,
identical to the decimal. CDi appears to improve as AR rises, but that is the
coefficient's normalization moving, not the force.

**Caveat — the closed form assumes fixed `e`.** The real VLM, computing induced
drag from the actual vortex system, gives 7254 → 7169 N across a 79.6 → 83.6 m²
change: a **1.2% drift**, not zero. That drift is physical — growing the junction
chord makes the wing less tapered, moving loading closer to elliptical and
raising span efficiency. So the VLM finds a small induced benefit the fixed-`e`
algebra cannot see. The conclusion holds (area is not a meaningful induced-drag
lever at fixed span), but it is not an exact invariance.

Consequence: with span pinned at the 118 ft limit, growing chord costs **nothing**
in induced drag — only wetted area. Any chord-growth decision (e.g. to reach a
required spar depth) is therefore a pure airfoil-quality contest ranked by
`area × section Cd`, and the sections needing least chord growth tend to be the
*worst*, because they are thick and thickness costs profile drag faster than the
apparent AR gain repays.

## READ FIRST: the optimized geometries are probably not flyable as drawn

**Status: inferred, not confirmed.** The reasoning below is a VLM local-Cl
result compared against 2D CLmax data. An independent check with
`asb.NonlinearLiftingLine` was attempted and did **not** resolve it: the implicit
solve reached CL 1.492 at alpha 6 deg, still climbing, then went NaN at alpha
8 deg without reaching CLmax. Divergence is suggestive of stall onset but is
equally consistent with the solver struggling on sections at the edge of
NeuralFoil's training range. Confirming it needs alpha continuation with warm
starts, and CST sections rather than 4-digit ones.


The optimizer spends stall margin because nothing in the model prices it. A VLM
is linear and inviscid — it will report a local Cl of 1.5, or 3.0, without
complaint.

Optimized ConstChord requires **local Cl = 1.491 at η = 0.96**. Against 2D CLmax
from the DOE at t/c 0.10:

| section | CLmax @ Re 1.7e6 | margin vs 1.491 |
|---|---|---|
| naca0010 | 1.030 | −30.9% |
| naca2410 | 1.300 | −12.8% |
| naca2412 | 1.345 | −9.8% |
| naca4410 | 1.570 | +5.3% |

Only a 4%-camber section clears it, by 5.3% — and the optimized 0.152 m tip
chord sits at **Re ≈ 0.7e6**, below the DOE's sampled range, where CLmax is
lower still.

Worse, the *stall order* inverts. Baseline peak section loading is at η = 0.03
(root-first stall: ailerons stay effective, wing drops straight ahead).
Optimized peak is at η = 0.96 — tip-first stall, roll-off exactly when roll
authority is needed.

**Mechanism**: region A is frozen under the `preserved` rule, so the whole area
cut comes out of the outboard taper (0.381 → 0.163, far more tapered than the
~0.4 that approximates elliptical loading). With outboard chord gone, incidence
is the only lever left to load the tip — so washin is the optimizer repairing a
lift distribution its own area cut broke, not a free choice.

**Cheapest real guard**: a stall-margin constraint capping local Cl at, say,
0.9 × 2D CLmax interpolated from the DOE against local t/c and Re. Both pieces
already exist — OAS gives the spanwise Cl distribution, the DOE gives CLmax vs
t/c vs Re. That targets the exploit rather than banning washin, which would be
treating the symptom. Secondary guards: a minimum tip chord, or letting region A
share the area change.

## What this model can and cannot answer

**Size is not optimizable here, and no amount of solver tuning will change that.**

At fixed lift, drag splits into induced (`≈ L²/(q·π·b²·e)` — depends on lift and
span, *not* area) and profile (`≈ q·S·CD_p` — grows with area). With lift and
span both fixed, induced drag is a constant and `dD/dS > 0` everywhere. There is
no interior optimum: the optimizer walks downhill until a constraint stops it,
then grinds on a degenerate active set. Gradients check clean at 4e-6; this is
the problem statement, not the solver.

| formulation | size | outcome |
|---|---|---|
| fixed_cl, `S_ref` pinned | pinned | **converged, 36–37 iters, exit 0** |
| fixed_cl, `S_ref` ±10% band | free | FAILED 716/615, sat on the floor |
| fixed_lift, CL ≤ 1.05, area free | free | FAILED 946/477, stopped at 74.1947 against a 74.1948 floor |

An aero-only VLM has no mechanism to make a bigger wing cost anything. In
reality a bigger wing is heavier → needs more lift → costs drag. Free area only
becomes meaningful once something prices it: the aerostructural path
(`AerostructPoint` + wingbox), or fuel burn as the objective.

So split the variables by whether they have an interior optimum:

- **Twist does.** There is a genuine best distribution for elliptical loading,
  and it converged reliably in every well-posed run. Gradient optimization earns
  its keep, especially at 5+ control points.
- **Span and area do not.** Span goes to the 118 ft limit; area goes to whatever
  floor is set. Sweep these in an outer loop and gradient-optimize twist and
  alpha inside — every inner solve is then well-posed, and you get a trade
  surface instead of one bound-determined point.

### Span

ConstChord is drawn *exactly* to the 118 ft limit (708.000 in × 2 = 1416 in).
Plan_L is 115.685 ft, leaving 2.315 ft unused. At constant area and lift that
unused span is worth:

| | AR | CDi | CDv | CDi share |
|---|---|---|---|---|
| Plan_L | 14.45 | 0.005511 | 0.008695 | 38.8% |
| ConstChord | 16.26 | 0.005214 | 0.008911 | 36.9% |

Stretching Plan_L to 118 ft raises b² by 4.04%: CDi −3.88%, **total drag
−1.51%** — comparable to everything the planform optimizer found, for free.

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
