"""Which real airfoils can carry a 7 inch aft spar at the winglet junction?

The constraint is structural and absolute: at least 7 in of thickness at the
0.75c rear spar, from the root out to where the winglet starts. Outboard of that
station the winglet may be thinner.

The binding station is therefore the *last section before the winglet*, because
chord falls monotonically outboard through region B. For ConstChord that is
section 90, chord 40.0 in as built.

Thickness at the spar is ``(t/c at 0.75c) * chord``, so an airfoil is described
here by ``t75 = t(0.75c)/c`` rather than by t_max. The chord needed to meet the
requirement is then ``7 / t75`` inches, which is the number that decides whether
the section is usable at the as-built chord or forces the chord to grow.

Only real, published sections are considered -- everything comes from
AeroSandbox's airfoil database, filtered to recognizable families. No synthesized
or CST-perturbed shapes.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/alex/repos/OpenAeroStruct")

import aerosandbox as asb

from studies.vsp_planform import config
from studies.vsp_planform.degen_csv import read_degen_csv
from studies.vsp_planform.regions import detect_regions

SPAR_X_C = 0.75
SPAR_MIN_IN = 7.0
IN = 0.0254

# The whole database is real published sections, so nothing is filtered by
# family -- everything with a usable thickness is scanned and ranked.
TOC_RANGE = (0.12, 0.30)
SHORTLIST = 30
AERO_CL = 0.5  # rank sections at a common lift coefficient, not a common alpha


def database_names():
    root = Path(asb.__file__).parent / "geometry" / "airfoil" / "airfoil_database"
    return sorted(p.stem for p in root.glob("*.dat"))


def measure(name):
    """t_max, its chordwise position, and thickness at the spar. None if unusable."""
    try:
        af = asb.Airfoil(name)
        if af.coordinates is None or len(af.coordinates) < 20:
            return None
        xs = np.linspace(0.02, 0.98, 97)
        t = np.array([float(af.local_thickness(x_over_c=x)) for x in xs])
        if not np.all(np.isfinite(t)) or t.max() <= 0:
            return None
        return {
            "name": name,
            "t_max": float(t.max()),
            "x_tmax": float(xs[t.argmax()]),
            "t75": float(af.local_thickness(x_over_c=SPAR_X_C)),
            "airfoil": af,
        }
    except Exception:
        return None


def main():
    comp = [c for c in read_degen_csv(config.BASELINES["const_chord"]) if c.surf_index == 0][0]
    stick = comp.stick
    regions = detect_regions(stick)
    j = regions.idx_c_start
    chord_in = float(stick.chord[j])
    toc_asbuilt = float(stick.toc[j])

    print("=" * 92)
    print("Binding station: last section before the winglet (ConstChord section "
          f"{j}, y = {stick.le[j, 1]:.1f} in)")
    print(f"  as-built chord {chord_in:.2f} in, t/c {toc_asbuilt:.4f}")
    print(f"  requirement: >= {SPAR_MIN_IN:.1f} in of thickness at {SPAR_X_C:.2f}c")
    print(f"  -> need t(0.75c)/c >= {SPAR_MIN_IN / chord_in:.4f} at the as-built chord")
    print("=" * 92)

    names = database_names()
    print(f"\nscanning {len(names)} database airfoils ...")
    rows = []
    for n in names:
        m = measure(n)
        if m is None or not (TOC_RANGE[0] <= m["t_max"] <= TOC_RANGE[1]):
            continue
        rows.append(m)
    print(f"  {len(rows)} real sections with t/c in [{TOC_RANGE[0]}, {TOC_RANGE[1]}]")

    for r in rows:
        r["t75_frac"] = r["t75"] / r["t_max"]
        r["chord_needed_in"] = SPAR_MIN_IN / r["t75"]

    rows.sort(key=lambda r: -r["t75"])
    shortlist = rows[:SHORTLIST]

    print(f"\n{'airfoil':>14} {'t_max':>7} {'x_tmax':>7} {'t@.75c':>8} {'/t_max':>7} "
          f"{'chord for 7in':>14} {'vs as-built':>12}")
    print("-" * 92)
    for r in shortlist:
        ratio = r["chord_needed_in"] / chord_in
        print(
            f"{r['name']:>14} {r['t_max']:7.4f} {r['x_tmax']:7.2f} {r['t75']:8.4f} "
            f"{r['t75_frac']:7.3f} {r['chord_needed_in']:11.1f} in {ratio:11.2f}x"
        )

    # Aero on the shortlist, at the Reynolds number of the chord each one needs.
    print(f"\n{'airfoil':>14} {'chord in':>9} {'Re':>10} {'Cd@Cl=.5':>9} "
          f"{'L/D@Cl=.5':>10} {'Cl_max':>7} {'a_stall':>8}")
    print("-" * 92)
    alphas = np.arange(-4.0, 20.1, 1.0)
    for r in shortlist:
        chord_m = r["chord_needed_in"] * IN
        re = config.RE_PER_M * chord_m
        try:
            aero = r["airfoil"].get_aero_from_neuralfoil(
                alpha=alphas, Re=re, mach=config.MACH, model_size="medium"
            )
            cl = np.atleast_1d(np.asarray(aero["CL"], dtype=float))
            cd = np.atleast_1d(np.asarray(aero["CD"], dtype=float))
            k = int(np.argmax(cl))
            # Interpolate Cd at a common Cl, on the pre-stall branch only.
            pre = slice(0, k + 1)
            if cl[pre].max() >= AERO_CL >= cl[pre].min():
                cd_at = float(np.interp(AERO_CL, cl[pre], cd[pre]))
                ld = AERO_CL / cd_at
                cd_s, ld_s = f"{cd_at:9.5f}", f"{ld:10.1f}"
            else:
                cd_s, ld_s = f"{'n/a':>9}", f"{'n/a':>10}"
            print(
                f"{r['name']:>14} {r['chord_needed_in']:9.1f} {re:10.3e} {cd_s} "
                f"{ld_s} {cl[k]:7.4f} {alphas[k]:7.1f}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{r['name']:>14} {r['chord_needed_in']:9.1f} {re:10.3e}  NeuralFoil failed: {type(exc).__name__}")

    print("\nNote: chord needed is what the 7 in spar forces at this station. The as-built")
    print(f"chord is {chord_in:.1f} in, so any ratio above 1.0 means the chord has to grow.")


if __name__ == "__main__":
    main()
