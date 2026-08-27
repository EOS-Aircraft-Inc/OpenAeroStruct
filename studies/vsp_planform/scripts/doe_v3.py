"""Airfoil DOE v3 -- corrected spar-position screen (queued item 1 in HANDOFF.md).

v1 ranked on thickness at 0.75c, which selected blunt aft-loaded sections and
produced picks later overturned.

v2 fixed the ranking but broke the feasibility screen: it pinned the aft spar AT
the section's max thickness and then applied the box-width constraint there. For
a section peaking at 0.29c that demands a chord of BOX/(0.29-0.12) = 5.9x BOX,
which threw out all five manual sections (they peak at 0.26-0.33c). Its top-15 is
real but answers a narrower question than the study was asking.

v3 scans the spar position. Depth and box pull in opposite directions -- moving
the spar aft widens the box but lands it on thinner section -- so the required
junction chord is

    c_req(x) = max( DEPTH / t(x),  BOX / (x - x_front) )

and the section's true cost is the MINIMUM of that over all admissible x, with
the minimising x reported as the spar position it wants. This is the minchord()
pattern from the spar-kink calc, applied per-section.

Box requirement is 20 in (user, 2026-08-15), down from the 25 in placeholder.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/alex/repos/OpenAeroStruct")
import aerosandbox as asb  # noqa: E402

from studies.vsp_planform import config  # noqa: E402
from studies.vsp_planform.degen_csv import read_degen_csv  # noqa: E402

AL = np.arange(-6, 20.05, 0.5)
CL_OP = 0.93
DEPTH = 7.0
BOX = 20.0
FMIN = 0.12
AMAX = 0.75
XS = np.linspace(FMIN, AMAX, 120)
CHORD_CAP = 120.0


def fp(cl):
    """First peak of the Cl curve. Never the global max -- see README."""
    for i in range(2, len(cl) - 1):
        if cl[i] >= cl[i - 1] and cl[i] > cl[i + 1]:
            return i
    return int(np.argmax(cl))


def selig(p):
    U, L, cur = [], [], None
    for ln in open(p):
        s = ln.strip()
        if s.startswith("#"):
            cur = U if "Upper" in s else L
            continue
        t = s.split()
        if len(t) == 2 and cur is not None:
            try:
                cur.append((float(t[0]), float(t[1])))
            except ValueError:
                pass
    U, L = np.array(U), np.array(L)
    return asb.Airfoil(name=Path(p).stem, coordinates=np.vstack([U[::-1], L[1:]])).repanel()


def asbuilt():
    c = [x for x in read_degen_csv(config.BASELINES["const_chord"]) if x.surf_index == 0][0]
    p = c.plate
    j = 90
    xc = p.x[j] + p.nCamber_x[j] * p.zCamber[j]
    zc = p.z[j] + p.nCamber_z[j] * p.zCamber[j]
    t = p.t[j]
    nx, nz = p.nCamber_x[j], p.nCamber_z[j]
    up = np.column_stack([xc + 0.5 * t * nx, zc + 0.5 * t * nz])
    lo = np.column_stack([xc - 0.5 * t * nx, zc - 0.5 * t * nz])
    le = np.array([xc[-1], zc[-1]])
    te = np.array([xc[0], zc[0]])
    ch = np.linalg.norm(te - le)
    ct, st = (te - le) / ch
    loc = lambda P: (lambda d: np.column_stack([(d[:, 0] * ct + d[:, 1] * st) / ch, (-d[:, 0] * st + d[:, 1] * ct) / ch]))(P - le)
    U, L = loc(up), loc(lo)
    return asb.Airfoil(name="as-built", coordinates=np.vstack([U, L[::-1][1:]])).repanel()


def minchord(T):
    """Smallest junction chord that meets BOTH depth and box, over spar position.

    Returns (chord_in, spar_x_over_c, thickness_fraction_there). The two terms
    move against each other, so this has an interior minimum; taking argmax(T)
    instead -- v2's bug -- lands on the depth-optimal spar and pays whatever the
    box then costs.
    """
    with np.errstate(divide="ignore"):
        c_depth = DEPTH / np.where(T > 1e-6, T, np.nan)
        width = XS - FMIN
        c_box = np.where(width > 1e-3, BOX / np.where(width > 1e-3, width, np.nan), np.inf)
    c_req = np.fmax(c_depth, c_box)
    if not np.any(np.isfinite(c_req)):
        return None
    k = int(np.nanargmin(c_req))
    return float(c_req[k]), float(XS[k]), float(T[k])


def evaluate(af):
    try:
        T = np.array([float(af.local_thickness(x_over_c=x)) for x in XS])
        if not np.all(np.isfinite(T)) or T.max() <= 0.05 or T.max() > 0.32:
            return None
        mc = minchord(T)
        if mc is None:
            return None
        cj, xspar, t_spar = mc
        if cj > CHORD_CAP:
            return None
        k = int(np.argmax(T))
        re = config.RE_PER_M * cj * 0.0254
        a = af.get_aero_from_neuralfoil(alpha=AL, Re=re, mach=config.MACH, model_size="medium")
        g = lambda n: np.atleast_1d(np.asarray(a[n], float)).ravel()
        cl, cd, cf = g("CL"), g("CD"), g("analysis_confidence")
        i = fp(cl)
        if cl[: i + 1].max() < CL_OP:
            return None
        cdop = float(np.interp(CL_OP, cl[: i + 1], cd[: i + 1]))
        return dict(
            name=af.name,
            tmax=float(T[k]),
            xt=float(XS[k]),
            xspar=xspar,
            t_spar=t_spar,
            cj=cj,
            re=re,
            ld=CL_OP / cdop,
            cd=cdop,
            clmax=float(cl[i]),
            ast=float(AL[i]),
            conf=float(cf.min()),
        )
    except Exception:
        return None


if __name__ == "__main__":
    rows = []
    manual = {}
    for nm, p in (("as-built", None), ("S6", "S6_airfoil"), ("S7", "S7_airfoil"), ("S9", "S9_airfoil"), ("S11", "S11_airfoil")):
        af = asbuilt() if p is None else selig(f"/mnt/c/Users/AlexanderAmos/Downloads/{p}.dat")
        af.name = nm
        r = evaluate(af)
        if r:
            manual[nm] = r
            rows.append(r)
        else:
            print(f"  NOTE: manual section {nm} still infeasible under the v3 screen")

    root = Path(asb.__file__).parent / "geometry" / "airfoil" / "airfoil_database"
    names = sorted(x.stem for x in root.glob("*.dat"))
    print(f"screening {len(names)} database sections ({DEPTH:.0f} in depth, {BOX:.0f} in box, spar scanned {FMIN}-{AMAX})...")
    for n in names:
        try:
            r = evaluate(asb.Airfoil(n))
        except Exception:
            r = None
        if r:
            rows.append(r)

    rows.sort(key=lambda r: -r["ld"])
    print(f"{len(rows)} feasible\n")
    hdr = f"{'rank':>4} {'airfoil':>16} {'t_max':>7} {'x_t':>5} {'spar':>6} {'t@spar':>7} {'chord in':>9} {'L/D@.93':>8} {'Clmax1':>7} {'a':>5} {'conf':>6}"
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows[:20], 1):
        tag = "  <-- manual" if r["name"] in manual else ""
        print(
            f"{i:4d} {r['name']:>16} {r['tmax']:7.4f} {r['xt']:5.2f} {r['xspar']:6.3f} {r['t_spar']:7.4f} "
            f"{r['cj']:9.1f} {r['ld']:8.1f} {r['clmax']:7.3f} {r['ast']:5.1f} {r['conf']:6.3f}{tag}"
        )

    print("\nmanual selections:")
    for nm, r in manual.items():
        rk = [i for i, x in enumerate(rows, 1) if x is r][0]
        print(
            f"   {nm:>9}  rank {rk:4d} of {len(rows)}   L/D {r['ld']:6.1f}  chord {r['cj']:6.1f} in "
            f" spar {r['xspar']:.3f}c  Clmax {r['clmax']:.3f}  conf {r['conf']:.3f}"
        )
