"""Build a WingCalc input deck from an OpenAeroStruct design, and size it.

WingCalc takes its planform from an OpenVSP station export, so a coupled run has
to write one (``geometry.py``) and then bring the rest of the deck into line with
the OAS model. Three things must be overridden or the sizing is quietly wrong:

  spar ratios   the shipped PlanL deck carries 0.1315 / 0.6; the study uses 0.12
                front. WingCalc reads ONE aft ratio (io/inputs.py) and cannot
                express the study's 0.750 -> 0.550 kink, so 0.750 is used --
                correct inboard, where the binding bays are.
  wingbox span  the reference constant-chord runs use 1356 in / 20 bays; the
                PlanL deck ships 1328 in / 19. 678 in semi also brackets the
                winglet junction at 674.9 in, which 664 in does not.
  weights       AC_Weight is the structural design weight (MTOW), NOT the cruise
                weight the aero point is trimmed to.

Deliberately NOT overridden: ``Fwd spar X/Z at BL0``. The tool's README makes the
fwd spar the fixed point that the LE/TE move around, and every reference run
carries the same 898.09 / 77.435. Rewriting it from the OAS mesh translates the
whole wing against the gear and cg stations -- a 49 in error when tried.

The ply bounds matter more than any of it: at the PlanL deck's 6-60/50/40 the
inboard bays cannot close on this geometry (11 of 13 groups pinned at their
maxima, margins still -0.165). The reference deck allows 6-100 and closes.
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from studies.vsp_planform.coupling import geometry as wg

# The tool and the reference deck live with the tool, not in a scratch directory.
# WINGCALC_ROOT overrides it, which is how the Python and Rust checkouts are A/B'd
# without editing the study. The default is the rust-kernel tool: its bay evaluator
# is a native extension (``wingcalc_kernel``), and optimization/_rust_bridge.py
# RAISES rather than falling back when it is missing, so a wrong root fails loudly.
WC_ROOT = Path(os.environ.get(
    "WINGCALC_ROOT",
    Path.home() / "Documents" / "Structures-WingCalc_Tool"))

# V3.5.3_ref was a locally-built deck and is not in the repository; a fresh clone
# has V3.5.1-V3.5.4. V3.5.3 is the substitute because the PLY BOUNDS are what
# govern: it allows 6-100 on every group but the aft web, which is what lets the
# inboard bays close on this geometry. The shipped PlanL deck's 6-60/50/40 pins 11
# of 13 groups at their maxima and still leaves margins at -0.165. Anything with
# 6-100 will do; the geometry is overwritten from OAS regardless.
# V3.5.4 before V3.5.3: they differ in exactly one line, and it is the one that
# matters here. V3.5.3 puts the access cut-out's alternative at Stg 9, which on
# this geometry is against a spar in bay 19 -- the sizer raises "bay 19 has
# nowhere to put the access cut-out" and the run dies outboard. V3.5.4 uses
# Stg 8, an interior stringer, and it is also the deck the study's independent
# cross-check was run against. Ply bounds are identical (6-100).
# ONE deck. The per-architecture decks (V3.6.1/2/3 = Arc A/B/C) were redundant: they
# were byte-identical outside OpenVSP/ apart from a single line, "Cut-out alternative
# STG location" (Stg 8 / 2 / 7) -- and resolve_cutout() below already computes that
# line per planform and rewrites it, so the only thing the arc decks encoded was
# something this module derives anyway. V3.5.4 and V3.6.1 were byte-identical outside
# OpenVSP/ entirely.
#
# The deck is a BASELINE plus whatever OAS exports into OpenVSP/; the arc identity
# lives in the export, not in the deck. WINGCALC_DECK overrides the name.
_DECK_CANDIDATES = ("Baseline",)
_DECK_ENV = os.environ.get("WINGCALC_DECK")
WC_DECK = ((WC_ROOT / "Inputs" / _DECK_ENV) if _DECK_ENV else
           next((WC_ROOT / "Inputs" / d for d in _DECK_CANDIDATES
                 if (WC_ROOT / "Inputs" / d).is_dir()),
                WC_ROOT / "Inputs" / _DECK_CANDIDATES[0]))

FRONT_PCT = 0.12
AFT_PCT_SCALAR = 0.750
WINGBOX_SPAN_IN = 1356.0
BASELINE = "const_chord"

def _wing_loading_value(deck, key):
    """One ``Name,Value,Units`` row out of wingLoadingIn.csv."""
    for line in (deck / "wingLoadingIn.csv").read_text(encoding="utf-8-sig").splitlines():
        parts = [c.strip() for c in line.split(",")]
        if parts and parts[0] == key:
            return float(parts[1])
    raise KeyError(f"{key!r} not found in {deck / 'wingLoadingIn.csv'}")


def write_deck(src, dst, mtow_lb, w_wing_lb, oas=None):
    """Copy the deck, update the weights, and re-export the OAS geometry."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    # AC_Weight IS NOT MTOW FOR EVERY CASE, and writing MTOW into every row was
    # wrong. Wing fuel is inertia RELIEF against up-bending, so the critical 2.5 g
    # up case is the one with an EMPTY wing -- and the heaviest aircraft that can
    # have an empty wing is MTOW minus the wing fuel. The deck says exactly that:
    # 81,920 lb on the rows at Fuel_fraction 0 against 86,000 lb at 1, a 4,080 lb
    # difference, which is 2 x the 2,040 lb per-side tank (wing_loading.py applies
    # "Max fuel weight outboard tank" to ONE half wing). Overwriting the lot with
    # MTOW asked for 86,000 lb carried with dry tanks -- a case the aircraft cannot
    # fly, with no fuel relief, so the 2.5 g cases were sized ~5% heavy.
    wing_fuel_lb = 2.0 * _wing_loading_value(dst, "Max fuel weight outboard tank")
    lc = dst / "loadCasesIn.csv"
    rows = list(csv.reader(lc.open(newline="", encoding="utf-8-sig")))
    head = rows[0]
    col = head.index("AC_Weight")
    fcol = head.index("Fuel_fraction")
    seen = {}
    for r in rows[1:]:
        if len(r) > max(col, fcol):
            frac = float(r[fcol])
            w = mtow_lb - (1.0 - frac) * wing_fuel_lb
            r[col] = f"{w:.4f}"
            seen[round(frac, 4)] = w
    with lc.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print("  weights:  " + ", ".join(
        f"Fuel_fraction {f:g} -> {w:.0f} lb" for f, w in sorted(seen.items()))
        + f"  (wing fuel {wing_fuel_lb:.0f} lb)", flush=True)

    wl = dst / "wingLoadingIn.csv"
    out = []
    for line in wl.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("Wing Estimated weight,"):
            line = f"Wing Estimated weight,{w_wing_lb:.4f},lbs"
        out.append(line)
    wl.write_text("\n".join(out) + "\n", encoding="utf-8")

    if oas is not None:
        # planformIn.csv: the box percentages and the wing's fore/aft anchor.
        # README: the fwd spar is the fixed point, so a stale "Fwd spar X at BL0"
        # silently translates the whole wing against the gear and cg stations.
        pf = dst / "planformIn.csv"
        lines = pf.read_text(encoding="utf-8-sig").splitlines()
        # NOTE: "Fwd spar X/Z at BL0" are deliberately NOT touched. The README makes
        # the fwd spar the fixed point that the LE/TE move around, and every
        # reference run (PlanL and all four V3.5.x) carries the same 898.09/77.435.
        # Rewriting it from the OAS mesh translates the whole wing against the
        # gear and cg stations -- a 49 in error when tried.
        # Only the span is still a planformIn key. The spar chord ratios moved OUT of
        # this file and into OAS_Inputs/*_spar.csv, so writing them here did nothing --
        # but it still PRINTED "aft 0.7500" while the spar file said 0.7403, which is
        # a log that contradicts the geometry actually sized. The scalars are gone.
        repl = {
            "Total wingbox span": f"Total wingbox span,Geom,{WINGBOX_SPAN_IN:.1f},in",
        }
        out2 = []
        for line in lines:
            key = line.split(",")[0].strip()
            out2.append(repl.get(key, line))
        pf.write_text("\n".join(out2) + "\n", encoding="utf-8")
        print(f"  planform: wingbox span {WINGBOX_SPAN_IN:.0f} in "
              f"(spar ratios come from the spar file below, not from planformIn)",
              flush=True)

        # OAS_Inputs, not OpenVSP: the tool renamed the folder and now REJECTS a deck
        # carrying the old name (io/oas.py), which is the right call -- nothing in it
        # comes from OpenVSP any more. The Baseline deck ships without the folder at
        # all and waits for this export to create it.
        vsp = dst / "OAS_Inputs"
        if vsp.exists():
            shutil.rmtree(vsp)
        # The spar file goes with the surface. Without it a V3.6 deck falls back to
        # AlternativeInputs/sparRatios.csv, which describes a different box (Arc A
        # ships the OFFSET construction there, 0.750c rising to 0.8044c) -- so the
        # wing would be sized on a box the OAS run never saw.
        from studies.vsp_planform import config as _cfg
        _csv, n, spar, lift = wg.export(
            oas["mesh"], oas["toc"], oas["plate"], oas["stick"],
            vsp, name="OAS_" + BASELINE, max_ws_in=oas["y_junction"],
            front_pct=_cfg.WINGBOX_FRONT_PCT,
            rear_schedule=_cfg.WINGBOX_REAR_SCHEDULE,
            airfoil=oas.get("airfoil"), spanload=oas.get("spanload"))
        print(f"  geometry: {n} stations exported to {vsp.name}/, section "
              f"{oas.get('airfoil') or 'BASELINE LOFT (no design section supplied)'}",
              flush=True)
        print(f"  spanload: "
              f"{lift.name if lift is not None else 'NONE -- WingCalc will use its elliptical fallback'}",
              flush=True)
        print(f"  spar:     {spar.name}, front {_cfg.WINGBOX_FRONT_PCT:.4f}c, "
              f"rear schedule {tuple((float(a), float(b)) for a, b in _cfg.WINGBOX_REAR_SCHEDULE)}",
              flush=True)

    # The access cut-out USED to be resolved here: the deck named one wing-wide
    # stringer pair and no single pair suited every bay of every planform, so this
    # module picked the alternative itself. The tool now derives it internally
    # (geometry/topology.py:_determine_co_stg) and 'Cut-out alternative STG location'
    # is no longer an input, so there is nothing left to resolve. resolve_cutout()
    # is kept below, unused, only until the arc scripts stop importing it.


def _stg_number(text):
    """Parse a wing-wide ``Stg`` number from a planformIn.csv cell."""
    t = str(text).strip().replace("Stg", "").replace("L", "").strip()
    try:
        return int(t)
    except ValueError:
        return None


def _cutout_ok(numbers, k):
    """WingCalc's own rule: the cut-out needs a stringer INTERIOR to this bay.

    Mirrors ``geometry/topology.py:_determine_co_stg``. A stringer with a spar on
    one side of it has no neighbour to carry the cut-out, so only a rung with one
    either side -- ``0 < index < n-1`` -- will do.
    """
    return k is not None and k in numbers and 0 < numbers.index(k) < len(numbers) - 1


def resolve_cutout(deck):
    """Give every bay a legal access cut-out, and say so when one moves.

    The deck names ONE wing-wide stringer pair, default and alternative, and every
    bay must find one of the two interior to its own ladder. Which stringers a bay
    carries depends on which spar marches into the straight stringer family:

      fwd spar sweeps aft   the low numbers run out and the survivors are HIGH.
                            Arc A and Arc C: bay 20 carries Stg 6..9, and the
                            shipped Stg 8 alternative is exactly what covers it.
      fwd spar is STRAIGHT  nothing crosses the family from the front, so the aft
                            spar does all the cutting and the survivors are LOW.
                            Arc B -- the straight-front-spar architecture, whose
                            whole point is a fwd spar that does not move (its
                            tan is 8e-5) -- carries Stg 1..6 by bay 16 and
                            Stg 1..5 by bay 19. Stg 6 is against the aft spar
                            there and Stg 8 does not exist at all, so BOTH halves
                            of the pair are unreachable and the run dies in
                            topology with "bay N has nowhere to put the access
                            cut-out" -- for bays 16..20, not just the first.

    That is a property of the planform, not a mistake in the deck, so it is fixed
    per design here rather than by editing a deck three architectures share. The
    default is never moved: it is the placement the other arcs use inboard, and
    holding it keeps the inboard cut-out comparable across arcs. Only the
    alternative moves, and only to the candidate nearest the default that covers
    every bay the default cannot -- Stg 4 for arc B, which sits at 0.54-0.69 of box
    width across bays 16..20, the same band as arc A/C's Stg 8 outboard (0.51-0.62).

    A cut-out that moves is PRINTED. A hole quietly relocated is a hole that is not
    where the drawing says it is.
    """
    _wingcalc()
    from WingCalc_Tool.geometry.topology import create_topology_context
    from WingCalc_Tool.optimization._sizing_common import _load_build_context, _wing_shell

    ctx = _load_build_context(deck)
    try:
        tc = create_topology_context(_wing_shell(deck, ctx))
    except Exception as exc:
        # The topology would not build for a reason that is NOT the cut-out -- a
        # discontinuous stringer, say. That belongs to the sizer, which raises it
        # where the caller already guards against a sizing failure; raising it from
        # here instead moves it OUT of that guard and costs the caller everything
        # it had already built (compare_wings.py loses a whole figure that way).
        # So decline to act, and let run_wingcalc report it in its own place.
        print(f"  cut-out: topology will not build, leaving the deck alone "
              f"({type(exc).__name__}: {str(exc)[:110]})", flush=True)
        return None
    ladders = [[n for n, _x in L] for L in tc.stg_ladders]
    default = _stg_number(ctx.planform.cutout_default_stg)
    alt = _stg_number(ctx.planform.cutout_alt_stg)
    if default is None:          # an empty default is how a deck asks for no cut-out
        return None

    short = [i for i, L in enumerate(ladders, 1)
             if not (_cutout_ok(L, default) or _cutout_ok(L, alt))]
    if not short:
        print(f"  cut-out: Stg {default} / Stg {alt} covers all {len(ladders)} bays",
              flush=True)
        return None

    # Nearest the default, aft-most on a tie -- the smallest defensible move.
    cands = sorted({n for L in ladders for n in L},
                   key=lambda k: (abs(k - default), -k))
    pick = next((k for k in cands
                 if all(_cutout_ok(ladders[i - 1], k) for i in short)), None)
    if pick is None:
        raise ValueError(
            f"no single alternative cut-out stringer suits bays {short}. They carry "
            + "; ".join(f"bay {i} {ladders[i - 1]}" for i in short)
            + f"; the deck asks for Stg {default} / Stg {alt}. Add a stringer or "
            f"move a spar -- the cut-out cannot be placed on this planform."
        )

    pf = deck / "planformIn.csv"
    lines = pf.read_text(encoding="utf-8-sig").splitlines()
    key = "Cut-out alternative STG location"
    out = [f"{key},Geom,Stg {pick}," if line.split(",")[0].strip() == key else line
           for line in lines]
    pf.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"  cut-out: Stg {default} leaves bays {short} with no interior stringer "
          f"({', '.join(f'bay {i} carries Stg {ladders[i-1][0]}..{ladders[i-1][-1]}' for i in short[:3])}"
          f"{'...' if len(short) > 3 else ''}); alternative Stg {alt} -> Stg {pick}",
          flush=True)
    return pick


def _wingcalc():
    """Import the tool as ``WingCalc_Tool`` regardless of the clone's folder name.

    The package resolves its own name from its directory (``main.py`` sets
    ``__package__`` from it), and every internal import is ``WingCalc_Tool.*``.
    The clone here is ``Structures-WingCalc_Tool``, so a plain path insert gives
    the wrong package name -- bind it explicitly instead of requiring the user to
    rename their checkout or keep a symlink around.
    """
    import importlib.util

    if "WingCalc_Tool" in sys.modules:
        return sys.modules["WingCalc_Tool"]
    spec = importlib.util.spec_from_file_location(
        "WingCalc_Tool", WC_ROOT / "__init__.py",
        submodule_search_locations=[str(WC_ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["WingCalc_Tool"] = mod
    spec.loader.exec_module(mod)
    return mod


def _link_dir(link, target):
    """Point ``link`` at directory ``target``, without needing elevation.

    A real symlink is the first choice, but os.symlink needs SeCreateSymbolicLink,
    which a normal Windows account does not hold unless Developer Mode is on -- it
    fails with WinError 1314. A directory JUNCTION does the same job for an
    absolute local path and needs no privilege, which is why the alias found here
    was one. mklink is a cmd builtin, so it has to run through the shell.
    """
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError as exc:
        if os.name != "nt" or getattr(exc, "winerror", None) != 1314:
            raise
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.lexists(link):
        raise OSError(
            f"could not link {link} -> {target}: symlink needs elevation (WinError "
            f"1314) and mklink /J returned {r.returncode}: "
            f"{(r.stderr or r.stdout).strip()}")


def _alias_dir_for_workers():
    """Make ``WingCalc_Tool`` importable by NAME, for the sizer's spawn workers.

    ``_wingcalc`` binds the package through importlib, which lives only in this
    interpreter. The bay sizer uses a "spawn" pool, so each worker starts a fresh
    interpreter and re-imports the pickled callable's module -- ``WingCalc_Tool``,
    which is not on any path because the clone is named
    ``Structures-WingCalc_Tool``. The workers then die on import in a loop that
    never raises, so the sizing appears to hang.

    A directory holding a correctly-named symlink, exported on PYTHONPATH so the
    children inherit it, is what closes that gap. Nothing is written inside
    either repository.
    """
    if WC_ROOT.name == "WingCalc_Tool":
        return WC_ROOT.parent
    alias = Path(tempfile.gettempdir()) / "wingcalc_pkg_alias"
    alias.mkdir(parents=True, exist_ok=True)
    link = alias / "WingCalc_Tool"

    # The alias OUTLIVES the run and is shared by every checkout, so a link left by a
    # previous session points at whichever tool ran last. Two ways that went wrong:
    # a dangling link (old ~/repos path) survives `exists()` and makes symlink_to
    # raise FileExistsError, and a link to a DIFFERENT but real checkout is worse --
    # the workers then silently import the other tool. Both are fixed by clearing
    # anything that is not already the wanted target, whatever kind of thing it is.
    want = WC_ROOT.resolve()

    # DETECTION HAS TO BE lexists + readlink, not is_symlink/exists. A Windows
    # directory JUNCTION -- which is what a previous session left here, aimed at a
    # Structures-WingCalc_Tool-v0.46.0 checkout that no longer exists -- reports
    # is_symlink() False, islink() False AND exists() False, while still occupying
    # the name. Every guard phrased on those three walks straight past it and
    # symlink_to then fails with WinError 183.
    current = None
    if os.path.lexists(link):
        try:
            current = Path(os.readlink(link).replace("\\?\\", "")).resolve()
        except OSError:
            current = None
    if current != want:
        if os.path.lexists(link):
            for remove in (os.rmdir, os.unlink):   # junction/dir-symlink, then file
                try:
                    remove(link)
                    break
                except OSError:
                    continue
            else:
                shutil.rmtree(link, ignore_errors=True)
        _link_dir(link, WC_ROOT)

    # Verify, because importing the WRONG tool is the failure this whole shim exists
    # to prevent and it is invisible in the results.
    if link.resolve() != want:
        raise RuntimeError(
            f"the worker alias {link} resolves to {link.resolve()}, not {want}. The "
            f"spawn workers would import the wrong WingCalc checkout.")
    return alias


def run_wingcalc(deck, outdir):
    """Size every bay on this deck and return the full wing weight, lb.

    Must be called under ``if __name__ == "__main__"``: the sizer spawns a
    multiprocessing pool, which is also why it has no business inside an
    optimizer loop.
    """
    alias = str(_alias_dir_for_workers())
    existing = os.environ.get("PYTHONPATH", "")
    if alias not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = (alias + os.pathsep + existing) if existing else alias
    # PYTHONPATH ALONE IS NOT ENOUGH, and the failure is silent. A "spawn" child does
    # not build its sys.path from PYTHONPATH: multiprocessing.spawn.get_preparation_data
    # ships the PARENT's sys.path and prepare() assigns it verbatim. This parent binds
    # WingCalc_Tool through importlib, so the alias never reaches sys.path, and every
    # worker dies in `pickle.load` with ModuleNotFoundError inside a pool that keeps
    # restarting it -- the run looks like a hang, not an error. Putting the alias on
    # sys.path here is what actually reaches the workers; PYTHONPATH is kept for any
    # child that is launched as a fresh interpreter instead.
    if alias not in sys.path:
        sys.path.insert(0, alias)
    _wingcalc()
    # Renamed from ``optimize_bay`` in the rust-kernel branch. ``staggered`` is left
    # at its new default (True), which overlaps bays across workers; the tool notes
    # staggered=False is the reference path for comparing against older runs.
    from WingCalc_Tool.main import optimize
    optimize(deck, outdir)
    for row in csv.reader((outdir / "04.Weights" / "wingWeightSummary.csv").open()):
        if row and row[0] == "W_wing":
            return float([x for x in row[1:] if x.strip()][-1])
    raise RuntimeError("W_wing not found")


