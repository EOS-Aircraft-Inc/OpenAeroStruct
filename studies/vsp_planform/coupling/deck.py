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
import shutil
import sys
from pathlib import Path

from studies.vsp_planform.coupling import geometry as wg

# The tool and the reference deck live with the tool, not in a scratch directory.
WC_ROOT = Path.home() / "repos" / "Structures-WingCalc_Tool"
WC_DECK = WC_ROOT / "Inputs" / "V3.5.3_ref"

FRONT_PCT = 0.12
AFT_PCT_SCALAR = 0.750
WINGBOX_SPAN_IN = 1356.0
BASELINE = "const_chord"

def write_deck(src, dst, mtow_lb, w_wing_lb, oas=None):
    """Copy the deck, update the weights, and re-export the OAS geometry."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    lc = dst / "loadCasesIn.csv"
    rows = list(csv.reader(lc.open(newline="", encoding="utf-8-sig")))
    head = rows[0]
    col = head.index("AC_Weight")
    for r in rows[1:]:
        if len(r) > col:
            r[col] = f"{mtow_lb:.4f}"
    with lc.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)

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
        repl = {
            "Fwd spar chord ratio": f"Fwd spar chord ratio,Geom,{FRONT_PCT:.4f},",
            "Aft spar chord ratio": f"Aft spar chord ratio,Geom,{AFT_PCT_SCALAR:.4f},",
            "Total wingbox span": f"Total wingbox span,Geom,{WINGBOX_SPAN_IN:.1f},in",
        }
        out2 = []
        for line in lines:
            key = line.split(",")[0].strip()
            out2.append(repl.get(key, line))
        pf.write_text("\n".join(out2) + "\n", encoding="utf-8")
        print(f"  planform: fwd {FRONT_PCT:.4f} / aft {AFT_PCT_SCALAR:.4f}, "
              f"wingbox span {WINGBOX_SPAN_IN:.0f} in", flush=True)

        vsp = dst / "OpenVSP"
        if vsp.exists():
            shutil.rmtree(vsp)
        _csv, n = wg.export(oas["mesh"], oas["toc"], oas["plate"], oas["stick"],
                            vsp, name="OAS_" + BASELINE, max_ws_in=oas["y_junction"])
        print(f"  geometry: {n} stations exported to {vsp.name}/", flush=True)



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


def run_wingcalc(deck, outdir):
    """Size every bay on this deck and return the full wing weight, lb.

    Must be called under ``if __name__ == "__main__"``: the sizer spawns a
    multiprocessing pool, which is also why it has no business inside an
    optimizer loop.
    """
    _wingcalc()
    from WingCalc_Tool.main import optimize_bay
    optimize_bay(deck, outdir)
    for row in csv.reader((outdir / "04.Weights" / "wingWeightSummary.csv").open()):
        if row and row[0] == "W_wing":
            return float([x for x in row[1:] if x.strip()][-1])
    raise RuntimeError("W_wing not found")


