"""Both reports, for one arc run, in one folder.

``arc_optimal_toc.py`` already ends by sizing the design in WingCalc, which
writes WingCalc's own report. What was missing is the matching OAS one and
somewhere to look at the two together, which is the whole reason the OAS report
is drawn by WingCalc's frontend in the first place: put them side by side and a
difference is a modelling difference, not a plotting one.

THE SHIM, AND WHY THERE IS ONE
------------------------------
``oas_snapshot.build_snapshot`` reads ``arc_aerostruct``'s four-point JSON, a
LIST of design points carrying a twist floor, an objective and (on newer runs)
the sized skin/spar control points. An ``arc_optimal_toc`` result is a single
dict with none of that vocabulary. Rather than teach the snapshot a second input
format -- two readers for one concept is how they drift apart -- this module
writes the arc run out in the shape the snapshot already reads. The shim is kept
on disk next to the reports rather than in a temporary file, because it is also
the record of exactly which design the OAS report was drawn from.

The box is not sized by ``arc_optimal_toc``: it optimizes drag and hands the
planform to WingCalc. So the snapshot sizes it once against the same 2.5 g case
(roughly twenty seconds) and caches the result beside the shim. That sizing is
what the OAS report's weight, margins and thicknesses are; the aero -- spanload,
drag, twist -- is the arc run's own, unchanged.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from studies.vsp_planform import config
from studies.vsp_planform.viewer.compare import write_compare_page

OAS_REPORT_NAME = "OAS_Wing_Report.html"
COMPARE_NAME = "Compare_WingCalc_vs_OAS.html"
SHIM_NAME = "oas_report_case.json"


def _wingcalc_summary(path):
    """WingCalc's own totals off its wingWeightSummary.csv; empty when there is none."""
    import csv

    path = Path(path)
    if not path.is_file():
        return {}
    out = {}
    for row in csv.reader(path.open(newline="", encoding="utf-8-sig")):
        if row and row[0].startswith(("W_", "k_misc")):
            vals = [x for x in row[1:] if x.strip()]
            if vals:
                try:
                    out[row[0].split(" ")[0]] = float(vals[-1])
                except ValueError:
                    pass
    return out


def _shim_case(result, label):
    """The arc run's converged point, in the vocabulary the snapshot reads."""
    return {
        "label": label,
        # arc_optimal_toc does not vary the twist floor; it runs on the config
        # bounds, so that is what the design point was produced under.
        "twist_lower_deg": float(config.TWIST_BOUNDS[0]),
        "objective": "drag",
        "twist_cp_deg": list(result["twist_cp"]),
        "alpha_deg": float(result["alpha"]),
        "success": bool(result.get("success", False)),
        # Deliberately absent: skin_cp_m / spar_cp_m. arc_optimal_toc sizes in
        # WingCalc, not in OAS, so there is no OAS box yet and the snapshot sizes
        # one. Writing zeros here would look like a sized box and report as one.
    }


class RebuildMismatch(RuntimeError):
    """The OAS model rebuilt from a design JSON is not the design that JSON describes."""


# Relative drag difference above which a rebuilt design is refused. The rebuild is
# the same model on the same inputs, so it reproduces a converged drag to ~1e-9;
# 1e-4 leaves room for solver tolerance and none for a different wing.
REBUILD_DRAG_RTOL = 1e-4


def write_design_reports(case_json, case, seed_json, out_dir, wingcalc_html=None,
                         label="", expect_drag_N=None, created_at=None, quiet=False):
    """The OAS report and the side-by-side page for one design, into ``out_dir``.

    Every study ends here, whichever script produced the design. ``case_json`` is a
    design list in ``arc_aerostruct``'s shape and ``case`` picks one entry;
    ``seed_json`` supplies the frozen geometry.

    ``expect_drag_N`` is the design's own drag. The OAS report is drawn from a
    REBUILT model, and a rebuild that does not reproduce the drag is a different
    wing -- a report of it would look right and describe something nobody designed.
    So a mismatch raises rather than writes. That is the guard that makes it safe
    for any study to call this.
    """
    from studies.vsp_planform.viewer.oas_report import generate_oas_viewer
    from studies.vsp_planform.viewer.oas_snapshot import build_snapshot

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created_at = created_at or datetime.now()
    snapshot = build_snapshot(case_json, case=case, seed_json=seed_json,
                              thickness_cache=out_dir / "oas_report_thickness.json")
    got = float(snapshot["flight"]["drag_N"])
    if expect_drag_N is not None and abs(got / float(expect_drag_N) - 1.0) > REBUILD_DRAG_RTOL:
        raise RebuildMismatch(
            f"the OAS rebuild of '{label}' gives {got:.2f} N of drag against the "
            f"design's {float(expect_drag_N):.2f} N ({100 * (got / expect_drag_N - 1):+.3f}%), "
            "so it is not the same wing; no OAS report written")
    oas_html = generate_oas_viewer(snapshot, out_dir / OAS_REPORT_NAME, created_at=created_at)

    # The headline pair, on the page itself, so the comparison starts before the
    # first click. Both tools' numbers are their own: OAS's from its build-up, and
    # WingCalc's from the summary its run wrote into this same folder -- read, not
    # remembered, so a re-run can never be headlined with a stale constant.
    terms = snapshot.get("weight", {}).get("buildup", {}).get("terms", {})
    subs = [label]
    if terms:
        wc = _wingcalc_summary(out_dir / "04.Weights" / "wingWeightSummary.csv")
        line = f"W_wing  OAS {terms['W_wing']:,.1f} lb"
        if "W_wing" in wc:
            line += f" vs WingCalc {wc['W_wing']:,.1f} lb ({100 * (terms['W_wing'] / wc['W_wing'] - 1):+.2f}%)"
        line += f"   |   box  OAS {terms['W_bays_full']:,.1f} lb"
        if "W_bays_full" in wc:
            line += f" vs WingCalc {wc['W_bays_full']:,.1f} lb"
        subs.append(line)
    compare_html = write_compare_page(
        out_dir / COMPARE_NAME,
        wingcalc_html=wingcalc_html,
        oas_html=oas_html,
        heading=f"WingCalc ↔ OAS  -  {label}",
        subtitle_lines=subs,
        wingcalc_sub="sized structure, eight load cases, ply-level",
        oas_sub="VLM + wingbox FEM, one 2.5 g case",
        created_at=created_at,
    )
    return oas_html, compare_html, snapshot


def write_reports(result, seed_json, out_dir, wingcalc_html=None,
                  label=None, created_at=None, quiet=False):
    """Both reports for an ``arc_optimal_toc``-style result (one design dict).

    Writes the result out in the shape the snapshot reads (the shim), then hands
    it to :func:`write_design_reports`. Returns ``(oas_html, compare_html)``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arc, profile = result.get("arc", "?"), result.get("profile", "?")
    airfoil = result.get("airfoil", "as-built")
    label = label or f"arc {arc} / {profile} / {airfoil}"
    shim = out_dir / SHIM_NAME
    shim.write_text(json.dumps([_shim_case(result, label)], indent=2), encoding="utf-8")
    if not quiet:
        print(f"  OAS report: rebuilding '{label}' and sizing its box against the "
              "same 2.5 g case", flush=True)
    oas_html, compare_html, _snap = write_design_reports(
        shim, 0, seed_json, out_dir, wingcalc_html=wingcalc_html, label=label,
        expect_drag_N=result.get("drag_N"), created_at=created_at, quiet=quiet)
    return oas_html, compare_html
