"""The WingCalc wing report's frontend, vendored verbatim so OAS can reuse it.

SOURCE
------
``Structures-WingCalc_Tool/viewer/wing_viewer.py``, tool version **0.63.0**
(``Structures-WingCalc_Tool/version.py``). Copied 2026-09-22.

The whole point of this file is that the two reports look identical, so that a
number that differs between them is a modelling difference and not a plotting
one. It is therefore a COPY and not an import: the OAS study must be able to
write a report with the WingCalc checkout absent, on a different branch, or
several versions ahead. The price is drift, and the way drift gets spotted is
this header plus a diff against the source file named above.

WHAT WAS CHANGED, AND WHY
-------------------------
Every edit is marked ``[OAS]`` in place. In summary:

1. ``_build_html`` is renamed ``build_html`` (it is this module's public entry)
   and takes ``report_title`` / ``subtitle_lines`` instead of hard-coding
   "Wing Report", so the OAS report is obviously the OAS one at a glance while
   the header layout, fonts and tab chrome stay exactly as they are.
2. The **Optimization tab is dropped** -- the button, its panel and its
   JavaScript. WingCalc's is a per-bay differential-evolution generation trace;
   OAS's SLSQP history is a different object and is deferred. The tab's CSS is
   left in place untouched: dropping the content while keeping the styling is
   what keeps this file diffable against its source.
3. ``isResultsTipBayIndex`` always returns false. WingCalc's last ``RESULTS.bays``
   entry is the tip RIB closing the last real bay rather than a bay of its own,
   so its MS callout is suppressed. OAS's entries are FEM elements, every one of
   which is real, so nothing is suppressed.
4. The two rib-line loops read ``ss.ribs`` when the payload carries it. WingCalc
   puts a bay AT each rib station, so ``ss.bays`` doubles as the rib list; OAS's
   bays sit BETWEEN nodes, so the rib stations are a separate array.
5. The Loading tab draws an extra ``markers`` trace wherever the payload carries
   a ``*_bay`` companion array. That is the native-grid-plus-resampled-markers
   comparison this whole report exists for: the line is OAS on its own 34-panel
   grid, the markers are the same quantity resampled onto WingCalc's 20 bay
   stations, so the two reports can be read off one another station by station.
6. ``RESULTS_DETAIL_CSV`` names the OAS arrays a margin came from instead of
   WingCalc's ``03.Results/*.csv``.

Nothing else is touched. In particular the CSS block and the HTML body are
verbatim apart from the Optimization panel, and the data contracts every
``draw*`` function reads are unchanged -- which is why
``studies/vsp_planform/viewer/oas_report.py`` builds WingCalc-shaped dicts
rather than a shape of its own.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# VMT helpers
# ---------------------------------------------------------------------------
_VMT_COMPONENTS = [
    ("Vx", "Axial Load", "lbs"),
    ("Vy", "Chordwise Shear", "lbs"),
    ("Vz", "Vertical Shear", "lbs"),
    ("Mx", "Torque", "in-lbs"),
    ("My", "Bending Moment", "in-lbs"),
    ("Mz", "Chordwise Moment", "in-lbs"),
]

# [AI mod] The report draws in two axis systems, and the axis titles alone do not say
# which one a given plot uses -- "X" is aft on the planform and outboard on a section.
# Every spatial plot therefore names its system in a caption under it. The caption is
# the label only: the sign conventions behind it belong in the code that applies them
# (see loading/wing_loading.py for the VMT signs), not repeated under every chart.
# ---------------------------------------------------------------------------
# Plotly bundle
# ---------------------------------------------------------------------------
# The report carries its own copy of Plotly rather than linking one from a CDN. A linked
# copy leaves the report blank, with no message, on any machine with no route out to the
# internet, and makes an archived report depend on plot.ly still serving that one URL
# years from now -- which a report kept as a design record cannot rely on. plotly-basic
# holds scatter, bar and pie; scatter and bar are every trace drawn here, and it costs
# about a quarter of what the full build would add.

_PLOTLY_BUNDLE = Path(__file__).parent / "vendor" / "plotly-basic.min.js"
# [AI] SHA-256 of plotly-basic.min.js v2.35.2 exactly as vendored. A megabyte of
# minified JavaScript goes into every report verbatim, which makes it the one ingredient
# of a report nobody reads before shipping it. Pinning the digest means a swapped,
# patched or truncated bundle stops the run instead of riding along into a design record.
_PLOTLY_SHA256 = "138c2e81014b979dc00867a93da55b7605a17495ee78dd7afb433b7f021dfcfa"
PLOTLY_PLACEHOLDER = "<!--PLOTLY-BUNDLE-->"


def plotly_script_block() -> str:
    """The vendored Plotly bundle inline, followed by a guard on it having loaded.

    [AI mod] Shared with control_surface_viewer so both reports carry the same offline
    copy rather than one of them linking a CDN.
    """
    if not _PLOTLY_BUNDLE.is_file():
        raise FileNotFoundError(
            f"Plotly bundle not found at {_PLOTLY_BUNDLE}. The report embeds it instead "
            "of linking a CDN copy, so it cannot be written without it -- restore it "
            "with 'git checkout studies/vsp_planform/viewer/vendor/plotly-basic.min.js'."
        )
    raw = _PLOTLY_BUNDLE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _PLOTLY_SHA256:
        raise ValueError(
            f"Plotly bundle at {_PLOTLY_BUNDLE} does not match its pinned SHA-256.\n"
            f"  expected {_PLOTLY_SHA256}\n"
            f"  found    {digest}\n"
            "The report inlines this file verbatim, so it is not written from a copy "
            "whose contents are unknown. Restore it with 'git checkout "
            "studies/vsp_planform/viewer/vendor/plotly-basic.min.js', or update "
            "_PLOTLY_SHA256 if the bundle was deliberately upgraded."
        )
    # A literal '</script>' in the source, in a string constant say, would close the
    # block early. Escaped, JavaScript reads '<\/script' back as the same characters.
    bundle = raw.decode("utf-8").replace("</script", r"<\/script")
    # Every tab builds its plot before its tables, so without Plotly the first call
    # throws and the tab is left blank -- not just chartless. Unreachable now that the
    # bundle is embedded, which is the point: it says so out loud if the file ever
    # reaches a reader truncated, instead of passing for a report with nothing in it.
    guard = """
if (!window.Plotly) {
  document.addEventListener('DOMContentLoaded', function () {
    var bar = document.createElement('div');
    bar.style.cssText = 'background:#c0392b;color:#fff;padding:10px 24px;'
      + 'font:600 13px sans-serif';
    bar.textContent = "This report's embedded plotting library did not load, so its "
      + 'charts and tables cannot be drawn. The file is most likely incomplete: download '
      + 'it again, or rebuild the report with "python -m studies.vsp_planform.viewer.oas_report".';
    document.body.insertBefore(bar, document.body.firstChild);
  });
}
"""
    return f"<script>\n{bundle}\n</script>\n<script>{guard}</script>"


# ---------------------------------------------------------------------------
# Embedding data in the page
# ---------------------------------------------------------------------------
# Every dataset the report draws is written into a <script> block as a JSON literal.
# That block is HTML before it is JavaScript, so what closes it is decided by the HTML
# parser -- which is why the escaping has to happen here, on the way out of Python, and
# not in the page's own escapeHtml() on the way into the DOM.


def json_for_script(obj) -> str:
    r"""``json.dumps`` for a value that lands inside a ``<script>`` block.

    [AI] json.dumps escapes quotes and backslashes but not '<'. A deck whose material,
    stringer or load case name contained "</script>" would therefore close the block
    early and have the rest of its name parsed as markup -- the page's own escapeHtml()
    never gets a say, because the HTML parser has already split the document by then.
    Written as \u003c / \u003e, JSON.parse reads back the same characters, so the block
    can only end where this module ends it. U+2028 and U+2029 are legal inside a JSON
    string but are line terminators to a JavaScript parser, so they go the same way.

    '<' and '>' only ever appear inside strings in json.dumps output -- the structural
    characters are {}[],:" -- so replacing them unconditionally cannot corrupt the JSON.
    """
    return (
        json.dumps(obj)
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("\u2028", r"\u2028")
        .replace("\u2029", r"\u2029")
    )


AXES_NOTE_GLOBAL = "<b>Global axis system</b>"
AXES_NOTE_LOCAL = "<b>Local axis system</b>"
AXES_NOTE_WS = "<b>Wing station (WS)</b>"


_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#393b79", "#637939",
    "#8c6d31", "#843c39", "#7b4173", "#5254a3",
]

# [AI] The control surfaces' colours, imported by the standalone flap and aileron
# viewers as well, so one surface reads as one thing across every report: its planform
# region, its Weight Visuals region, and every cross-section it spans. The interiors
# are deliberately light -- they are laid over the wing planform in one place and over
# the airfoil interior in another -- and the outlines are what carry the shapes.
FLAP_FILL = "rgba(52, 152, 219, 0.10)"
FLAP_LINE = "rgba(52, 152, 219, 0.85)"
AILERON_FILL = "rgba(46, 204, 113, 0.10)"
AILERON_LINE = "rgba(46, 204, 113, 0.85)"

# Keyed by the control_surface_geom surface name, for callers that hold one of those
# rather than a hardcoded surface.
CONTROL_SURFACE_COLORS = {
    "flap": (FLAP_FILL, FLAP_LINE),
    "aileron": (AILERON_FILL, AILERON_LINE),
}

def build_html(planform: dict, airfoil: dict,
               bays: list[dict], vmt_datasets: list[dict],
               loading: dict | None = None,
               results: dict | None = None,
               weight: dict | None = None,
               weight_visuals: dict | None = None,
               output_folder_name: str = "",
               created_at_text: str = "",
               tool_version: str = "",
               report_title: str = "Wing Report") -> str:
    """[OAS] Renamed from ``_build_html``; ``optimization`` dropped, title parameterized.

    Still a PURE function of JSON-serializable dicts, which is the property that
    makes vendoring it worth doing at all: OAS can reach the identical page by
    building the identical dicts, and nothing in here has to know what produced
    them. ``report_title`` is the one cosmetic difference between the two
    reports -- it says out loud which tool wrote the page, while the header
    layout, the fonts and the tab chrome stay exactly as WingCalc draws them.
    """
    planform_json = json_for_script(planform)
    airfoil_json = json_for_script(airfoil)
    bays_json = json_for_script(bays)
    vmt_json = json_for_script(vmt_datasets)
    colors_json = json_for_script(_COLORS)
    components_json = json_for_script(_VMT_COMPONENTS)
    loading_json = json_for_script(loading)
    results_json = json_for_script(results)
    weight_json = json_for_script(weight)
    weight_visuals_json = json_for_script(weight_visuals)
    title_suffix = f" - {html_lib.escape(output_folder_name)}" if output_folder_name else ""
    title_html = html_lib.escape(report_title)
    created_at_html = html_lib.escape(created_at_text)
    tool_version_html = html_lib.escape(tool_version)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title_html}{title_suffix}</title>
<!--PLOTLY-BUNDLE-->
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
/* Height of the fixed chrome above the tab contents (header + warning bar).
   Measured at runtime by syncChromeHeight(); the value here is only a fallback,
   since the header grows with the optional folder / date / version lines. */
:root {{ --chrome-h: 56px; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f5f5f5;
}}
#header {{
  background: #2c3e50; color: white;
  padding: 12px 24px; font-size: 20px; font-weight: 600;
  display: flex; align-items: center; gap: 32px;
}}
.header-title {{
  display: flex; flex-direction: column; line-height: 1.1;
}}
.header-output-folder {{
  font-size: 12px; font-weight: 400; opacity: 0.85; margin-top: 3px;
}}
.tabs {{
  display: flex; gap: 4px;
}}
.tab-btn {{
  background: rgba(255,255,255,0.15); color: white; border: none;
  padding: 8px 20px; cursor: pointer; font-size: 14px; border-radius: 6px 6px 0 0;
  transition: background 0.15s;
}}
.tab-btn:hover {{ background: rgba(255,255,255,0.25); }}
.tab-btn.active {{ background: white; color: #2c3e50; font-weight: 600; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
.negative-margin-warning {{
  display: none;
  padding: 10px 24px;
  background: #fff5f5;
  color: #c0392b;
  border-bottom: 1px solid #e6a2a2;
  font-size: 14px;
  font-weight: 700;
}}
.negative-margin-warning .warning-detail {{
  font-weight: 500;
  margin-left: 8px;
}}

/* Cross-section layout. The tab is a flex column so the toolbar keeps its natural
   height and the body takes exactly what is left, with no hard-coded toolbar size. */
#tab-xsec.active {{
  display: flex; flex-direction: column; height: calc(100vh - var(--chrome-h));
}}
.xsec-toolbar {{
  flex: 0 0 auto;
  display: flex; align-items: center; gap: 28px;
  padding: 12px 24px; background: #fff; border-bottom: 1px solid #ddd;
}}
.xsec-toolbar select {{
  font-size: 14px; padding: 6px 12px; border-radius: 4px;
  border: 1px solid #ccc; min-width: 280px; margin-left: 6px;
}}
#xsec-vmt-lc-select {{ min-width: 220px; }}
/* Axis-scale picker, sitting under its plot. Quieter than the toolbar pickers. */
.scale-bar {{
  flex: 0 0 auto;
  display: flex; align-items: center; justify-content: center;
  padding: 2px 12px 8px; font-size: 12px; font-weight: 400; color: #666;
}}
.scale-bar select {{
  min-width: 0; margin-left: 6px; font-size: 12px; padding: 2px 6px;
  border-radius: 4px; border: 1px solid #ccc; color: #666;
}}
.xsec-body {{
  display: flex; flex: 1 1 auto; min-height: 0;
}}
.xsec-plot-wrap {{
  flex: 2; min-width: 0; display: flex; flex-direction: column;
}}
.xsec-plot {{ flex: 1 1 auto; min-height: 0; }}
.xsec-props {{
  flex: 0 0 520px; padding: 20px; overflow-y: auto;
  background: #fff; border-left: 1px solid #ddd;
}}
.xsec-click-detail {{
  margin-top: 12px; padding: 10px 12px; background: #f8f9fa; border-radius: 6px;
  font-size: 12px; color: #34495e; line-height: 1.5; min-height: 2.8em;
}}
.props-table-compact th {{
  font-size: 12px; padding: 5px 6px 6px;
}}
.props-table-compact td {{
  font-size: 13px; padding: 3px 6px;
}}
.xsec-props h3 {{
  margin-bottom: 12px; font-size: 15px; color: #2c3e50;
  border-bottom: 2px solid #2c3e50; padding-bottom: 6px;
}}
.props-table {{ width: 100%; border-collapse: collapse; }}
.props-table th {{
  text-align: left; font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em;
  color: #7f8c8d; padding: 6px 8px 8px; border-bottom: 2px solid #dfe6e9;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
.props-table td {{
  padding: 5px 8px; font-size: 15px; font-family: 'Consolas', 'Courier New', monospace;
  border-bottom: 1px solid #f0f0f0;
}}
.props-table td:first-child {{ color: #555; font-weight: 600; }}
.props-table td:nth-child(2) {{ text-align: right; }}
.props-table td:nth-child(3) {{ color: #888; font-size: 13px; padding-left: 4px; white-space: nowrap; }}
.props-table td:nth-child(4) {{
  color: #666; font-size: 13px; line-height: 1.35;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-weight: 400;
}}
.props-table tr.spacer td {{ border: none; height: 8px; }}
.props-table td input.val {{
  border: none; background: transparent; text-align: right; width: 100%;
  font-family: inherit; font-size: inherit; color: inherit; outline: none;
  cursor: text;
}}
.props-table td input.val:focus {{ background: #eef5ff; border-radius: 2px; }}

/* Planform split layout */
.planform-body {{
  display: flex;
  height: calc(100vh - var(--chrome-h));
}}
.planform-plot-wrap {{
  flex: 2;
  min-width: 0;
  display: flex;
  flex-direction: column;
}}
#planform-plot {{
  flex: 1;
  min-height: 200px;
  width: 100%;
}}
.planform-props {{
  flex: 0 0 min(560px, 42vw);
  padding: 20px;
  overflow-y: auto;
  background: #fff;
  border-left: 1px solid #ddd;
}}
.planform-props h3 {{
  margin-bottom: 12px;
  font-size: 15px;
  color: #2c3e50;
  border-bottom: 2px solid #2c3e50;
  padding-bottom: 6px;
}}
.planform-props h3:not(:first-child) {{
  margin-top: 22px;
}}
.planform-bay-geom {{
  padding: 20px;
  background: #fff;
  border-top: 1px solid #ddd;
}}
.planform-bay-geom h3 {{
  margin-bottom: 12px;
  font-size: 15px;
  color: #2c3e50;
  border-bottom: 2px solid #2c3e50;
  padding-bottom: 6px;
}}
.planform-bay-geom-scroll {{
  overflow-x: auto;
}}
#planform-bay-geom-table td:first-child,
#planform-bay-geom-table th:first-child {{
  text-align: left;
  position: sticky;
  left: 0;
  background: #f7f9fa;
  white-space: nowrap;
}}
#planform-bay-geom-table td {{
  text-align: right;
  white-space: nowrap;
}}
.props-table-planform td {{
  font-size: 13px;
  padding: 5px 6px;
}}
.props-table-planform td:nth-child(5) {{
  color: #666;
  font-size: 12px;
  line-height: 1.35;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-weight: 400;
}}
.props-table-planform td.category-cell {{
  color: #5d6d7e;
  font-size: 13px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-variant-numeric: tabular-nums;
}}

/* Loading tab: the loading and punctual-load graphs on the left, two
   component-selectable VMT charts on the right. All four plots share one height so
   the two rows fill the viewport; the 200px is the tab's own chrome above and below
   them -- toolbar, block padding and headings. The body scrolls rather than squeezing
   the plots when the viewport is short. */
.loading-body {{
  height: calc(100vh - var(--chrome-h));
  overflow-y: auto;
  padding: 14px 20px 24px;
  background: #f5f5f5;
}}
.loading-empty {{
  padding: 40px 24px;
  color: #888;
  font-size: 14px;
}}
.loading-toolbar {{
  display: flex;
  align-items: center;
  gap: 18px;
  margin-bottom: 12px;
  font-size: 13px;
  color: #2c3e50;
}}
.loading-toolbar select, .loading-plot-block h3 select {{
  font-size: 13px;
  padding: 5px 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #fff;
}}
.loading-layout {{
  display: grid;
  grid-template-columns: minmax(360px, 1fr) minmax(420px, 1.1fr);
  gap: 14px;
  align-items: start;
}}
.loading-col {{
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-width: 0;
}}
.loading-plot-block {{
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 12px 16px 6px;
}}
.loading-plot-block h3 {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
  font-size: 15px;
  color: #2c3e50;
  border-bottom: 2px solid #2c3e50;
  padding-bottom: 6px;
}}
.loading-plot-block h3 select {{
  font-weight: 400;
  margin-left: auto;
}}
.loading-plot {{
  width: 100%;
  height: calc((100vh - var(--chrome-h) - 200px) / 2);
  min-height: 250px;
}}
/* Two columns need roughly 800px before the plots stop being readable. */
@media (max-width: 1180px) {{
  .loading-layout {{ grid-template-columns: 1fr; }}
}}

/* Optimization tab */
.optimization-body {{
  display: flex;
  height: calc(100vh - var(--chrome-h));
}}
.optimization-plot-wrap {{
  flex: 1;
  min-width: 0;
  background: #fff;
  display: flex;
  flex-direction: column;
}}
#optimization-plot {{
  width: 100%;
  flex: 1 1 52%;
  min-height: 240px;
}}
#optimization-xsec-plot {{
  width: 100%;
  flex: 1 1 40%;
  min-height: 220px;
  border-top: 1px solid #ddd;
}}
.optimization-slider {{
  flex: 0 0 auto;
  padding: 8px 14px 12px;
  border-top: 1px solid #eee;
  background: #fafafa;
  font-size: 13px;
}}
.optimization-slider label {{
  display: flex;
  align-items: center;
  gap: 10px;
}}
.optimization-slider input[type="range"] {{
  flex: 1;
}}
#optimization-generation-label {{
  min-width: 120px;
  font-family: 'Consolas', 'Courier New', monospace;
  color: #2c3e50;
}}
.optimization-sidebar {{
  flex: 0 0 min(420px, 36vw);
  padding: 16px;
  overflow-y: auto;
  background: #fff;
  border-left: 1px solid #ddd;
}}
.optimization-sidebar h3 {{
  margin-bottom: 12px;
  font-size: 15px;
  color: #2c3e50;
  border-bottom: 2px solid #2c3e50;
  padding-bottom: 6px;
}}
.optimization-control {{
  margin-bottom: 14px;
  font-size: 13px;
}}
.optimization-control label {{
  display: block;
  margin-bottom: 5px;
  font-weight: 600;
  color: #2c3e50;
}}
.optimization-control select {{
  width: 100%;
  font-size: 14px;
  padding: 6px 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
}}
/* [AI] Whole-wing objective totals shown under the bay selector */
.optimization-totals {{
  margin-bottom: 14px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: #fafafa;
  font-size: 12px;
}}
.optimization-total-row {{
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding: 5px 8px;
  color: #34495e;
}}
.optimization-total-row + .optimization-total-row {{
  border-top: 1px solid #eee;
}}
.optimization-total-value {{
  font-family: 'Consolas', 'Courier New', monospace;
  color: #2c3e50;
}}
.optimization-check-group {{
  margin-top: 12px;
}}
.optimization-check-group h4 {{
  margin-bottom: 6px;
  font-size: 13px;
  color: #2c3e50;
}}
.optimization-checkboxes {{
  max-height: 180px;
  overflow-y: auto;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 6px 8px;
  background: #fafafa;
}}
.optimization-checkboxes label {{
  display: block;
  margin: 4px 0;
  font-size: 12px;
  font-weight: 400;
  color: #34495e;
}}
.optimization-checkboxes input {{
  margin-right: 6px;
}}
.optimization-empty {{
  padding: 32px;
  color: #666;
  font-size: 15px;
}}
.optimization-source {{
  margin-top: 14px;
  font-size: 12px;
  color: #666;
  line-height: 1.4;
}}

/* Weight tab */
.weight-empty {{
  padding: 48px 24px;
  text-align: center;
  color: #7f8c8d;
  font-size: 15px;
}}
.weight-body {{
  height: calc(100vh - var(--chrome-h));
  overflow-y: auto;
  padding: 18px 24px 28px;
  background: #f5f5f5;
}}
.weight-right-column {{
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
}}
.weight-summary-cards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 12px;
}}
.weight-card {{
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 12px 14px;
}}
.weight-card-label {{
  color: #7f8c8d;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 6px;
}}
.weight-card-value {{
  color: #2c3e50;
  font-family: 'Consolas', 'Courier New', monospace;
  font-size: 22px;
  font-weight: 700;
}}
.weight-grid {{
  display: grid;
  /* [AI] The breakdown table is sized to its own content, so the left column is
     max-content: it comes out exactly as wide as that table needs, and every pixel left
     over goes to the spanwise chart and the station table, which are what read better
     wide. Under 1100px the media query below stacks the two panels instead. */
  grid-template-columns: max-content minmax(420px, 1fr);
  gap: 16px;
  align-items: start;
}}
.weight-panel {{
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 16px;
  overflow: hidden;
}}
.weight-panel h3 {{
  margin-bottom: 12px;
  font-size: 15px;
  color: #2c3e50;
  border-bottom: 2px solid #2c3e50;
  padding-bottom: 6px;
}}
.weight-panel h3:not(:first-child) {{
  margin-top: 18px;
}}
/* [AI] The chart's selectors sit on the panel's title line instead of on a row of their
   own: the row is already there, and the panel is tight vertically. The heading border
   every other panel carries moves onto this row so it still runs the full width of the
   panel, with the selectors inside it -- which also puts them directly above the legend
   column the chart now draws in its right margin. */
.weight-plot-head {{
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
  border-bottom: 2px solid #2c3e50;
  padding-bottom: 6px;
}}
.weight-plot-head h3 {{
  margin-bottom: 0;
  padding-bottom: 0;
  border-bottom: none;
}}
.weight-plot-toolbar {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #2c3e50;
}}
.weight-plot-toolbar select {{
  font-size: 13px;
  padding: 5px 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #fff;
}}
#weight-plot {{
  width: 100%;
  /* [AI] Shorter than the other charts in the report, and it can be: the legend is a
     column in the right margin rather than a band under the axis, so what is lost here
     comes off the legend's old row and not off the bars. The floor is the legend, not
     the bars -- see the margins in drawWeightPlot(). */
  height: 350px;
}}
.weight-station-wrap {{
  max-height: 420px;
  overflow: auto;
}}
.weight-table-row-total td {{
  font-weight: 700;
  color: #2c3e50;
  border-top: 2px solid #dfe6e9;
}}
.weight-table-row-subtotal td {{
  font-weight: 600;
  color: #2c3e50;
}}
/* [AI] Sized to its content rather than to the panel: width auto plus nowrap means
   each column is only as wide as its longest entry, instead of Component absorbing all
   the slack a 100%-wide table has to distribute. The right padding is what separates
   the columns, so the last one carries none. */
#weight-breakdown-table {{
  width: auto;
}}
/* [AI mod] Row height is set here rather than left to the default line box: the 25.5px
   a 15px monospace line comes to does not fit the page for 31 rows, and 20px does. The
   value column keeps its 15px -- what is tightened is the leading around it, not the
   number. Set to fill the page rather than to be as short as it can be: the panel beside
   this one is the taller of the two, so the rows are given back what that leaves over. */
#weight-breakdown-table th,
#weight-breakdown-table td {{
  padding: 2px 14px 2px 0;
  line-height: 1.05;
  white-space: nowrap;
}}
#weight-breakdown-table th {{
  padding-bottom: 5px;
}}
#weight-breakdown-table th:last-child,
#weight-breakdown-table td:last-child {{
  padding-right: 0;
}}
#weight-breakdown-table td:nth-child(2) {{
  text-align: left;
  color: #555;
}}
/* Header right-aligned over its numbers, and the weight set larger than the rest of the
   row -- it is what the table is read for. */
#weight-breakdown-table th:nth-child(3) {{
  text-align: right;
}}
#weight-breakdown-table td:nth-child(3) {{
  text-align: right;
  color: #2c3e50;
  font-size: 15px;
  font-weight: 600;
}}
#weight-breakdown-table td:nth-child(4) {{
  font-size: 12px;
  color: #7f8c8d;
}}
/* [AI] The table sets the width of the max-content column it sits in, so the footnote
   under it is capped just inside the table's own width: left free, its max-content
   contribution is the whole citation on one line and it would stretch the column. */
#weight-breakdown-panel .weight-note {{
  max-width: 440px;
}}

/* Weight Visuals tab */
.weightvis-body {{
  height: calc(100vh - var(--chrome-h));
  overflow-y: auto;
  padding: 18px 24px 28px;
  background: #f5f5f5;
}}
.weightvis-toolbar {{
  display: flex;
  align-items: center;
  gap: 18px;
  margin-bottom: 14px;
  font-size: 13px;
  color: #2c3e50;
}}
.weightvis-toolbar select {{
  font-size: 13px;
  padding: 5px 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #fff;
}}
.weightvis-cg-readout {{
  margin-left: auto;
  font-family: 'Consolas', 'Courier New', monospace;
  font-size: 13px;
  color: #2c3e50;
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 8px 12px;
}}
.weightvis-layout {{
  display: grid;
  grid-template-columns: minmax(520px, 1.6fr) minmax(400px, 1fr);
  gap: 16px;
  align-items: start;
}}
.weightvis-plots {{
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-width: 0;
}}
.weightvis-side {{
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 16px;
  min-width: 0;
}}
.weightvis-side h3 {{
  margin-bottom: 12px;
  font-size: 15px;
  color: #2c3e50;
  border-bottom: 2px solid #2c3e50;
  padding-bottom: 6px;
}}
.weightvis-station-picker {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #2c3e50;
  margin-bottom: 12px;
}}
.weightvis-station-picker select {{
  font-size: 13px;
  padding: 5px 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #fff;
}}
.weightvis-hint {{
  color: #7f8c8d;
  font-size: 12px;
}}
.weightvis-station-summary {{
  font-family: 'Consolas', 'Courier New', monospace;
  font-size: 13px;
  color: #2c3e50;
  background: #f7f9fa;
  border: 1px solid #e1e6e8;
  border-radius: 4px;
  padding: 10px 12px;
  margin-bottom: 12px;
  line-height: 1.6;
}}
.weightvis-table-wrap {{
  max-height: 620px;
  overflow: auto;
}}
.weightvis-kind-dot {{
  display: inline-block;
  width: 9px;
  height: 9px;
  border-radius: 50%;
  margin-right: 6px;
  vertical-align: middle;
}}
.weightvis-plot-block {{
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 16px;
}}
.weightvis-plot-block h3 {{
  margin-bottom: 12px;
  font-size: 15px;
  color: #2c3e50;
  border-bottom: 2px solid #2c3e50;
  padding-bottom: 6px;
}}
#weightvis-top-plot {{
  width: 100%;
  height: 460px;
}}
#weightvis-aft-plot {{
  width: 100%;
  height: 340px;
}}
/* Bay / WS / rib type read as labels; every column after them is a weight. The
   column count follows the scope selector, so these are positional, not per-column.

   [AI] Fixed layout with a width on the three label columns only: the weight columns
   have none, so the browser shares what is left equally between them however many the
   scope selector asks for. That is what makes them one width and keeps the numbers under
   one another; the headers wrap instead of setting the width the way they used to. */
#weight-station-table {{
  table-layout: fixed;
  width: 100%;
}}
/* [AI] Fourteen columns of one width leave about 50px for a name, so these headers drop
   the capitals and the letter spacing the rest of the report's tables use -- mixed case
   at 11px fits "Stringers" where capitals do not. Nothing may run into the column beside
   it: a name too long for the width wraps, and a single long word breaks. */
#weight-station-table th {{
  text-align: center;
  vertical-align: bottom;
  line-height: 1.15;
  letter-spacing: 0;
  text-transform: none;
  font-size: 11px;
  font-weight: 600;
  color: #5d6d7e;
  padding: 4px 3px 5px;
  white-space: normal;
  overflow-wrap: break-word;
}}
#weight-station-table td {{
  padding: 2px 4px;
}}
#weight-station-table th:nth-child(1) {{ width: 40px; }}
#weight-station-table th:nth-child(2) {{ width: 58px; }}
#weight-station-table th:nth-child(3) {{ width: 100px; text-align: left; }}
#weight-station-table th:nth-child(1),
#weight-station-table td:nth-child(1),
#weight-station-table th:nth-child(2),
#weight-station-table td:nth-child(2) {{
  text-align: center;
  color: #333;
  font-size: 13px;
}}
#weight-station-table td:nth-child(3) {{
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
#weight-station-table th:nth-child(n+4),
#weight-station-table td:nth-child(n+4) {{
  text-align: center;
  color: #333;
  font-size: 13px;
}}
#weight-station-table td:last-child {{
  font-weight: 700;
  color: #2c3e50;
}}
/* The units row belongs to the header, so it is set like one rather than like data. */
#weight-station-table tr.weight-station-units td {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 11px;
  color: #95a5a6;
  text-align: center;
  padding: 0 4px 4px;
  border-bottom: 2px solid #dfe6e9;
}}
.weight-note {{
  margin-top: 10px;
  color: #666;
  font-size: 12px;
  line-height: 1.4;
}}
/* [AI] Caption naming the axis system a plot is drawn in. Deliberately quiet: it is
   a label on the plot, not a finding. flex:0 0 auto so it never squeezes the plot
   inside the column-flex wrappers, matching .scale-bar. */
.coord-note {{
  flex: 0 0 auto;
  padding: 4px 12px 8px;
  font-size: 11px;
  color: #8a949e;
  line-height: 1.35;
}}
.coord-note b {{ color: #5d6d7e; font-weight: 600; }}
/* [AI] Each caption names the axis system of the chart it belongs to, so it cannot move
   to the toolbar the way the scale selector did -- the two charts in this tab are on
   different systems. It is laid over the bottom-left of its own chart instead: this
   column is height bound, and a caption band under each chart costs the cross-section
   room it can spend on the section. The corner is clear on both charts, left of the y
   tick labels and below the x axis line. */
.results-zone1, .results-zone2 {{ position: relative; }}
.coord-note-overlay {{
  position: absolute;
  left: 6px;
  bottom: 4px;
  padding: 0;
  z-index: 2;
  pointer-events: none;   /* never in the way of a click or hover on the plot behind */
}}
@media (max-width: 1100px) {{
  .weight-grid {{
    grid-template-columns: 1fr;
  }}
}}

/* Results tab */
.results-body {{
  display: flex;
  flex-direction: column;
  height: calc(100vh - var(--chrome-h));
  overflow: hidden;
}}
/* Same toolbar treatment as the cross-section tab. */
.results-lc-bar {{
  flex: 0 0 auto;
  display: flex; align-items: center; gap: 28px;
  padding: 12px 24px; background: #fff; border-bottom: 1px solid #ddd;
  font-size: 14px;
}}
.results-lc-bar select {{
  font-size: 14px; padding: 6px 12px; border-radius: 4px;
  border: 1px solid #ccc; min-width: 280px; margin-left: 6px;
}}
#results-lc-select {{ max-width: min(480px, 60vw); }}
.results-content-row {{
  display: flex;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}}
.results-main {{
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  border-right: 1px solid #ddd;
}}
/* [AI mod] The overview is a fixed-content schematic -- one band of bay boxes across the
   span -- so extra height does nothing for it, while the cross-section under it is height
   bound: its axes are locked 1:1 and the section is far wider than it is deep. The share
   is therefore both cut and capped, so past a point every further pixel of window height
   goes to the cross-section rather than to more white space up here. */
.results-zone1 {{
  flex: 0 0 32%;
  min-height: 150px;
  max-height: 260px;
  display: flex;
  flex-direction: column;
  background: #fff;
  border-bottom: 1px solid #ddd;
}}
#results-overview-plot {{
  flex: 1;
  min-height: 100px;
}}
.results-zone2 {{
  flex: 1;
  min-height: 200px;
  display: flex;
  flex-direction: column;
  background: #fff;
}}
#results-xsec-plot {{
  flex: 1;
  min-height: 0;
}}
.results-sidebar {{
  flex: 0 0 min(508px, 43.6vw);
  overflow-y: auto;
  background: #fff;
  padding: 16px;
}}
.results-sidebar h3 {{
  margin-bottom: 10px;
  font-size: 14px;
  color: #2c3e50;
  border-bottom: 2px solid #2c3e50;
  padding-bottom: 4px;
}}
.results-sidebar h3:not(:first-child) {{ margin-top: 20px; }}
.results-empty {{
  padding: 48px 24px;
  text-align: center;
  color: #7f8c8d;
  font-size: 15px;
}}
.ms-neg {{ color: #c0392b; font-weight: 600; }}
.ms-low {{ color: #d68910; font-weight: 600; }}
.ms-ok {{ color: #1e8449; }}
.results-meta {{
  font-size: 12px;
  color: #666;
  margin-bottom: 12px;
}}
.results-detail-placeholder {{
  font-size: 14px;
  color: #888;
  font-style: italic;
  padding: 8px 0;
}}
.results-detail-source {{
  font-size: 13px;
  color: #5d6d7e;
  line-height: 1.45;
  margin-bottom: 10px;
  padding: 6px 0 4px;
  display: none;
}}
.results-detail-source.is-visible {{
  display: block;
}}
.results-detail-source code {{
  font-family: 'Consolas', 'Courier New', monospace;
  font-size: 12px;
  background: #f4f6f7;
  padding: 1px 5px;
  border-radius: 3px;
}}
.results-xsec-summary {{
  margin-bottom: 12px;
  font-size: 14px;
}}
.results-min-ms-line {{
  margin-bottom: 10px;
  font-size: 16px;
  font-weight: 600;
}}
.results-cs-table {{
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 8px;
  font-size: 16px;
  table-layout: fixed;
}}
.results-cs-table col.results-cs-col-ms {{ width: 28%; }}
.results-cs-table col.results-cs-col-loc {{ width: 20%; }}
.results-cs-table col.results-cs-col-lc {{ width: 28%; }}
.results-cs-table col.results-cs-col-mode {{ width: 24%; }}
.results-overall-summary {{
  margin-bottom: 16px;
}}
.results-selected-lc-summary {{
  margin-bottom: 16px;
}}
.results-sidebar h3.results-selected-lc-title {{
  margin-top: 20px;
}}
.results-cs-table-overall col.results-cs-col-ms {{ width: 14%; }}
.results-cs-table-overall col.results-cs-col-bay {{ width: 10%; }}
.results-cs-table-overall col.results-cs-col-loc {{ width: 24%; }}
.results-cs-table-overall col.results-cs-col-lc {{ width: 28%; }}
.results-cs-table-overall col.results-cs-col-mode {{ width: 24%; }}
.results-sidebar h3.results-overall-title {{
  margin-top: 0;
}}
.results-cs-table tr.results-cs-h th {{
  text-align: left;
  font-weight: 600;
  color: #7f8c8d;
  padding: 6px 6px 2px 0;
  vertical-align: bottom;
}}
.results-cs-table tr.results-cs-h:first-child th {{
  padding-top: 0;
}}
.results-cs-table tr.results-cs-d td {{
  padding: 0 6px 8px 0;
  vertical-align: top;
  color: #34495e;
  word-wrap: break-word;
}}
.results-cs-table tr.results-cs-d:last-child td {{
  padding-bottom: 0;
}}
.results-cs-ms {{
  font-family: Consolas, 'Courier New', monospace;
  font-weight: 600;
}}
</style>
</head>
<body>

<div id="header">
  <div class="header-title">
    <span>{title_html}</span>
    {f'<span class="header-output-folder">{html_lib.escape(output_folder_name)}</span>' if output_folder_name else ''}
    {f'<span class="header-output-folder">{created_at_html}</span>' if created_at_html else ''}
    {f'<span class="header-output-folder">Tool version {tool_version_html}</span>' if tool_version_html else ''}
  </div>
  <div class="tabs">
    <button class="tab-btn active" data-tab="planform" onclick="switchTab('planform', this)">Planform</button>
    <button class="tab-btn" data-tab="xsec" onclick="switchTab('xsec', this)">Cross section &amp; VMT</button>
    <button class="tab-btn" data-tab="loading" onclick="switchTab('loading', this)">Loading</button>
    <!-- [OAS] The Optimization tab button is dropped. WingCalc's is a per-bay
         differential-evolution generation trace read out of optimization_details.csv;
         OAS's driver history is an SLSQP design-vector trace, a different object with
         no per-bay axis, and it is deferred rather than faked into this shape. The
         tab's CSS is left in place so this file still diffs cleanly against its
         source. -->
    <button class="tab-btn" data-tab="results" onclick="switchTab('results', this)">Stress Results</button>
    <button class="tab-btn" data-tab="weightvis" onclick="switchTab('weightvis', this)">Weight Visuals</button>
    <button class="tab-btn" data-tab="weight" onclick="switchTab('weight', this)">Weight Summary</button>
  </div>
</div>

<div id="negative-margin-warning" class="negative-margin-warning"></div>

<div id="tab-planform" class="tab-content active">
  <div class="planform-body">
    <div class="planform-plot-wrap">
      <div id="planform-plot"></div>
      <div class="coord-note">{AXES_NOTE_GLOBAL}</div>
    </div>
    <div class="planform-props">
      <h3>Planform overview</h3>
      <table class="props-table" id="planform-overview-table"></table>
      <h3>Planform inputs</h3>
      <table class="props-table props-table-planform" id="planform-props-table"></table>
    </div>
  </div>
  <div class="planform-bay-geom">
    <h3>Per-bay taper and sweep</h3>
    <div class="planform-bay-geom-scroll">
      <table class="props-table props-table-compact" id="planform-bay-geom-table"></table>
    </div>
  </div>
</div>

<div id="tab-xsec" class="tab-content">
  <div class="xsec-toolbar">
    <label><b>Bay:</b>
      <select id="bay-select" onchange="updateXsec()"></select>
    </label>
    <label><b>Load case:</b>
      <select id="xsec-vmt-lc-select" onchange="drawXsecVmtTable()"></select>
    </label>
  </div>
  <div class="xsec-body">
    <div class="xsec-plot-wrap">
      <div id="xsec-plot" class="xsec-plot"></div>
      <div class="coord-note">{AXES_NOTE_LOCAL}</div>
      <div class="scale-bar">
        <label>Scale:
          <select id="xsec-scale-select" onchange="updateXsec()">
            <option value="adaptive">Adaptive</option>
            <option value="fixed">Fixed (Bay 1)</option>
          </select>
        </label>
      </div>
    </div>
    <div class="xsec-props">
      <!-- [AI] "(Limit Load)" because this table interpolates the VMT CSVs, which
           loading/wing_loading.py writes at limit: the case's ult_factor is applied
           only in ultimate_loads_at_station, downstream of these arrays. -->
      <h3>VMT at this cross section (Limit Load)</h3>
      <table class="props-table props-table-compact" id="xsec-vmt-table"></table>
      <h3 style="margin-top:16px">Cross Section Properties</h3>
      <table class="props-table" id="props-table"></table>
      <div id="xsec-click-detail" class="xsec-click-detail">Click a structural item on the plot (skin, stringer, spar) for details.</div>
      <h3 style="margin-top:16px">Section components</h3>
      <table class="props-table props-table-compact" id="xsec-components-table"></table>
    </div>
  </div>
</div>

<div id="tab-loading" class="tab-content">
  <div id="loading-empty" class="loading-empty" style="display:none">
    No load cases available. Run the tool with a load case deck to populate the Loading tab.
  </div>
  <div id="loading-body" class="loading-body" style="display:none">
    <div class="loading-toolbar">
      <label><b>Load case:</b>
        <select id="loading-lc-select" onchange="drawLoading()"></select>
      </label>
    </div>
    <div class="loading-layout">
      <div class="loading-col">
        <div class="loading-plot-block">
          <h3>Loading distribution
            <select id="loading-aero-select" onchange="drawLoadingAero()">
              <option value="lift">Lift (lbs/in)</option>
              <option value="drag">Drag (lbs/in)</option>
              <option value="cl">Cl norm. (section coefficient shape)</option>
              <option value="cd">Cd norm. (section coefficient shape)</option>
            </select>
          </h3>
          <div id="loading-aero-plot" class="loading-plot"></div>
        </div>
        <div class="loading-plot-block">
          <h3>Weight and punctual loading</h3>
          <div id="loading-weight-plot" class="loading-plot"></div>
        </div>
      </div>
      <div class="loading-col">
        <div class="loading-plot-block">
          <h3>VMT
            <select id="loading-vmt-a-select" onchange="drawLoadingVmt('a')"></select>
          </h3>
          <div id="loading-vmt-a-plot" class="loading-plot"></div>
        </div>
        <div class="loading-plot-block">
          <h3>VMT
            <select id="loading-vmt-b-select" onchange="drawLoadingVmt('b')"></select>
          </h3>
          <div id="loading-vmt-b-plot" class="loading-plot"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- [OAS] The Optimization tab's panel is dropped with its button; see the note on
     the tab bar above. Its CSS block is deliberately kept, so the only difference
     from the source file is the content. -->
<div id="tab-weight" class="tab-content">
  <div id="weight-empty" class="weight-empty" style="display:none">
    Run wing weight calculation to populate Weight Summary.
  </div>
  <div id="weight-body" class="weight-body" style="display:none">
    <div class="weight-grid">
      <div class="weight-panel" id="weight-breakdown-panel">
        <h3>Weight Breakdown</h3>
        <table class="props-table props-table-compact" id="weight-breakdown-table"></table>
        <!-- [OAS] WingCalc's footnote cites Torenbeek, which is where most of its
             secondary-structure correlations come from. Nothing on this page uses one:
             OAS sizes the box from its own FEM and everything else is held constant, so
             the footnote names the two things a reader of this table has to know
             instead. -->
        <div class="weight-note">
          The box is sized by OpenAeroStruct's own wingbox FEM against one 2.5 g load
          case. <b>F_box</b> ties it to WingCalc's <code>W_bays</code> at the shipped
          Arc A point, where the two agree to 0.26%. The <b>fixed remainder</b> is
          WingCalc's <code>W_wing</code> less its <code>W_bays</code>, carried as a
          constant because span, taper and the chord distribution are all frozen in this
          run — it is not computed here.
        </div>
      </div>
      <div class="weight-right-column">
        <div id="weight-summary-cards" class="weight-summary-cards"></div>
        <div class="weight-panel">
        <div class="weight-plot-head">
          <h3>Spanwise Wing Weight</h3>
          <div class="weight-plot-toolbar">
            <label for="weight-plot-scope"><b>Scope:</b></label>
            <!-- [OAS] The "Whole wing" option is dropped, not disabled. OAS sizes the
                 wingbox and nothing else; the rest of the calibrated wing weight is a
                 single constant with no spanwise distribution behind it, so a whole-wing
                 scope would either redraw the wingbox bars under a different name or
                 smear a number across stations it was never resolved on. The selector
                 stays, with its one honest choice, so the toolbar reads the same. -->
            <select id="weight-plot-scope" onchange="updateWeightPlot()">
              <option value="wingbox">Wingbox only</option>
            </select>
            <label for="weight-plot-mode"><b>Display:</b></label>
            <select id="weight-plot-mode" onchange="updateWeightPlot()">
              <option value="weight">Weight (lbs)</option>
              <option value="linear">Linear weight (lbs/in span)</option>
            </select>
          </div>
        </div>
        <div id="weight-plot"></div>
        <div class="coord-note">{AXES_NOTE_WS}</div>
        <div class="weight-note" id="weight-plot-note"></div>
        <div class="weight-station-wrap">
          <table class="props-table props-table-compact" id="weight-station-table"></table>
        </div>
        </div>
      </div>
    </div>
  </div>
</div>

<div id="tab-weightvis" class="tab-content">
  <div id="weightvis-empty" class="weight-empty" style="display:none">
    Run wing weight calculation to populate Weight Visuals.
  </div>
  <div id="weightvis-body" class="weightvis-body" style="display:none">
    <div class="weightvis-toolbar">
      <label><b>Weight grouped:</b>
        <!-- [OAS] "Per rib" is dropped and "per bay" renamed: a bay here is an FEM
             element between two mesh stations and there are no ribs in the model at
             all, so a rib grouping has nothing to group and the old label named a
             part this wing does not carry. -->
        <select id="weightvis-group-mode" onchange="onWeightVisGroupChange()">
          <option value="bay">Per FEM element (cg between mesh stations)</option>
        </select>
      </label>
      <label><b>Marker size:</b>
        <select id="weightvis-size-mode" onchange="drawWeightVisuals()">
          <option value="weight">Scaled by weight</option>
          <option value="uniform">Uniform</option>
        </select>
      </label>
      <label><b>Looking-aft scale:</b>
        <select id="weightvis-aft-scale" onchange="drawWeightVisuals()">
          <option value="stretched">Z exaggerated</option>
          <option value="true">True scale (1:1)</option>
        </select>
      </label>
      <div id="weightvis-cg-readout" class="weightvis-cg-readout"></div>
    </div>
    <div class="weight-note" id="weightvis-group-note"></div>
    <div class="weightvis-layout">
      <div class="weightvis-plots">
        <div class="weightvis-plot-block">
          <h3>Top down view (right wing)</h3>
          <div id="weightvis-top-plot"></div>
          <div class="coord-note">{AXES_NOTE_GLOBAL}</div>
        </div>
        <div class="weightvis-plot-block">
          <h3>Looking aft view (right wing)</h3>
          <div id="weightvis-aft-plot"></div>
          <div class="coord-note">{AXES_NOTE_GLOBAL}</div>
          <div class="weight-note" id="weightvis-aft-note"></div>
        </div>
      </div>
      <div class="weightvis-side">
        <h3 id="weightvis-side-title">Bay contributions</h3>
        <div class="weightvis-station-picker">
          <label for="weightvis-station-select"><b id="weightvis-station-label">Bay:</b></label>
          <select id="weightvis-station-select" onchange="onWeightVisStationSelect()"></select>
          <span class="weightvis-hint" id="weightvis-station-hint">or click a bay cg on a plot</span>
        </div>
        <div id="weightvis-station-summary" class="weightvis-station-summary"></div>
        <div class="weightvis-table-wrap">
          <table class="props-table props-table-compact" id="weightvis-component-table"></table>
        </div>
      </div>
    </div>
  </div>
</div>

<div id="tab-results" class="tab-content">
  <div id="results-empty" class="results-empty" style="display:none">
    Run margin-of-safety calculation (load cases required) to populate Results.
  </div>
  <div id="results-body" class="results-body" style="display:none">
    <div class="results-lc-bar" id="results-lc-bar">
      <label><b>Bay:</b>
        <select id="results-bay-select" onchange="onResultsBayChange()"></select>
      </label>
      <label><b>Load case:</b>
        <select id="results-lc-select" onchange="onResultsLoadcaseChange()"></select>
      </label>
      <label><b>Section scale:</b>
        <select id="results-scale-select" onchange="onResultsScaleChange()">
          <option value="adaptive">Adaptive</option>
          <option value="fixed">Fixed (Bay 1)</option>
        </select>
      </label>
    </div>
    <div class="results-content-row">
    <div class="results-main">
      <div class="results-zone1">
        <div id="results-overview-plot"></div>
        <div class="coord-note coord-note-overlay">{AXES_NOTE_GLOBAL}</div>
      </div>
      <div class="results-zone2">
        <div id="results-xsec-plot"></div>
        <div class="coord-note coord-note-overlay">{AXES_NOTE_LOCAL}</div>
      </div>
    </div>
    <div class="results-sidebar">
      <div id="results-meta" class="results-meta"></div>
      <h3 class="results-overall-title">Overall Results</h3>
      <div id="results-overall-summary" class="results-overall-summary"></div>
      <h3 id="results-selected-lc-title" class="results-selected-lc-title">Selected loadcase results</h3>
      <div id="results-selected-lc-summary" class="results-selected-lc-summary"></div>
      <h3 id="results-xsec-title">Cross section results</h3>
      <div id="results-xsec-summary" class="results-xsec-summary"></div>
      <h3>Component detail</h3>
      <div id="results-detail-placeholder" class="results-detail-placeholder">
        Hover a component for a preview; click to pin details here.
      </div>
      <div id="results-detail-source" class="results-detail-source"></div>
      <table class="props-table" id="results-detail-table"></table>
    </div>
    </div>
  </div>
</div>

<script>
// ---- Embedded data ----
var PLANFORM = {planform_json};
var AIRFOIL  = {airfoil_json};
var BAYS     = {bays_json};
var VMT_DATA = {vmt_json};
var COLORS   = {colors_json};
var VMT_COMP = {components_json};
var RESULTS  = {results_json};
// [OAS] var OPTIMIZATION dropped with the tab.
var WEIGHT = {weight_json};
var WEIGHTVIS = {weight_visuals_json};
var LOADING = {loading_json};

// ---- Tab switching ----
var currentTab = 'planform';
var plotInitialized = {{ planform: false, xsec: false, loading: false, weight: false, weightvis: false, results: false }};

// Tab heights are viewport height minus the chrome above them. That chrome varies
// with the optional header lines, the warning bar, and header wrapping at narrow
// widths, so measure it instead of hard-coding an offset. Hidden elements report 0.
function syncChromeHeight() {{
  var header = document.getElementById('header');
  var warning = document.getElementById('negative-margin-warning');
  var h = (header ? header.offsetHeight : 0) + (warning ? warning.offsetHeight : 0);
  document.documentElement.style.setProperty('--chrome-h', h + 'px');
}}

function resizeCurrentTabPlots() {{
  if (currentTab === 'results') {{
    ['results-overview-plot', 'results-xsec-plot'].forEach(function(id) {{
      var el = document.getElementById(id);
      if (el && el.data) Plotly.Plots.resize(el);
    }});
    // [AI] The callout offsets are in pixels, so a resize changes which of them collide.
    // Re-spread from the as-built layout: below the size thresholds this puts the boxes
    // back where they started, which is what a window shrunk past them should do.
    var xsecEl = document.getElementById('results-xsec-plot');
    if (xsecEl && xsecEl.data) spreadResultsXsecAnnotations(xsecEl);
    return;
  }}
  // [OAS] The Optimization branch is dropped with the tab.
  if (currentTab === 'weight') {{
    var weightEl = document.getElementById('weight-plot');
    if (weightEl && weightEl.data) Plotly.Plots.resize(weightEl);
    return;
  }}
  if (currentTab === 'weightvis') {{
    ['weightvis-top-plot', 'weightvis-aft-plot'].forEach(function(id) {{
      var el = document.getElementById(id);
      if (el && el.data) Plotly.Plots.resize(el);
    }});
    return;
  }}
  if (currentTab === 'loading') {{
    ['loading-aero-plot', 'loading-weight-plot',
     'loading-vmt-a-plot', 'loading-vmt-b-plot'].forEach(function(id) {{
      var el = document.getElementById(id);
      if (el && el.data) Plotly.Plots.resize(el);
    }});
    return;
  }}
  var el = document.getElementById(currentTab + '-plot');
  if (el && el.data) Plotly.Plots.resize(el);
}}

function switchTab(name, btn) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  if (btn) btn.classList.add('active');
  currentTab = name;
  if (!plotInitialized[name]) {{
    if (name === 'planform') initPlanform();
    else if (name === 'xsec') initXsec();
    else if (name === 'loading') initLoading();
    else if (name === 'weight') initWeight();
    else if (name === 'weightvis') initWeightVisuals();
    else if (name === 'results') initResults();
    plotInitialized[name] = true;
  }}
  resizeCurrentTabPlots();
}}

// ---- Planform ----
function formatPlanformRowValue(r) {{
  if (r.kind === 'str') return String(r.value);
  if (r.kind === 'int') return String(r.value);
  return Number(r.value).toFixed(Number(r.decimals));
}}

function drawPlanformOverviewTable() {{
  var rows = PLANFORM.overview_rows || [];
  var html = '<thead><tr>'
    + '<th>Property</th><th>Value</th><th>Units</th>'
    + '</tr></thead><tbody>';

  rows.forEach(function(r) {{
    var valStr = escapeHtml(formatPlanformRowValue(r));
    html += '<tr>'
      + '<td>' + escapeHtml(r.property) + '</td>'
      + '<td><input class="val" readonly value="' + valStr + '" '
      +   'onclick="this.select()" title="Click to select"></td>'
      + '<td>' + escapeHtml(r.units) + '</td>'
      + '</tr>';
  }});

  html += '</tbody>';
  var el = document.getElementById('planform-overview-table');
  if (el) el.innerHTML = html;
}}

function drawPlanformInputsTable() {{
  var rows = PLANFORM.input_rows || [];
  var html = '<thead><tr>'
    + '<th>Property</th><th>Value</th><th>Units</th>'
    + '<th>Category</th><th>Description</th>'
    + '</tr></thead><tbody>';

  rows.forEach(function(r) {{
    var valStr = escapeHtml(formatPlanformRowValue(r));
    var desc = escapeHtml(r.description);

    html += '<tr>'
      + '<td>' + escapeHtml(r.property) + '</td>'
      + '<td><input class="val" readonly value="' + valStr + '" '
      +   'onclick="this.select()" title="Click to select"></td>'
      + '<td>' + escapeHtml(r.units) + '</td>'
      + '<td class="category-cell">' + escapeHtml(r.category) + '</td>'
      + '<td>' + desc + '</td>'
      + '</tr>';
  }});

  html += '</tbody>';
  var el = document.getElementById('planform-props-table');
  if (el) el.innerHTML = html;
}}

function drawPlanformBayGeomTable() {{
  var el = document.getElementById('planform-bay-geom-table');
  if (!el) return;
  var data = PLANFORM.bay_geometry || {{ columns: [], rows: [] }};
  var cols = data.columns || [];
  var rows = data.rows || [];
  if (!cols.length) {{
    el.innerHTML = '<tbody><tr><td>No bay geometry available.</td></tr></tbody>';
    return;
  }}

  var head = '<thead><tr><th>Quantity</th>';
  cols.forEach(function(c) {{
    head += '<th title="WS ' + formatNumber(c.ws_inboard, 1)
      + ' &rarr; ' + formatNumber(c.ws_outboard, 1) + ' in">Bay ' + c.bay_id + '</th>';
  }});
  head += '</tr></thead>';

  var body = '<tbody>';
  var wsRow = '<tr><td>WS range (in)</td>';
  cols.forEach(function(c) {{
    wsRow += '<td>' + formatNumber(c.ws_inboard, 1) + '&ndash;'
      + formatNumber(c.ws_outboard, 1) + '</td>';
  }});
  wsRow += '</tr>';
  body += wsRow;

  rows.forEach(function(r) {{
    var tr = '<tr><td>' + escapeHtml(r.label) + '</td>';
    (r.values || []).forEach(function(v) {{
      tr += '<td>' + formatNumber(v, r.decimals) + '</td>';
    }});
    tr += '</tr>';
    body += tr;
  }});
  body += '</tbody>';

  el.innerHTML = head + body;
}}

function planformLineColor(fillColor) {{
  return fillColor.replace(/0\\.\\d+\\)/, '0.85)');
}}

// Every control-surface region, for the planform, the Weight Visuals semispan --
// which used to carry their own identical copies of this list -- and the flap and
// aileron drawn on the cross-sections. The colours are injected from the Python side
// and shared with the standalone flap and aileron viewers, so one surface is one
// colour everywhere it appears. All the fills are faint on purpose: they are washes
// laid over the wing planform in one place and over the airfoil interior in another,
// and it is each region's outline -- the same colour at 0.85, from planformLineColor
// -- that actually carries its shape.
var SURFACE_SPECS = [
  {{ key: 'flap', name: 'Flap', color: '{FLAP_FILL}' }},
  {{ key: 'aileron', name: 'Aileron', color: '{AILERON_FILL}' }},
  {{ key: 'mfs1', name: 'MFS 1', color: 'rgba(241, 196, 15, 0.10)' }},
  {{ key: 'mfs2', name: 'MFS 2', color: 'rgba(230, 126, 34, 0.10)' }},
  {{ key: 'winglet', name: 'Winglet', color: 'rgba(155, 89, 182, 0.10)' }},
];

function surfaceSpec(key) {{
  for (var i = 0; i < SURFACE_SPECS.length; i++) {{
    if (SURFACE_SPECS[i].key === key) return SURFACE_SPECS[i];
  }}
  return null;
}}

function addPlanformSurfaceTraces(traces) {{
  var surfaceSpecs = SURFACE_SPECS;

  surfaceSpecs.forEach(function(spec) {{
    var polygons = (PLANFORM.surfaces && PLANFORM.surfaces[spec.key]) || [];
    polygons.forEach(function(poly, idx) {{
      traces.push({{
        x: poly.ws,
        y: poly.x,
        mode: 'lines',
        name: spec.name,
        fill: 'toself',
        fillcolor: spec.color,
        line: {{ color: planformLineColor(spec.color), width: 1 }},
        showlegend: idx === 0,
        hovertemplate: spec.name + '<br>Y: %{{x:.1f}} in<br>X: %{{y:.1f}} in<extra></extra>'
      }});
    }});
  }});
}}

function addPlanformStringerTrace(traces, runs, label, color, dash) {{
  // [AI] One surface's whole family is one trace, its runs separated by nulls, so the
  // legend carries a single entry per surface that shows or hides all of them at once
  // -- a dozen stringers per wing would otherwise bury every other entry in the
  // legend. 'legendonly' is what leaves each off until asked for: the entry is drawn
  // greyed out and one click brings the lines in.
  //
  // The two surfaces share one ladder, so the lower lines land on the upper ones
  // except at the access cut-out. The lower one is dashed and a different colour for
  // that reason: with both shown, what reads is the gap, not the overlap.
  if (!runs || !runs.length) return;

  var xs = [], ys = [], cd = [];
  runs.forEach(function(run) {{
    // Both wings. The family is symmetric about the centreline, so the right-hand run
    // reflected in Y is the left-hand one -- the same mirror the outlines above use.
    [-1, 1].forEach(function(sign) {{
      for (var i = 0; i < run.ws.length; i++) {{
        xs.push(sign * run.ws[i]);
        ys.push(run.x[i]);
        cd.push(run.number);
      }}
      xs.push(null); ys.push(null); cd.push(null);
    }});
  }});

  traces.push({{
    x: xs, y: ys, mode: 'lines',
    name: 'Stringers (' + label + ')',
    line: {{ color: color, width: 1, dash: dash }},
    customdata: cd,
    visible: 'legendonly',
    hovertemplate: 'Stg %{{customdata}} (' + label + ')<br>Y: %{{x:.1f}} in'
                   + '<br>X: %{{y:.1f}} in<extra></extra>'
  }});
}}

// [AI] ---- Coordinate-system axes ----
// A flat drawing of the global and local coordinate systems, sat as a key in the empty
// space below the port wing. Nothing has to be projected: the plot's own axes already
// ARE global axes -- horizontal is global Y (starboard positive) and vertical is global
// X (aft positive, drawn reversed so aft is down) -- so a unit vector's Y and X
// components are the offset to draw, as they stand.
//
// Only the two in-plane axes are drawn. Z is normal to a plan view, so on this plot it
// is a point rather than a direction, and the local z' is within a few degrees of it.
// The dihedral that separates the two is reported as a number under the local pair
// instead, where it can be read rather than guessed at from a few pixels of arrow.
var AXIS_ARROW_DEG = 22;  // half-angle of the arrowheads

// [AI] One arm and its two arrowhead barbs, appended to pts as null-separated runs. The
// two plot axes are equally scaled (scaleanchor below), so a perpendicular taken in
// data units is a perpendicular on screen; the reversed station axis is a reflection,
// which leaves the symmetric arrowhead unchanged.
function axisArrow(pts, x0, y0, dx, dy) {{
  var len = Math.sqrt(dx * dx + dy * dy);
  if (len < 1e-9) return;
  var xt = x0 + dx, yt = y0 + dy;
  var ux = dx / len, uy = dy / len;
  var h = 0.2 * len, k = h * Math.tan(AXIS_ARROW_DEG * Math.PI / 180);
  pts.x.push(x0, xt, null);
  pts.y.push(y0, yt, null);
  pts.x.push(xt - ux * h - uy * k, xt, xt - ux * h + uy * k, null);
  pts.y.push(yt - uy * h + ux * k, yt, yt - uy * h - ux * k, null);
}}

// [AI] Where a tip label sits, from the arm's direction on SCREEN -- data y is drawn
// reversed, so screen up is -dy.
function axisLabelPos(dx, dy) {{
  var sx = dx, sy = -dy;
  var v = Math.abs(sy) > 0.4 * Math.abs(sx) ? (sy >= 0 ? 'top' : 'bottom') : 'middle';
  var h = Math.abs(sx) > 0.4 * Math.abs(sy) ? (sx >= 0 ? 'right' : 'left') : 'center';
  return v + ' ' + h;
}}

function addPlanformAxisKey(traces) {{
  var sys = PLANFORM.axis_systems;
  if (!sys) return;

  var span = Math.max.apply(null, PLANFORM.ws.map(Math.abs));
  var arm = 0.11 * span;
  // [AI] Clear of the trailing edge at its aft-most, so the key never lands on the wing.
  var yBase = Math.max.apply(null, PLANFORM.te) + 0.20 * span;

  var lines = {{ x: [], y: [] }};
  var labels = {{ x: [], y: [], text: [], pos: [] }};

  function label(x, y, text, pos) {{
    labels.x.push(x); labels.y.push(y); labels.text.push(text); labels.pos.push(pos);
  }}
  function axis(x0, v, text) {{
    var dx = v[1] * arm, dy = v[0] * arm;
    axisArrow(lines, x0, yBase, dx, dy);
    label(x0 + dx, yBase + dy, text, axisLabelPos(dx, dy));
  }}
  function pair(x0, e_x, e_y, names, title) {{
    axis(x0, e_x, names[0]);
    axis(x0, e_y, names[1]);
    label(x0, yBase - 0.9 * arm, title, 'top center');
  }}

  pair(-0.88 * span, [1, 0, 0], [0, 1, 0], ['X', 'Y'], 'Global');
  pair(-0.52 * span, sys.e_x, sys.e_y, ['X\u2032', 'Y\u2032'],
       'Local (bay ' + sys.bay_id + ')<br>\u039b<sub>EA</sub> '
       + sys.sweep_ea_deg.toFixed(1) + '\u00b0, \u0393 '
       + sys.dihedral_deg.toFixed(1) + '\u00b0');

  // [AI] Off until asked for: the key answers a question about the drawing rather than
  // adding anything to it, so it starts out of the way. The two traces share a
  // legendgroup, so the one legend entry brings the arrows and their labels in
  // together.
  traces.push({{
    x: lines.x, y: lines.y, mode: 'lines',
    name: 'Coordinate systems', legendgroup: 'axes', visible: 'legendonly',
    line: {{ color: '#2c3e50', width: 1.5 }},
    hoverinfo: 'skip'
  }});
  traces.push({{
    x: labels.x, y: labels.y, mode: 'text',
    text: labels.text, textposition: labels.pos,
    legendgroup: 'axes', showlegend: false, visible: 'legendonly',
    textfont: {{ color: '#2c3e50', size: 12 }},
    hoverinfo: 'skip'
  }});
}}

function initPlanform() {{
  var traces = [];
  addPlanformSurfaceTraces(traces);
  addPlanformStringerTrace(traces, PLANFORM.stringers_upper,
                           'upper surface', '#8c6d3f', 'solid');
  addPlanformStringerTrace(traces, PLANFORM.stringers_lower,
                           'lower surface', '#3f6d8c', 'dot');
  traces.push(
    {{ x: PLANFORM.ws, y: PLANFORM.le, mode: 'lines', name: 'Leading Edge',
       line: {{ color: '#1f77b4', width: 2 }} }},
    {{ x: PLANFORM.ws, y: PLANFORM.te, mode: 'lines', name: 'Trailing Edge',
       line: {{ color: '#1f77b4', width: 2 }} }},
    {{ x: PLANFORM.ws, y: PLANFORM.fwd, mode: 'lines', name: 'Fwd Spar',
       line: {{ color: '#d62728', width: 1.5, dash: 'dash' }} }},
    {{ x: PLANFORM.ws, y: PLANFORM.aft, mode: 'lines', name: 'Aft Spar',
       line: {{ color: '#2ca02c', width: 1.5, dash: 'dash' }} }}
  );
  if (PLANFORM.semispan) {{
    traces.push(
      {{ x: PLANFORM.semispan.ws, y: PLANFORM.semispan.quarter_chord,
         mode: 'lines', name: 'Quarter Chord (right half)',
         line: {{ color: '#ff7f0e', width: 2, dash: 'dot' }},
         hovertemplate: 'Quarter Chord<br>Y: %{{x:.1f}} in<br>X: %{{y:.1f}} in<extra></extra>' }},
      {{ x: PLANFORM.semispan.ws, y: PLANFORM.semispan.elastic_axis,
         mode: 'lines', name: 'Elastic Axis (right half)',
         line: {{ color: '#9467bd', width: 2, dash: 'dashdot' }},
         hovertemplate: 'Elastic Axis<br>Y: %{{x:.1f}} in<br>X: %{{y:.1f}} in<extra></extra>' }}
    );
    if (PLANFORM.semispan.bay_labels) {{
      traces.push({{
        x: PLANFORM.semispan.bay_labels.map(function(label) {{ return label.ws; }}),
        y: PLANFORM.semispan.bay_labels.map(function(label) {{ return label.x; }}),
        text: PLANFORM.semispan.bay_labels.map(function(label) {{ return label.text; }}),
        mode: 'text',
        name: 'Bay Number',
        textfont: {{ color: '#111', size: 12 }},
        showlegend: false,
        hoverinfo: 'skip'
      }});
    }}
  }}

  // Rib lines
  for (var i = 0; i < PLANFORM.ws.length; i++) {{
    traces.push({{
      x: [PLANFORM.ws[i], PLANFORM.ws[i]],
      y: [PLANFORM.fwd[i], PLANFORM.aft[i]],
      mode: 'lines', line: {{ color: '#333', width: 0.5 }},
      showlegend: false, hoverinfo: 'skip'
    }});
  }}

  addPlanformAxisKey(traces);

  var shapes = [];

  // Nacelle reference at ±Y (symmetric full wing)
  [
    {{ y: PLANFORM.nacelle_ib_y, color: '#ff7f0e',
       legend: 'Inboard Nacelle (\\u00b1Y)', showLeg: true }},
    {{ y: -PLANFORM.nacelle_ib_y, color: '#ff7f0e', legend: '', showLeg: false }},
    {{ y: PLANFORM.nacelle_ob_y, color: '#9467bd',
       legend: 'Outboard Nacelle (\\u00b1Y)', showLeg: true }},
    {{ y: -PLANFORM.nacelle_ob_y, color: '#9467bd', legend: '', showLeg: false }},
  ].forEach(function(spec) {{
    shapes.push({{
      type: 'line', x0: spec.y, x1: spec.y, y0: 0, y1: 1, yref: 'paper',
      line: {{ color: spec.color, width: 1.5, dash: 'dot' }}
    }});
    if (spec.showLeg) {{
      traces.push({{
        x: [null], y: [null], mode: 'lines',
        name: spec.legend,
        line: {{ color: spec.color, dash: 'dot' }}
      }});
    }}
  }});

  var layout = {{
    xaxis: {{
      title: 'Spanwise distance from centerline (in, \\u00b1Y)',
      zeroline: true, zerolinewidth: 2, zerolinecolor: '#bdc3c7',
    }},
    yaxis: {{ title: 'Fuselage Station X (in)', autorange: 'reversed',
              scaleanchor: 'x', scaleratio: 1 }},
    legend: {{ orientation: 'h', x: 0.5, xanchor: 'center', y: -0.08 }},
    margin: {{ t: 30, b: 80, l: 80, r: 30 }},
    shapes: shapes,
    hovermode: 'closest'
  }};

  Plotly.newPlot('planform-plot', traces, layout, {{ responsive: true }});
  drawPlanformOverviewTable();
  drawPlanformInputsTable();
  drawPlanformBayGeomTable();
}}

// ---- Cross Sections ----
var xsecInited = false;

function escapeHtml(str) {{
  if (str == null || str === undefined) return '';
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\"/g,'&quot;');
}}

function formatNumber(v, decimals) {{
  var n = Number(v);
  if (!isFinite(n)) return '-';
  return n.toFixed(decimals === undefined ? 2 : decimals);
}}

function formatXsecClickDetail(cd) {{
  if (!cd) {{
    return 'No component data for this trace. Select a skin marker, stringer, or spar trace.';
  }}
  if (Object.prototype.toString.call(cd) !== '[object Array]') {{
    return 'No component data for this trace.';
  }}
  var kind = cd[0];
  if (kind === 'skin') {{
    return '<b>' + escapeHtml(cd[1]) + ' skin — ' + escapeHtml(cd[2]) + '</b><br>'
      + 'Material: ' + escapeHtml(cd[3]) + '<br>'
      + 'Plies: ' + escapeHtml(cd[4]) + '<br>'
      + 'Thickness: ' + Number(cd[5]).toFixed(4) + ' in<br>'
      + 'Area: ' + Number(cd[12]).toFixed(4) + ' in&sup2;<br>'
      + 'a (panel length): ' + Number(cd[6]).toFixed(4) + ' in<br>'
      + 'Bay width (pitch): ' + Number(cd[7]).toFixed(2) + ' in &nbsp; ex_scale: ' + Number(cd[8]).toFixed(4) + '<br>'
      + 'R_chord: ' + (cd[11] == null || !isFinite(cd[11]) ? 'flat' : Number(cd[11]).toFixed(2) + ' in') + '<br>'
      + 'Y=' + Number(cd[9]).toFixed(3) + ' in, Z=' + Number(cd[10]).toFixed(3) + ' in';
  }}
  if (kind === 'stringer') {{
    return '<b>' + escapeHtml(cd[1]) + ' stringer — ' + escapeHtml(cd[2]) + '</b><br>'
      + 'Material: ' + escapeHtml(cd[3]) + '<br>'
      + 'Plies: ' + escapeHtml(cd[4]) + '<br>'
      + 'width=' + Number(cd[5]).toFixed(3) + ' in, height=' + Number(cd[6]).toFixed(3) + ' in, t='
      + Number(cd[7]).toFixed(4) + ' in<br>'
      + 'Area: ' + Number(cd[10]).toFixed(4) + ' in&sup2;<br>'
      + 'Y=' + Number(cd[8]).toFixed(3) + ' in, Z=' + Number(cd[9]).toFixed(3) + ' in';
  }}
  if (kind === 'spar_web') {{
    return '<b>' + escapeHtml(cd[1]) + ' spar web — ' + escapeHtml(cd[2]) + '</b><br>'
      + 'Material: ' + escapeHtml(cd[3]) + '<br>'
      + 'Plies: ' + escapeHtml(cd[4]) + '<br>'
      + 'Height: ' + Number(cd[5]).toFixed(2) + ' in<br>'
      + 'Thickness: ' + Number(cd[6]).toFixed(4) + ' in<br>'
      + 'Area: ' + Number(cd[9]).toFixed(4) + ' in&sup2;<br>'
      + 'Y=' + Number(cd[7]).toFixed(3) + ' in, Z=' + Number(cd[8]).toFixed(3) + ' in';
  }}
  if (kind === 'spar_cap') {{
    return '<b>' + escapeHtml(cd[1]) + ' spar cap (' + escapeHtml(cd[2]) + ') — ' + escapeHtml(cd[3]) + '</b><br>'
      + 'Material: ' + escapeHtml(cd[4]) + '<br>'
      + 'Plies: ' + escapeHtml(cd[5]) + '<br>'
      + 'width=' + Number(cd[6]).toFixed(3) + ' in, t=' + Number(cd[7]).toFixed(4) + ' in<br>'
      + 'Area: ' + Number(cd[12]).toFixed(4) + ' in&sup2;<br>'
      + 'Skin leg: Y=' + Number(cd[8]).toFixed(3) + ' in, Z=' + Number(cd[9]).toFixed(3) + ' in<br>'
      + 'Web leg: Y=' + Number(cd[10]).toFixed(3) + ' in, Z=' + Number(cd[11]).toFixed(3) + ' in';
  }}
  if (kind === 'section') {{
    return '<b>' + escapeHtml(cd[1]) + '</b><br>Y=' + Number(cd[2]).toFixed(3) + ' in, Z=' + Number(cd[3]).toFixed(3) + ' in';
  }}
  return escapeHtml(JSON.stringify(cd));
}}

function bindXsecPlotClickOnce() {{
  if (window.__xsecPlotClickBound) return;
  var gd = document.getElementById('xsec-plot');
  if (!gd) return;
  gd.on('plotly_click', function(ev) {{
    var el = document.getElementById('xsec-click-detail');
    if (!el) return;
    if (!ev.points || !ev.points.length) {{
      el.textContent = 'No point selected.';
      return;
    }}
    el.innerHTML = formatXsecClickDetail(ev.points[0].customdata);
  }});
  window.__xsecPlotClickBound = true;
}}

function initXsec() {{
  var sel = document.getElementById('bay-select');
  BAYS.forEach(function(b, i) {{
    var opt = document.createElement('option');
    opt.value = i;
    opt.textContent = 'Bay ' + b.bay_id + ", WS = " + b.ws.toFixed(1) + "''";
    sel.appendChild(opt);
  }});
  var lcSel = document.getElementById('xsec-vmt-lc-select');
  orderedVmtIndices().forEach(function(vmtIdx) {{
    var opt = document.createElement('option');
    opt.value = vmtIdx;
    opt.textContent = VMT_DATA[vmtIdx].name;
    lcSel.appendChild(opt);
  }});
  xsecInited = true;
  updateXsec();
}}

// VMT dataset indices in the Stress Results load case order. VMT names come from
// sanitized filenames, so match on alphanumerics only; anything unmatched keeps
// its original position at the end.
function orderedVmtIndices() {{
  function key(s) {{ return String(s).toLowerCase().replace(/[^a-z0-9]/g, ''); }}
  var remaining = VMT_DATA.map(function(ds, i) {{ return i; }});
  var ordered = [];
  ((RESULTS && RESULTS.load_cases) || []).forEach(function(lcName) {{
    for (var j = 0; j < remaining.length; j++) {{
      if (key(VMT_DATA[remaining[j]].name) === key(lcName)) {{
        ordered.push(remaining[j]);
        remaining.splice(j, 1);
        return;
      }}
    }}
  }});
  return ordered.concat(remaining);
}}

function updateXsec() {{
  var idx = parseInt(document.getElementById('bay-select').value);
  var bay = BAYS[idx];
  var scale = document.getElementById('xsec-scale-select');
  var opts = {{ showControlSurfaces: true }};
  if (scale && scale.value === 'fixed') opts.fixedRange = xsecFullRange(BAYS[0]);
  drawXsecPlot(bay, null, opts);
  drawXsecVmtTable();
  drawPropsTable(bay);
  buildXsecComponentsTable(bay);
}}

// Axis ranges covering the full-airfoil view of a bay: the section itself plus the
// chord / height dimension lines and labels drawn around it. Applying the first
// bay's ranges to every bay is what the fixed-scale option does, so sections stay
// comparable in size instead of each being auto-fitted.
function xsecFullRange(bay) {{
  var AF = bay.airfoil || AIRFOIL;
  var zScale = bay.chord * (bay.t_c / AF.ref_tc);
  var zMin = Math.min.apply(null, AF.lower) * zScale;
  var zMax = Math.max.apply(null, AF.upper) * zScale;
  var dimOff = (zMax - zMin) * 0.3;
  return {{
    x: [-dimOff * 1.3, bay.chord + dimOff * 0.3],
    y: [zMin - dimOff * 1.2, zMax + dimOff * 1.2]
  }};
}}

// Linear interpolation with clamping outside the range, matching np.interp so the
// table agrees with the loads the stress calculation uses at the same station.
function interpAtWs(wsArr, valArr, ws) {{
  var n = wsArr.length;
  if (n === 0) return NaN;
  if (ws <= wsArr[0]) return valArr[0];
  if (ws >= wsArr[n - 1]) return valArr[n - 1];
  for (var i = 1; i < n; i++) {{
    if (wsArr[i] >= ws) {{
      var span = wsArr[i] - wsArr[i - 1];
      if (span === 0) return valArr[i];
      return valArr[i - 1] + (ws - wsArr[i - 1]) / span * (valArr[i] - valArr[i - 1]);
    }}
  }}
  return valArr[n - 1];
}}

function drawXsecVmtTable() {{
  var el = document.getElementById('xsec-vmt-table');
  if (VMT_DATA.length === 0) {{
    el.innerHTML = '<tbody><tr><td colspan="4">No VMT data available.</td></tr></tbody>';
    return;
  }}
  var bay = BAYS[parseInt(document.getElementById('bay-select').value)];
  var ds = VMT_DATA[parseInt(document.getElementById('xsec-vmt-lc-select').value)];

  var byCol = {{}};
  VMT_COMP.forEach(function(comp) {{ byCol[comp[0]] = comp; }});

  var html = '<thead><tr>'
    + '<th>Load</th><th>Value</th><th>Units</th><th>Description</th>'
    + '</tr></thead><tbody>';
  ['Vx', 'Vy', 'Vz', 'Mx', 'My', 'Mz'].forEach(function(col) {{
    var comp = byCol[col];
    if (!comp) return;
    var val = interpAtWs(ds.WS, ds[col], bay.ws);
    html += '<tr>'
      + '<td>' + col + '</td>'
      + '<td><input class="val" readonly value="' + fmtSci(val) + '" '
      +   'onclick="this.select()" title="Click to select"></td>'
      + '<td>' + escapeHtml(comp[2]) + '</td>'
      + '<td>' + escapeHtml(comp[1]) + '</td>'
      + '</tr>';
  }});
  html += '</tbody>';
  el.innerHTML = html;
}}

function drawXsecPlot(bay, targetId, options) {{
  targetId = targetId || 'xsec-plot';
  options = options || {{}};
  var chord = bay.chord;
  var tc = bay.t_c;
  var AF = bay.airfoil || AIRFOIL;
  var tcScale = tc / AF.ref_tc;

  var xAf = AF.x.map(v => v * chord);
  var yUp = AF.upper.map(v => v * chord * tcScale);
  var yLo = AF.lower.map(v => v * chord * tcScale);

  var xFwd = chord * bay.fwd_ratio;
  var xAft = chord * bay.aft_ratio;

  function interp(xq) {{
    var ax = AF.x;
    for (var i = 1; i < ax.length; i++) {{
      if (ax[i] >= xq) {{
        var t = (xq - ax[i-1]) / (ax[i] - ax[i-1]);
        return function(arr) {{ return arr[i-1] + t * (arr[i] - arr[i-1]); }};
      }}
    }}
    return function(arr) {{ return arr[arr.length - 1]; }};
  }}

  function centroidOffsetFromBondStg(w, h, t) {{
    var A_fl = w * t, y_fl = t / 2;
    var h_web = h - t, A_web = 2 * t * h_web, y_web = t + h_web / 2;
    var A_tot = A_fl + A_web;
    if (A_tot <= 1e-15) return 0;
    return (A_fl * y_fl + A_web * y_web) / A_tot;
  }}
  function zSurfPhys(xcn, zCurve) {{
    var fi = interp(xcn);
    return fi(zCurve) * chord * tcScale;
  }}
  function dZdYSurf(xcn, zCurve) {{
    var ax = AF.x;
    var dxC = Math.max(1e-6, 1e-4 * Math.max(ax[ax.length - 1] - ax[0], 0.1));
    var xc0 = Math.max(ax[0], Math.min(ax[ax.length - 1], xcn));
    var xcLo = Math.max(ax[0], xc0 - dxC);
    var xcHi = Math.min(ax[ax.length - 1], xc0 + dxC);
    if (xcHi <= xcLo + 1e-12) {{ xcLo = ax[0]; xcHi = ax[ax.length - 1]; }}
    var zLo = zSurfPhys(xcLo, zCurve);
    var zHi = zSurfPhys(xcHi, zCurve);
    var xLo = xcLo * chord, xHi = xcHi * chord;
    return (zHi - zLo) / (xHi - xLo);
  }}
  function skinPanelPolygon(panel, isUpper) {{
    var xl = panel.x_left / chord, xr = panel.x_right / chord;
    var ax = AF.x;
    xl = Math.max(ax[0], Math.min(ax[ax.length - 1], xl));
    xr = Math.max(ax[0], Math.min(ax[ax.length - 1], xr));
    var zCurve = isUpper ? AF.upper : AF.lower;
    var nSamp = 20;
    var dx = (xr - xl) / nSamp;
    var omlX = [], omlZ = [], imlX = [], imlZ = [];
    var tSkin = panel.thickness;
    for (var k = 0; k <= nSamp; k++) {{
      var xn = xl + k * dx;
      var zOml = zSurfPhys(xn, zCurve);
      var xP = xn * chord;
      var dzdy = dZdYSurf(xn, zCurve);
      var nrm = Math.hypot(dzdy, 1);
      var nx = isUpper ? dzdy / nrm : -dzdy / nrm;
      var nz = isUpper ? -1 / nrm : 1 / nrm;
      omlX.push(xP); omlZ.push(zOml);
      imlX.push(xP + tSkin * nx); imlZ.push(zOml + tSkin * nz);
    }}
    var px = omlX.concat(imlX.reverse());
    var pz = omlZ.concat(imlZ.reverse());
    px.push(px[0]); pz.push(pz[0]);
    return {{ x: px, z: pz }};
  }}

  function stringerPolygons(s, isUpper) {{
    var xcNorm = s.x / chord;
    xcNorm = Math.max(AF.x[0], Math.min(AF.x[AF.x.length - 1], xcNorm));
    var zCurve = isUpper ? AF.upper : AF.lower;
    var dzdy = dZdYSurf(xcNorm, zCurve);
    var nrm = Math.hypot(dzdy, 1);
    var nx = isUpper ? dzdy / nrm : -dzdy / nrm;
    var nz = isUpper ? -1 / nrm : 1 / nrm;
    var tnorm = Math.hypot(1, dzdy);
    var tx = 1 / tnorm, tz = dzdy / tnorm;
    var t = s.thickness;
    var tw = 2 * t;
    var w = s.width, h = s.height;
    var yBond = centroidOffsetFromBondStg(w, h, t);
    var bx = s.x - yBond * nx, bz = s.z - yBond * nz;
    // Flange: w x t rectangle at the bond surface, angled to OML
    var fOx = bx - 0.5 * w * tx, fOz = bz - 0.5 * w * tz;
    var fIx = bx + 0.5 * w * tx, fIz = bz + 0.5 * w * tz;
    var flange = {{
      x: [fOx, fIx, fIx + t * nx, fOx + t * nx, fOx],
      z: [fOz, fIz, fIz + t * nz, fOz + t * nz, fOz]
    }};
    // Web: 2t x (h - t) rectangle, normal to skin, from top of flange
    var hWeb = h - t;
    var wbx = bx - 0.5 * tw * tx + t * nx, wbz = bz - 0.5 * tw * tz + t * nz;
    var web = {{
      x: [wbx, wbx + tw * tx, wbx + tw * tx + hWeb * nx, wbx + hWeb * nx, wbx],
      z: [wbz, wbz + tw * tz, wbz + tw * tz + hWeb * nz, wbz + hWeb * nz, wbz]
    }};
    return {{ flange: flange, web: web }};
  }}

  var fwdInterp = interp(bay.fwd_ratio);
  var aftInterp = interp(bay.aft_ratio);
  var zUF = fwdInterp(AF.upper) * chord * tcScale;
  var zLF = fwdInterp(AF.lower) * chord * tcScale;
  var zUA = aftInterp(AF.upper) * chord * tcScale;
  var zLA = aftInterp(AF.lower) * chord * tcScale;

  // Max thickness location
  var maxIdx = 0, maxTh = 0;
  for (var i = 0; i < AF.x.length; i++) {{
    var th = AF.upper[i] - AF.lower[i];
    if (th > maxTh) {{ maxTh = th; maxIdx = i; }}
  }}
  var zTop = AF.upper[maxIdx] * chord * tcScale;
  var zBot = AF.lower[maxIdx] * chord * tcScale;
  var sectH = zTop - zBot;
  var yMin = Math.min.apply(null, yLo);
  var yMax = Math.max.apply(null, yUp);
  var hRange = yMax - yMin;
  var dimOff = hRange * 0.3;

  var traces = [
    {{ x: xAf, y: yUp, mode: 'lines', name: 'Upper Surface',
       line: {{ color: '#1f77b4', width: 1.5 }}, showlegend: false, hoverinfo: 'skip' }},
    {{ x: xAf, y: yLo, mode: 'lines', name: 'Lower Surface',
       line: {{ color: '#1f77b4', width: 1.5 }}, showlegend: false, hoverinfo: 'skip' }},
  ];

  var upSk = bay.upper_skin_panels || [];
  upSk.forEach(function(p) {{
    var poly = skinPanelPolygon(p, true);
    traces.push({{
      x: poly.x, y: poly.z,
      mode: 'lines', name: 'Upper skin — ' + p.name,
      line: {{ color: '#1f77b4', width: 0.5 }},
      fill: 'toself', fillcolor: 'rgba(31,119,180,0.25)',
      showlegend: false, hoverinfo: 'skip'
    }});
  }});
  var loSk = bay.lower_skin_panels || [];
  loSk.forEach(function(p) {{
    var poly = skinPanelPolygon(p, false);
    traces.push({{
      x: poly.x, y: poly.z,
      mode: 'lines', name: 'Lower skin — ' + p.name,
      line: {{ color: '#1f77b4', width: 0.5 }},
      fill: 'toself', fillcolor: 'rgba(31,119,180,0.25)',
      showlegend: false, hoverinfo: 'skip'
    }});
  }});

  function pushSparDrawGeom(sp, label, spec) {{
    if (!sp || !sp.draw) return;
    var d = sp.draw;
    var wm = sp.web;
    var uc = sp.upper_cap;
    var lc = sp.lower_cap;
    function pushPoly(poly, traceName, lineColor, fillRgba) {{
      if (!poly || !poly.x || poly.x.length < 3) return;
      traces.push({{
        x: poly.x, y: poly.z,
        mode: 'lines', name: traceName,
        line: {{ color: lineColor, width: 1.5 }},
        fill: 'toself', fillcolor: fillRgba,
        showlegend: false, hoverinfo: 'skip'
      }});
    }}
    pushPoly(d.web, label + ' spar web', spec.webLine, spec.webFill);
    if (d.upper_skin_leg) pushPoly(d.upper_skin_leg, label + ' upper skin cap', spec.capLine, spec.upperSkinFill);
    pushPoly(d.upper_web_leg, label + ' upper web cap', spec.capLine, spec.upperWebFill);
    if (d.lower_skin_leg) pushPoly(d.lower_skin_leg, label + ' lower skin cap', spec.capLine, spec.lowerSkinFill);
    pushPoly(d.lower_web_leg, label + ' lower web cap', spec.capLine, spec.lowerWebFill);

    var hoverX = [], hoverZ = [], hoverCd = [];
    var hWeb = '<b>' + label + ' spar web — %{{customdata[2]}}</b><br>Material: %{{customdata[3]}}<br>Plies: %{{customdata[4]}}<br>Height: %{{customdata[5]:.2f}} in<br>t: %{{customdata[6]:.4f}} in<br>Area: %{{customdata[9]:.4f}} in²<br>Y=%{{customdata[7]:.3f}} in, Z=%{{customdata[8]:.3f}} in<extra></extra>';
    var hCap = '<b>%{{customdata[1]}} spar cap (%{{customdata[2]}}) — %{{customdata[3]}}</b><br>Material: %{{customdata[4]}}<br>Plies: %{{customdata[5]}}<br>width=%{{customdata[6]:.3f}} in, t=%{{customdata[7]:.4f}} in<br>Area: %{{customdata[12]:.4f}} in²<br>Skin leg: Y=%{{customdata[8]:.3f}} in, Z=%{{customdata[9]:.3f}} in<br>Web leg: Y=%{{customdata[10]:.3f}} in, Z=%{{customdata[11]:.3f}} in<extra></extra>';
    if (d.web && d.web.x.length >= 3) {{
      hoverX.push(wm.x); hoverZ.push(wm.z);
      hoverCd.push(['spar_web', label, wm.name, wm.material, wm.plies, wm.height, wm.thickness, wm.x, wm.z, wm.area]);
    }}
    function addCapHover(poly, pos, capData, xMk, zMk) {{
      if (!poly || !poly.x || poly.x.length < 3) return;
      hoverX.push(xMk); hoverZ.push(zMk);
      hoverCd.push(['spar_cap', label, pos, capData.name, capData.material, capData.plies,
        capData.width, capData.thickness,
        capData.x_skin_leg, capData.z_skin_leg, capData.x_web_leg, capData.z_web_leg,
        capData.area]);
    }}
    addCapHover(d.upper_skin_leg, 'Upper (skin leg)', uc, uc.x_skin_leg, uc.z_skin_leg);
    addCapHover(d.upper_web_leg, 'Upper (web leg)', uc, uc.x_web_leg, uc.z_web_leg);
    addCapHover(d.lower_skin_leg, 'Lower (skin leg)', lc, lc.x_skin_leg, lc.z_skin_leg);
    addCapHover(d.lower_web_leg, 'Lower (web leg)', lc, lc.x_web_leg, lc.z_web_leg);
    if (hoverX.length) {{
      traces.push({{
        x: hoverX, y: hoverZ,
        mode: 'markers', name: label + ' spar hover',
        marker: {{ size: 8, opacity: 0.01, color: spec.webLine, symbol: 'square', line: {{ width: 0 }} }},
        showlegend: false,
        customdata: hoverCd,
        hovertemplate: hoverCd.map(function(cd) {{
          return cd[0] === 'spar_web' ? hWeb : hCap;
        }})
      }});
    }}
  }}

  pushSparDrawGeom(bay.fwd_spar, 'Forward', {{
    webLine: '#9e2a2b', webFill: 'rgba(214,39,40,0.55)',
    capLine: '#6d1f20',
    upperSkinFill: 'rgba(214,39,40,0.45)', upperWebFill: 'rgba(214,39,40,0.55)',
    lowerSkinFill: 'rgba(214,39,40,0.45)', lowerWebFill: 'rgba(214,39,40,0.55)',
  }});
  pushSparDrawGeom(bay.aft_spar, 'Aft', {{
    webLine: '#1f6b3d', webFill: 'rgba(39,174,96,0.55)',
    capLine: '#14532a',
    upperSkinFill: 'rgba(39,174,96,0.45)', upperWebFill: 'rgba(39,174,96,0.55)',
    lowerSkinFill: 'rgba(39,174,96,0.45)', lowerWebFill: 'rgba(39,174,96,0.55)',
  }});

  if (upSk.length) {{
    traces.push({{
      x: upSk.map(function(p) {{ return p.x; }}),
      y: upSk.map(function(p) {{ return p.z; }}),
      mode: 'markers', name: 'Upper skin panels',
      showlegend: false,
      marker: {{ size: 6, opacity: 0.14, color: '#d62728', symbol: 'square',
        line: {{ width: 0 }} }},
      customdata: upSk.map(function(p) {{
        return ['skin', 'Upper', p.name, p.material, p.plies, p.thickness, p.panel_length, p.bay_width, p.ex_scale, p.x, p.z, p.R_chord, p.area];
      }}),
      hovertemplate: '<b>Upper skin — %{{customdata[2]}}</b><br>Material: %{{customdata[3]}}<br>Plies: %{{customdata[4]}}<br>Thickness: %{{customdata[5]:.4f}} in<br>Area: %{{customdata[12]:.4f}} in²<br>a (panel length): %{{customdata[6]:.4f}} in<br>Bay width (pitch): %{{customdata[7]:.2f}} in<br>ex_scale: %{{customdata[8]:.4f}}<br>R_chord: %{{customdata[11]:.2f}} in<br>Y=%{{customdata[9]:.3f}} in, Z=%{{customdata[10]:.3f}} in<extra></extra>'
    }});
  }}
  if (loSk.length) {{
    traces.push({{
      x: loSk.map(function(p) {{ return p.x; }}),
      y: loSk.map(function(p) {{ return p.z; }}),
      mode: 'markers', name: 'Lower skin panels',
      showlegend: false,
      marker: {{ size: 6, opacity: 0.14, color: '#1f77b4', symbol: 'square',
        line: {{ width: 0 }} }},
      customdata: loSk.map(function(p) {{
        return ['skin', 'Lower', p.name, p.material, p.plies, p.thickness, p.panel_length, p.bay_width, p.ex_scale, p.x, p.z, p.R_chord, p.area];
      }}),
      hovertemplate: '<b>Lower skin — %{{customdata[2]}}</b><br>Material: %{{customdata[3]}}<br>Plies: %{{customdata[4]}}<br>Thickness: %{{customdata[5]:.4f}} in<br>Area: %{{customdata[12]:.4f}} in²<br>a (panel length): %{{customdata[6]:.4f}} in<br>Bay width (pitch): %{{customdata[7]:.2f}} in<br>ex_scale: %{{customdata[8]:.4f}}<br>R_chord: %{{customdata[11]:.2f}} in<br>Y=%{{customdata[9]:.3f}} in, Z=%{{customdata[10]:.3f}} in<extra></extra>'
    }});
  }}

  var upStg = bay.upper_stringers || [];
  upStg.forEach(function(s) {{
    var g = stringerPolygons(s, true);
    traces.push({{
      x: g.flange.x, y: g.flange.z,
      mode: 'lines', name: 'Upper stringer flange — ' + s.name,
      line: {{ color: '#a51c30', width: 1 }},
      fill: 'toself', fillcolor: 'rgba(165,28,48,0.45)',
      showlegend: false, hoverinfo: 'skip'
    }});
    traces.push({{
      x: g.web.x, y: g.web.z,
      mode: 'lines', name: 'Upper stringer web — ' + s.name,
      line: {{ color: '#a51c30', width: 1 }},
      fill: 'toself', fillcolor: 'rgba(165,28,48,0.55)',
      showlegend: false, hoverinfo: 'skip'
    }});
  }});
  if (upStg.length) {{
    traces.push({{
      x: upStg.map(function(s) {{ return s.x; }}),
      y: upStg.map(function(s) {{ return s.z; }}),
      mode: 'markers', name: 'Upper stringers',
      showlegend: false,
      marker: {{ size: 5, opacity: 0.01, color: '#d62728', symbol: 'circle',
        line: {{ width: 0 }} }},
      customdata: upStg.map(function(s) {{
        return ['stringer', 'Upper', s.name, s.material, s.plies, s.width, s.height, s.thickness, s.x, s.z, s.area];
      }}),
      hovertemplate: '<b>Upper stringer — %{{customdata[2]}}</b><br>Material: %{{customdata[3]}}<br>Plies: %{{customdata[4]}}<br>width=%{{customdata[5]:.3f}} in, height=%{{customdata[6]:.3f}} in, t=%{{customdata[7]:.4f}} in<br>Area: %{{customdata[10]:.4f}} in²<br>Y=%{{customdata[8]:.3f}} in, Z=%{{customdata[9]:.3f}} in<extra></extra>'
    }});
  }}
  var loStg = bay.lower_stringers || [];
  loStg.forEach(function(s) {{
    var g = stringerPolygons(s, false);
    traces.push({{
      x: g.flange.x, y: g.flange.z,
      mode: 'lines', name: 'Lower stringer flange — ' + s.name,
      line: {{ color: '#1565a8', width: 1 }},
      fill: 'toself', fillcolor: 'rgba(21,101,168,0.45)',
      showlegend: false, hoverinfo: 'skip'
    }});
    traces.push({{
      x: g.web.x, y: g.web.z,
      mode: 'lines', name: 'Lower stringer web — ' + s.name,
      line: {{ color: '#1565a8', width: 1 }},
      fill: 'toself', fillcolor: 'rgba(21,101,168,0.55)',
      showlegend: false, hoverinfo: 'skip'
    }});
  }});
  if (loStg.length) {{
    traces.push({{
      x: loStg.map(function(s) {{ return s.x; }}),
      y: loStg.map(function(s) {{ return s.z; }}),
      mode: 'markers', name: 'Lower stringers',
      showlegend: false,
      marker: {{ size: 5, opacity: 0.01, color: '#1f77b4', symbol: 'circle',
        line: {{ width: 0 }} }},
      customdata: loStg.map(function(s) {{
        return ['stringer', 'Lower', s.name, s.material, s.plies, s.width, s.height, s.thickness, s.x, s.z, s.area];
      }}),
      hovertemplate: '<b>Lower stringer — %{{customdata[2]}}</b><br>Material: %{{customdata[3]}}<br>Plies: %{{customdata[4]}}<br>width=%{{customdata[5]:.3f}} in, height=%{{customdata[6]:.3f}} in, t=%{{customdata[7]:.4f}} in<br>Area: %{{customdata[10]:.4f}} in²<br>Y=%{{customdata[8]:.3f}} in, Z=%{{customdata[9]:.3f}} in<extra></extra>'
    }});
  }}

  traces.push({{
    x: [bay.YCG_geom], y: [bay.ZCG_geom],
    mode: 'markers', name: 'Wingbox section CG',
    showlegend: false,
    marker: {{
      size: 8, symbol: 'cross', color: '#2f2f2f',
      line: {{ width: 1, color: '#f7dc6f' }}
    }},
    customdata: [['section', 'Wingbox section CG (geom)', bay.YCG_geom, bay.ZCG_geom]],
    hovertemplate: '<b>%{{customdata[1]}}</b><br>Y=%{{customdata[2]:.3f}} in<br>Z=%{{customdata[3]:.3f}} in<extra></extra>'
  }});

  traces.push({{
    x: [bay.YCG], y: [bay.ZCG],
    mode: 'markers', name: 'Neutral axis',
    showlegend: false,
    marker: {{
      size: 8, symbol: 'x', color: '#c0392b',
      line: {{ width: 1, color: '#f7dc6f' }}
    }},
    customdata: [['section', 'Neutral axis (E-weighted)', bay.YCG, bay.ZCG]],
    hovertemplate: '<b>%{{customdata[1]}}</b><br>Y=%{{customdata[2]:.3f}} in<br>Z=%{{customdata[3]:.3f}} in<extra></extra>'
  }});

  function pushHorizontalDim(x0, x1, y, color) {{
    traces.push({{
      x: [x0, x1], y: [y, y], mode: 'lines',
      line: {{ color: color, width: 2 }}, showlegend: false, hoverinfo: 'skip'
    }});
    traces.push({{
      x: [x0, x1], y: [y, y], mode: 'markers',
      marker: {{
        symbol: ['triangle-left', 'triangle-right'], size: 11, color: color,
        line: {{ width: 0 }}
      }},
      showlegend: false, hoverinfo: 'skip'
    }});
  }}
  function pushVerticalDim(x, z0, z1, color) {{
    traces.push({{
      x: [x, x], y: [z0, z1], mode: 'lines',
      line: {{ color: color, width: 2 }}, showlegend: false, hoverinfo: 'skip'
    }});
    traces.push({{
      x: [x, x], y: [z0, z1], mode: 'markers',
      marker: {{
        symbol: ['triangle-down', 'triangle-up'], size: 11, color: color,
        line: {{ width: 0 }}
      }},
      showlegend: false, hoverinfo: 'skip'
    }});
  }}

  var yFc = yMin - dimOff * 0.7;
  var yBc = yMax + dimOff * 0.7;
  var xHt = -dimOff * 0.5;

  // [AI] The Optimization tab draws this same section zoomed to the wingbox, where a
  // dimension measured to the LE or the TE runs off-screen and leaves a stray
  // arrowhead at the edge. Those two are for the full-section view only.
  var showEdgeDims = !options.wingboxZoom;

  pushHorizontalDim(0, chord, yFc, '#7b2d8e');
  // The box chord and the two runs either side of it share one line, so they read as
  // a chain across the whole chord: LE to fwd spar, the box, aft spar to TE. The
  // three add up to the full chord dimension drawn below them.
  if (showEdgeDims) {{ pushHorizontalDim(0, xFwd, yBc, '#5d6d7e'); }}
  pushHorizontalDim(xFwd, xAft, yBc, '#e67e22');
  if (showEdgeDims) {{ pushHorizontalDim(xAft, chord, yBc, '#5d6d7e'); }}
  pushVerticalDim(xHt, zBot, zTop, '#008080');

  var annotations = [
    {{ x: chord / 2, y: yFc - dimOff * 0.35, text: 'Full chord = ' + chord.toFixed(1) + ' in',
       showarrow: false, font: {{ size: 12, color: '#7b2d8e' }} }},
    {{ x: (xFwd + xAft) / 2, y: yBc + dimOff * 0.35,
       text: 'Box chord = ' + bay.box_chord.toFixed(1) + ' in',
       showarrow: false, font: {{ size: 12, color: '#e67e22' }} }},
    {{ x: xHt - dimOff * 0.55, y: (zTop + zBot) / 2,
       text: 'h = ' + sectH.toFixed(2) + ' in', textangle: -90,
       showarrow: false, font: {{ size: 11, color: '#008080' }} }},
    // Fwd spar height
    {{ x: xFwd, y: (zUF + zLF) / 2 , xanchor: 'left', xshift: 6,
       text: bay.spar_h_fwd.toFixed(2) + ' in',
       showarrow: false, font: {{ size: 10, color: '#d62728' }},
       bgcolor: 'rgba(255,255,255,0.8)' }},
    // Aft spar height
    {{ x: xAft, y: (zUA + zLA) / 2, xanchor: 'right', xshift: -6,
       text: bay.spar_h_aft.toFixed(2) + ' in',
       showarrow: false, font: {{ size: 10, color: '#2ca02c' }},
       bgcolor: 'rgba(255,255,255,0.8)' }},
  ];

  // LE to fwd spar, and aft spar to TE. Each carries the chord ratio of the spar it is
  // anchored on, on the line above the distance -- those are the numbers
  // sparRatios.csv holds, and the run underneath is the leading or trailing edge
  // structure that spar leaves room for. Both ratios are this bay's own, so they move
  // across the span. One two-line annotation rather than two stacked ones: Plotly does
  // not autorange on annotations, so a second row placed further out would clip on a
  // narrow tip section.
  if (showEdgeDims) {{
    annotations.push(
      {{ x: xFwd / 2, y: yBc + dimOff * 0.35,
         text: '(fwd x/c ' + bay.fwd_ratio.toFixed(4) + ')<br>From LE = '
               + xFwd.toFixed(1) + ' in',
         showarrow: false, font: {{ size: 11, color: '#5d6d7e' }} }},
      {{ x: (xAft + chord) / 2, y: yBc + dimOff * 0.35,
         text: '(aft x/c ' + bay.aft_ratio.toFixed(4) + ')<br>From TE = '
               + (chord - xAft).toFixed(1) + ' in',
         showarrow: false, font: {{ size: 11, color: '#5d6d7e' }} }}
    );
  }}

  // The flap and the aileron, each drawn on the stations it spans.
  // bay.control_surfaces is empty everywhere neither reaches, and only updateXsec
  // sets the flag: the Optimization tab calls this function too, for a wingbox zoom
  // no control surface is anywhere near. The Stress Results tab draws its own section
  // through drawResultsXsecPlot and is untouched either way. The outlines are already
  // in this plot's axes, so they need no transform, and they sit inside the airfoil
  // envelope, so they never widen the axis ranges. Both surfaces are drawn out to the
  // OML, so where an MFS sits chordwise on top of one, the drawn section occupies
  // space the spoiler really holds -- see control_surface_geom.py.
  if (options.showControlSurfaces && bay.control_surfaces) {{
    bay.control_surfaces.forEach(function(cs) {{
      var spec = surfaceSpec(cs.name);
      var fill = spec ? spec.color : 'rgba(127, 127, 127, 0.10)';
      var line = planformLineColor(fill);
      traces.push(
        {{ x: cs.oml.x, y: cs.oml.z, mode: 'lines', name: cs.label,
           fill: 'toself', fillcolor: fill,
           line: {{ color: line, width: 1.8 }},
           hovertemplate: cs.label
             + ' OML<br>X: %{{x:.2f}} in<br>Z: %{{y:.2f}} in<extra></extra>' }},
        {{ x: cs.mid.x, y: cs.mid.z, mode: 'lines',
           name: cs.label + ' mid-line',
           line: {{ color: line, width: 1, dash: 'dash' }},
           hovertemplate: cs.label
             + ' mid-line<br>X: %{{x:.2f}} in<br>Z: %{{y:.2f}} in<extra></extra>' }}
      );
    }});
  }}

  // Leader lines for section height only (dimension line uses markers above)
  var shapes = [
    {{ type: 'line', x0: AF.x[maxIdx] * chord, y0: zTop, x1: xHt, y1: zTop,
       line: {{ color: '#aaa', width: 0.8, dash: 'dot' }} }},
    {{ type: 'line', x0: AF.x[maxIdx] * chord, y0: zBot, x1: xHt, y1: zBot,
       line: {{ color: '#aaa', width: 0.8, dash: 'dot' }} }},
  ];

  var layout = {{
    xaxis: {{ title: 'Y (in)' }},
    yaxis: {{ title: 'Z (in)', scaleanchor: 'x', scaleratio: 1 }},
    title: {{ text: 'Bay ' + bay.bay_id + '  |  WS = ' + bay.ws.toFixed(1) + ' in  |  t/c = ' + bay.t_c.toFixed(4),
              font: {{ size: 14 }} }},
    margin: {{ t: 50, b: 50, l: 60, r: 20 }},
    annotations: annotations,
    shapes: shapes,
    showlegend: false,
    hovermode: 'closest'
  }};
  if (options.wingboxZoom) {{
    if (options.fixedRange) {{
      layout.xaxis.range = options.fixedRange.x.slice();
      layout.yaxis.range = options.fixedRange.y.slice();
    }} else {{
      var xPad = Math.max(0.05 * (xAft - xFwd), 2.0);
      var zVals = [zUF, zLF, zUA, zLA];
      (bay.upper_stringers || []).forEach(function(s) {{ zVals.push(s.z); }});
      (bay.lower_stringers || []).forEach(function(s) {{ zVals.push(s.z); }});
      var zMin = Math.min.apply(null, zVals);
      var zMax = Math.max.apply(null, zVals);
      var zPad = Math.max(0.08 * (zMax - zMin), 1.0);
      layout.xaxis.range = [xFwd - xPad, xAft + xPad];
      layout.yaxis.range = [zMin - zPad, zMax + zPad];
    }}
    layout.margin = {{ t: 45, b: 45, l: 55, r: 15 }};
  }} else if (options.fixedRange) {{
    layout.xaxis.range = options.fixedRange.x.slice();
    layout.yaxis.range = options.fixedRange.y.slice();
  }} else {{
    // Explicit, so switching back from fixed scale drops the pinned range.
    layout.xaxis.autorange = true;
    layout.yaxis.autorange = true;
  }}

  var el = document.getElementById(targetId);
  var cfg = {{ responsive: true }};
  if (!el.data || el.data.length === 0) {{
    Plotly.newPlot(targetId, traces, layout, cfg).then(function() {{
      if (targetId === 'xsec-plot') bindXsecPlotClickOnce();
    }});
  }} else {{
    Plotly.react(targetId, traces, layout, cfg);
  }}
}}

function buildXsecComponentsTable(bay) {{
  var tbl = document.getElementById('xsec-components-table');
  if (!tbl) return;
  // [OAS] Explicit empty state instead of a fabricated parts list. This table is a
  // laminate bill of materials -- each stringer, skin panel and spar cap with its
  // material, its ply count and its dimensions. OAS's wingbox is two smeared
  // thicknesses of one isotropic-equivalent material: there are no stringers, no
  // panels, no caps and no plies to list, and inventing rows that add up to the right
  // area would put engineering detail on the page that no analysis produced. The
  // thicknesses themselves are real and are in the Cross Section Properties table
  // above and drawn on the section beside it.
  if (bay && bay.components_note) {{
    tbl.innerHTML = '<tbody><tr><td style="white-space:normal;line-height:1.45;'
      + 'font-family:-apple-system,BlinkMacSystemFont,\\'Segoe UI\\',sans-serif;'
      + 'font-size:13px;color:#7f8c8d">'
      + escapeHtml(bay.components_note) + '</td></tr></tbody>';
    return;
  }}
  var rows = [];
  function addRow(typ, nam, mat, plies, dims, xk, zk) {{
    rows.push({{ typ: typ, nam: nam, mat: mat, plies: plies, dims: dims, xk: xk, zk: zk }});
  }}
  function nz(v, dig) {{
    if (v === undefined || v === null || v === '' || v === '—') return '—';
    var n = Number(v);
    if (isNaN(n)) return '—';
    return n.toFixed(dig);
  }}
  (bay.upper_skin_panels || []).forEach(function(p) {{
    addRow('Skin (upper)', p.name, p.material, p.plies,
      'a=' + Number(p.panel_length).toFixed(4) + ' in, t=' + Number(p.thickness).toFixed(4)
        + ' in, pitch=' + Number(p.bay_width).toFixed(2) + ' in, R_chord='
        + (p.R_chord == null ? 'flat' : Number(p.R_chord).toFixed(2) + ' in')
        + ', ex_scale=' + Number(p.ex_scale).toFixed(4),
      p.x, p.z);
  }});
  (bay.lower_skin_panels || []).forEach(function(p) {{
    addRow('Skin (lower)', p.name, p.material, p.plies,
      'a=' + Number(p.panel_length).toFixed(4) + ' in, t=' + Number(p.thickness).toFixed(4)
        + ' in, pitch=' + Number(p.bay_width).toFixed(2) + ' in, R_chord='
        + (p.R_chord == null ? 'flat' : Number(p.R_chord).toFixed(2) + ' in')
        + ', ex_scale=' + Number(p.ex_scale).toFixed(4),
      p.x, p.z);
  }});
  (bay.upper_stringers || []).forEach(function(s) {{
    addRow('Stringer (upper)', s.name, s.material, s.plies,
      'w=' + Number(s.width).toFixed(3) + ' in, h=' + Number(s.height).toFixed(3) + ' in, t=' + Number(s.thickness).toFixed(4) + ' in',
      s.x, s.z);
  }});
  (bay.lower_stringers || []).forEach(function(s) {{
    addRow('Stringer (lower)', s.name, s.material, s.plies,
      'w=' + Number(s.width).toFixed(3) + ' in, h=' + Number(s.height).toFixed(3) + ' in, t=' + Number(s.thickness).toFixed(4) + ' in',
      s.x, s.z);
  }});
  function sparRows(sp, label) {{
    if (!sp) return;
    var w = sp.web;
    addRow('Spar web (' + label + ')', w.name, w.material, w.plies,
      'h=' + Number(w.height).toFixed(2) + ' in, t=' + Number(w.thickness).toFixed(4) + ' in', w.x, w.z);
    var uc = sp.upper_cap;
    addRow('Spar cap (' + label + ' upper)', uc.name, uc.material, uc.plies,
      'w=' + Number(uc.width).toFixed(3) + ' in, t=' + Number(uc.thickness).toFixed(4) + ' in'
        + '; skin leg Y=' + Number(uc.x_skin_leg).toFixed(3) + ' Z=' + Number(uc.z_skin_leg).toFixed(3)
        + '; web leg Y=' + Number(uc.x_web_leg).toFixed(3) + ' Z=' + Number(uc.z_web_leg).toFixed(3),
      uc.x_skin_leg, uc.z_skin_leg);
    addRow('Spar cap (' + label + ' upper web leg)', uc.name, uc.material, uc.plies,
      'web leg', uc.x_web_leg, uc.z_web_leg);
    var lc = sp.lower_cap;
    addRow('Spar cap (' + label + ' lower)', lc.name, lc.material, lc.plies,
      'w=' + Number(lc.width).toFixed(3) + ' in, t=' + Number(lc.thickness).toFixed(4) + ' in'
        + '; skin leg Y=' + Number(lc.x_skin_leg).toFixed(3) + ' Z=' + Number(lc.z_skin_leg).toFixed(3)
        + '; web leg Y=' + Number(lc.x_web_leg).toFixed(3) + ' Z=' + Number(lc.z_web_leg).toFixed(3),
      lc.x_skin_leg, lc.z_skin_leg);
    addRow('Spar cap (' + label + ' lower web leg)', lc.name, lc.material, lc.plies,
      'web leg', lc.x_web_leg, lc.z_web_leg);
  }}
  sparRows(bay.fwd_spar, 'fwd');
  sparRows(bay.aft_spar, 'aft');

  var html = '<thead><tr>'
    + '<th>Type</th><th>Name</th><th>Material</th><th>Plies</th><th>Dimensions</th><th>Y (in)</th><th>Z (in)</th>'
    + '</tr></thead><tbody>';
  rows.forEach(function(r) {{
    html += '<tr>'
      + '<td>' + escapeHtml(r.typ) + '</td>'
      + '<td>' + escapeHtml(r.nam) + '</td>'
      + '<td>' + escapeHtml(r.mat) + '</td>'
      + '<td>' + escapeHtml(String(r.plies)) + '</td>'
      + '<td>' + escapeHtml(r.dims) + '</td>'
      + '<td>' + nz(r.xk, 3) + '</td>'
      + '<td>' + nz(r.zk, 3) + '</td>'
      + '</tr>';
  }});
  html += '</tbody>';
  tbl.innerHTML = html;
}}

function drawPropsTable(bay) {{
  var rows = [
    ['Eref',     bay.E,     'psi', fmtSci,
      'Reference Young modulus (maximum Ex in the section).'],
    ['G_ref', bay.G_ref, 'psi', fmtSci,
      'Reference shear modulus (maximum Gxy in the section).'],
    ['A',     bay.A,     'in\\u00B2', fmt4,
      'Transformed effective area: sum of A_i times (E_i / E_ref).'],
    ['A_geom', bay.A_geom, 'in\\u00B2', fmt4,
      'Physical gross area.'],
    ['spacer'],
    ['YCG',   bay.YCG,   'in', fmt2,
      'Neutral axis Y in section coords (stiffness-weighted centroid, LE origin, +Y aft).'],
    ['ZCG',   bay.ZCG,   'in', fmt2,
      'Neutral axis Z in section coords (+Z up).'],
    ['spacer'],
    ['IYCG',  bay.IYCG,  'in\\u2074', fmt1,
      'Effective Iy about the neutral axis (vertical bending, horizontal neutral axis).'],
    ['IZCG',  bay.IZCG,  'in\\u2074', fmt1,
      'Effective Iz about neutral axis.'],
    ['IYZCG', bay.IYZCG, 'in\\u2074', fmt1,
      'Product of inertia about neutral axes.'],
    ['J',     bay.J,     'in\\u2074', fmt1,
      'Bredt-Batho torsional constant for the single closed cell.'],
    ['Am',    bay.Am,    'in\\u00B2', fmt1,
      'Enclosed cell area inside skin between spars (shear flow / Bredt).'],
    ['spacer'],
    ['EA',    bay.EA,    'in\\u00B2\\u00B7psi', fmtSci,
      'Axial stiffness: E_ref times effective area.'],
    ['EI',    bay.E * bay.IYCG,    'psi\\u00B7in\\u2074', fmtSci,
      'Bending stiffness: E_ref times IYCG.'],
    ['GJ',    bay.GJ,    'psi\\u00B7in\\u2074', fmtSci,
      'Torsional rigidity: G_ref times J.'],
    ['EI/(GJ)',    bay.E * bay.IYCG / bay.GJ,     '', fmtSci,
      'Bending to torsional stiffness ratio.'],
  ];

  var html = '<thead><tr>'
    + '<th>Property</th><th>Value</th><th>Units</th><th>Description</th>'
    + '</tr></thead><tbody>';

  rows.forEach(function(r) {{
    if (r[0] === 'spacer') {{
      html += '<tr class="spacer"><td colspan="4"></td></tr>';
    }} else {{
      var valStr = r[3](r[1]);
      var desc = r[4].replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      html += '<tr>'
        + '<td>' + r[0] + '</td>'
        + '<td><input class="val" readonly value="' + valStr + '" '
        +   'onclick="this.select()" title="Click to select"></td>'
        + '<td>' + r[2] + '</td>'
        + '<td>' + desc + '</td>'
        + '</tr>';
    }}
  }});
  html += '</tbody>';
  document.getElementById('props-table').innerHTML = html;
}}

// [OAS] All four formatters print an em dash for a value that is not a finite
// number, instead of throwing on null or writing "NaN" into a table. OAS does not
// compute every quantity WingCalc does -- there is no product of inertia, for one --
// and those arrive as null in the payload on purpose: JSON has no NaN, and a bare NaN
// token would make the embedded blob unparseable as JSON, which is the report's one
// cheap self-check. A dash says "not computed"; "NaN" reads like a bug.
function fmtOr(v, digits) {{
  var n = Number(v);
  if (v === null || v === undefined || !isFinite(n)) return '\\u2014';
  return n.toFixed(digits);
}}
function fmt1(v) {{ return fmtOr(v, 1); }}
function fmt2(v) {{ return fmtOr(v, 2); }}
function fmt4(v) {{ return fmtOr(v, 4); }}
function fmtSci(v) {{
  var n = Number(v);
  if (v === null || v === undefined || !isFinite(n)) return '\\u2014';
  if (Math.abs(n) >= 1e6) return n.toExponential(4);
  return n.toFixed(1);
}}

// ---- Results ----
var resultsSelectedBay = 0;
var resultsSelectedLoadcase = 'worst';
var resultsPinnedDetail = null;

function getResultsBayData(bayIndex) {{
  if (!RESULTS || !RESULTS.bays) return null;
  var b = RESULTS.bays[bayIndex];
  if (!b) return null;
  if (resultsSelectedLoadcase === 'worst') return b;
  var lc = b.by_loadcase && b.by_loadcase[resultsSelectedLoadcase];
  if (!lc) return b;
  return {{
    bay_id: b.bay_id,
    ws: b.ws,
    bay_index: b.bay_index,
    ms: lc.ms,
    stringers: lc.stringers,
    skins: lc.skins,
    spar_caps: lc.spar_caps,
    spar_webs: lc.spar_webs,
    cross_section_summary: lc.cross_section_summary
  }};
}}

function getResultsBayMs(bayIndex) {{
  var data = getResultsBayData(bayIndex);
  return data && data.ms ? data.ms : null;
}}

function msClass(v) {{
  var n = Number(v);
  if (isNaN(n) || n >= 900) return 'ms-ok';
  if (n < 0) return 'ms-neg';
  if (n < 0.25) return 'ms-low';
  return 'ms-ok';
}}

function formatMsVal(v, decimals) {{
  var n = Number(v);
  if (isNaN(n) || n >= 900) return '—';
  var d = decimals === undefined ? 3 : decimals;
  return n.toFixed(d);
}}

function createXsecSurfaceHelpers(bay) {{
  var chord = bay.chord;
  var tc = bay.t_c;
  var AF = bay.airfoil || AIRFOIL;
  var tcScale = tc / AF.ref_tc;
  var xFwd = chord * bay.fwd_ratio;
  var xAft = chord * bay.aft_ratio;

  function interp(xq) {{
    var ax = AF.x;
    for (var i = 1; i < ax.length; i++) {{
      if (ax[i] >= xq) {{
        var t = (xq - ax[i-1]) / (ax[i] - ax[i-1]);
        return function(arr) {{ return arr[i-1] + t * (arr[i] - arr[i-1]); }};
      }}
    }}
    return function(arr) {{ return arr[arr.length - 1]; }};
  }}
  function zSurfPhys(xcn, zCurve) {{
    var fi = interp(xcn);
    return fi(zCurve) * chord * tcScale;
  }}
  function dZdYSurf(xcn, zCurve) {{
    var ax = AF.x;
    var dxC = Math.max(1e-6, 1e-4 * Math.max(ax[ax.length - 1] - ax[0], 0.1));
    var xc0 = Math.max(ax[0], Math.min(ax[ax.length - 1], xcn));
    var xcLo = Math.max(ax[0], xc0 - dxC);
    var xcHi = Math.min(ax[ax.length - 1], xc0 + dxC);
    if (xcHi <= xcLo + 1e-12) {{ xcLo = ax[0]; xcHi = ax[ax.length - 1]; }}
    var zLo = zSurfPhys(xcLo, zCurve);
    var zHi = zSurfPhys(xcHi, zCurve);
    return (zHi - zLo) / ((xcHi - xcLo) * chord);
  }}
  function skinPanelPolygon(panel, isUpper) {{
    var xl = panel.x_left / chord, xr = panel.x_right / chord;
    var ax = AF.x;
    xl = Math.max(ax[0], Math.min(ax[ax.length - 1], xl));
    xr = Math.max(ax[0], Math.min(ax[ax.length - 1], xr));
    var zCurve = isUpper ? AF.upper : AF.lower;
    var nSamp = 20, dx = (xr - xl) / nSamp;
    var omlX = [], omlZ = [], imlX = [], imlZ = [];
    var tSkin = panel.thickness;
    for (var k = 0; k <= nSamp; k++) {{
      var xn = xl + k * dx;
      var zOml = zSurfPhys(xn, zCurve);
      var xP = xn * chord;
      var dzdy = dZdYSurf(xn, zCurve);
      var nrm = Math.hypot(dzdy, 1);
      var nx = isUpper ? dzdy / nrm : -dzdy / nrm;
      var nz = isUpper ? -1 / nrm : 1 / nrm;
      omlX.push(xP); omlZ.push(zOml);
      imlX.push(xP + tSkin * nx); imlZ.push(zOml + tSkin * nz);
    }}
    var px = omlX.concat(imlX.reverse());
    var pz = omlZ.concat(imlZ.reverse());
    px.push(px[0]); pz.push(pz[0]);
    return {{ x: px, z: pz }};
  }}
  function centroidOffsetFromBondStg(w, h, t) {{
    var A_fl = w * t, y_fl = t / 2;
    var h_web = h - t, A_web = 2 * t * h_web, y_web = t + h_web / 2;
    var A_tot = A_fl + A_web;
    if (A_tot <= 1e-15) return 0;
    return (A_fl * y_fl + A_web * y_web) / A_tot;
  }}
  function stringerPolygons(s, isUpper) {{
    var xcNorm = s.x / chord;
    xcNorm = Math.max(AF.x[0], Math.min(AF.x[AF.x.length - 1], xcNorm));
    var zCurve = isUpper ? AF.upper : AF.lower;
    var dzdy = dZdYSurf(xcNorm, zCurve);
    var nrm = Math.hypot(dzdy, 1);
    var nx = isUpper ? dzdy / nrm : -dzdy / nrm;
    var nz = isUpper ? -1 / nrm : 1 / nrm;
    var tnorm = Math.hypot(1, dzdy);
    var tx = 1 / tnorm, tz = dzdy / tnorm;
    var t = s.thickness, tw = 2 * t, w = s.width, h = s.height;
    var yBond = centroidOffsetFromBondStg(w, h, t);
    var bx = s.x - yBond * nx, bz = s.z - yBond * nz;
    var fOx = bx - 0.5 * w * tx, fOz = bz - 0.5 * w * tz;
    var fIx = bx + 0.5 * w * tx, fIz = bz + 0.5 * w * tz;
    var flange = {{
      x: [fOx, fIx, fIx + t * nx, fOx + t * nx, fOx],
      z: [fOz, fIz, fIz + t * nz, fOz + t * nz, fOz]
    }};
    var hWeb = h - t;
    var wbx = bx - 0.5 * tw * tx + t * nx, wbz = bz - 0.5 * tw * tz + t * nz;
    var web = {{
      x: [wbx, wbx + tw * tx, wbx + tw * tx + hWeb * nx, wbx + hWeb * nx, wbx],
      z: [wbz, wbz + tw * tz, wbz + tw * tz + hWeb * nz, wbz + hWeb * nz, wbz]
    }};
    return {{ flange: flange, web: web, cx: s.x, cz: s.z }};
  }}
  return {{
    chord: chord, tcScale: tcScale, xFwd: xFwd, xAft: xAft,
    skinPanelPolygon: skinPanelPolygon, stringerPolygons: stringerPolygons
  }};
}}

function getResultsRecord(bayIndex, kind, key) {{
  var b = getResultsBayData(bayIndex);
  if (!b) return null;
  if (kind === 'stringer') return b.stringers[key] || null;
  if (kind === 'skin') return b.skins[key] || null;
  if (kind === 'spar_cap') return b.spar_caps[key] || null;
  if (kind === 'spar_web') return b.spar_webs[key] || null;
  return null;
}}

var MS_FIELD_LABELS = {{
  'MS_buckling': 'MS Buckling',
  'MS_longitudinal_strength': 'MS Longitudinal strength',
  'MS_shear_strength': 'MS Shear strength',
  'MS_crippling': 'MS Crippling',
  'MS_elm_buckling': 'MS Element buckling',
  'MS_strength': 'MS Strength',
  'Min_inertia_check': 'Inertia check',
  'MS_cap_buckling': 'MS Cap buckling',
  'MS_cap_crippling': 'MS Cap crippling',
  'MS_cap_elm_buckling': 'MS Cap element buckling',
  'MS_cap_strength': 'MS Cap strength',
  'MS_web_buckling': 'MS Web buckling',
  'MS_web_longitudinal_strength': 'MS Web longitudinal strength',
  'MS_web_shear_strength': 'MS Web shear strength',
  // [OAS] The one failure mode OAS has. It is a combined-stress strength check --
  // bending plus axial plus shear at one point, against a laminate allowable -- so it
  // is not any of WingCalc's modes and is not filed as one.
  'MS_von_mises': 'MS von Mises strength'
}};

// [OAS] 'MS_von_mises' appended to the skin and spar-web orders. These lists are what
// minMsFromRecord scans, and a record whose only MS key is absent from the list comes
// back with no margin at all -- so leaving it out would silently drop every OAS callout
// off the cross-section. WingCalc's own keys are untouched, so a WingCalc payload still
// renders through this file unchanged.
var MS_FIELD_ORDER = {{
  stringer: ['MS_buckling', 'MS_crippling', 'MS_elm_buckling', 'MS_strength', 'Min_inertia_check'],
  skin: ['MS_buckling', 'MS_longitudinal_strength', 'MS_shear_strength', 'MS_von_mises'],
  spar_cap: ['MS_cap_buckling', 'MS_cap_crippling', 'MS_cap_elm_buckling', 'MS_cap_strength'],
  spar_web: ['MS_web_buckling', 'MS_web_longitudinal_strength', 'MS_web_shear_strength',
             'MS_von_mises']
}};

function msFieldLabel(key) {{
  return MS_FIELD_LABELS[key] || key.replace(/^MS_/, '').replace(/_/g, ' ');
}}

function msRecordHoverText(rec, displayName, kind) {{
  if (!rec) return 'No MS data';
  var name = displayName
    || rec['Panel Name'] || rec['Stringer Name'] || rec['Spar Name'] || 'Component';
  if (!displayName && rec['Cap']) name += ' (' + rec['Cap'] + ' cap)';
  if (!displayName && rec['Spar Name'] && kind === 'spar_web') name = rec['Spar Name'] + ' web';
  var lines = ['<b>' + escapeHtml(name) + '</b>'];
  var order = (kind && MS_FIELD_ORDER[kind]) ? MS_FIELD_ORDER[kind] : [];
  var msKeys = [];
  order.forEach(function(k) {{
    if (rec[k] !== undefined) msKeys.push(k);
  }});
  if (!order.length) {{
    Object.keys(rec).forEach(function(k) {{
      if ((k.indexOf('MS_') === 0 || k === 'Min_inertia_check') && msKeys.indexOf(k) < 0) {{
        msKeys.push(k);
      }}
    }});
  }}
  if (!msKeys.length) {{
    lines.push('No MS data');
  }} else {{
    msKeys.forEach(function(k) {{
      lines.push(escapeHtml(msFieldLabel(k)) + ': ' + escapeHtml(formatMsVal(rec[k], 2)));
    }});
  }}
  return lines.join('<br>');
}}

function minMsFromRecord(rec, kind) {{
  if (!rec) return null;
  var order = (kind && MS_FIELD_ORDER[kind]) ? MS_FIELD_ORDER[kind] : null;
  var best = null;
  function consider(k) {{
    if (k.indexOf('MS_') !== 0 && k !== 'Min_inertia_check') return;
    var ms = parseFloat(rec[k]);
    if (isNaN(ms) || ms >= 900) return;
    if (!best || ms < best.ms) best = {{ ms: ms, failureMode: k }};
  }}
  if (order) {{
    order.forEach(consider);
  }} else {{
    Object.keys(rec).forEach(consider);
  }}
  return best;
}}

function msAnnotationColor(ms) {{
  if (ms < 0) return '#c0392b';
  if (ms < 0.25) return '#d68910';
  return '#1e8449';
}}

function centroidOfPoly(poly) {{
  if (!poly || !poly.x || !poly.z || poly.x.length < 2) return null;
  var n = poly.x.length - 1;
  var sx = 0, sz = 0;
  for (var i = 0; i < n; i++) {{ sx += poly.x[i]; sz += poly.z[i]; }}
  return {{ x: sx / n, z: sz / n }};
}}

function findWorstStringer(bayIndex, upper) {{
  var bay = BAYS[bayIndex];
  var rb = getResultsBayData(bayIndex);
  if (!bay || !rb) return null;
  var stgs = upper ? (bay.upper_stringers || []) : (bay.lower_stringers || []);
  var best = null;
  stgs.forEach(function(s) {{
    var rec = rb.stringers && rb.stringers[s.name];
    var msInfo = minMsFromRecord(rec, 'stringer');
    if (!msInfo) return;
    if (!best || msInfo.ms < best.ms) {{
      best = {{
        name: s.name,
        x: s.x,
        z: s.z,
        ms: msInfo.ms
      }};
    }}
  }});
  return best;
}}

function findWorstSkinPanel(bayIndex, upper) {{
  var bay = BAYS[bayIndex];
  var rb = getResultsBayData(bayIndex);
  if (!bay || !rb) return null;
  var panels = upper ? (bay.upper_skin_panels || []) : (bay.lower_skin_panels || []);
  var excludeName = null;
  if (!upper) {{
    excludeName = bay.cutout_panel_name || null;
  }}
  var best = null;
  panels.forEach(function(p) {{
    if (excludeName && p.name === excludeName) return;
    var rec = rb.skins && rb.skins[p.name];
    var msInfo = minMsFromRecord(rec, 'skin');
    if (!msInfo) return;
    if (!best || msInfo.ms < best.ms) {{
      best = {{
        name: p.name,
        x: p.x,
        z: p.z,
        ms: msInfo.ms
      }};
    }}
  }});
  return best;
}}

function findWorstCutoutSkinPanel(bayIndex) {{
  var bay = BAYS[bayIndex];
  var rb = getResultsBayData(bayIndex);
  if (!bay || !rb) return null;
  var coIdx = bay.cutout_stg_index;
  if (coIdx == null || coIdx < 0) return null;
  // [AI mod] Merged cut-out skin bay (topology / wing_calc co_label), not adjacent
  // panels that get q_cutout.
  var targetName = bay.cutout_panel_name;
  var panels = bay.lower_skin_panels || [];
  var p = null;
  for (var i = 0; i < panels.length; i++) {{
    if (panels[i].name === targetName) {{ p = panels[i]; break; }}
  }}
  if (!p && coIdx < panels.length) p = panels[coIdx];
  if (!p) return null;
  var rec = rb.skins && rb.skins[p.name];
  var msInfo = minMsFromRecord(rec, 'skin');
  if (!msInfo) return null;
  return {{
    name: p.name,
    x: p.x,
    z: p.z,
    ms: msInfo.ms
  }};
}}

function findSparCapPoint(bay, label, capKey) {{
  var sp = (label === 'Fwd') ? bay.fwd_spar : bay.aft_spar;
  var d = sp && sp.draw;
  if (!d) return null;
  // Prefer the skin-leg polygon for each cap (one point per cap).
  if (capKey === 'upper') return centroidOfPoly(d.upper_skin_leg || d.upper_web_leg);
  return centroidOfPoly(d.lower_skin_leg || d.lower_web_leg);
}}

function findSparWebPoint(bay, label) {{
  var sp = (label === 'Fwd') ? bay.fwd_spar : bay.aft_spar;
  var d = sp && sp.draw;
  if (!d) return null;
  return centroidOfPoly(d.web);
}}

function buildResultsXsecAnnotations(bayIndex, xFwd, xAft, zMid) {{
  var bay = BAYS[bayIndex];
  var rb = RESULTS && RESULTS.bays[bayIndex];
  if (!bay || !rb) return [];

  var xCenter = (xFwd + xAft) / 2;
  var zCenter = zMid;
  var pending = [];
  var ANNO_STAGGER = 30;

  function queueAnno(x, z, ms, text, stagger, outboardX) {{
    pending.push({{
      x: x, z: z, ms: ms, text: text,
      stagger: stagger || 0,
      outboardX: !!outboardX
    }});
  }}

  function buildAnnoObj(item) {{
    var color = msAnnotationColor(item.ms);
    var dx = xCenter - item.x;
    var dz = zCenter - item.z;
    var ax = 0;
    var ay = 0;
    var lenX = 30 + item.stagger;
    var lenZ = 52 + item.stagger;
    if (Math.abs(dx) > 1e-6) {{
      var signX = item.outboardX ? (dx > 0 ? -1 : 1) : (dx > 0 ? 1 : -1);
      ax = signX * lenX;
    }}
    if (Math.abs(dz) > 1e-6) ay = (dz < 0 ? 1 : -1) * lenZ;
    return {{
      x: item.x, y: item.z, xref: 'x', yref: 'y',
      text: item.text,
      showarrow: true,
      arrowhead: 2,
      arrowsize: 1,
      arrowwidth: 1.5,
      arrowcolor: color,
      ax: ax,
      ay: ay,
      xanchor: ax === 0 ? 'center' : (ax > 0 ? 'left' : 'right'),
      yanchor: ay === 0 ? 'middle' : (ay < 0 ? 'bottom' : 'top'),
      font: {{ size: 12, color: color, family: 'Consolas, monospace' }},
      bgcolor: 'rgba(255, 255, 255, 0.92)',
      bordercolor: '#bdc3c7',
      borderwidth: 1,
      borderpad: 3,
      // [AI] Without this Plotly does not raise plotly_clickannotation, which is what
      // brings a buried box to the front -- see bindResultsXsecPlotHandlers.
      captureevents: true
    }};
  }}

  // Skins: outboard chordwise, inward vertically; stringers inward both ways.
  var skU = findWorstSkinPanel(bayIndex, true);
  if (skU) queueAnno(skU.x, skU.z, skU.ms, skU.name + '<br>MS ' + formatMsVal(skU.ms, 2), 0, true);
  var stgU = findWorstStringer(bayIndex, true);
  if (stgU) queueAnno(stgU.x, stgU.z, stgU.ms, stgU.name + '<br>MS ' + formatMsVal(stgU.ms, 2), ANNO_STAGGER);

  var skL = findWorstSkinPanel(bayIndex, false);
  if (skL) queueAnno(skL.x, skL.z, skL.ms, skL.name + '<br>MS ' + formatMsVal(skL.ms, 2), 0, true);
  var skCO = findWorstCutoutSkinPanel(bayIndex);
  var lowerStagger = skL ? ANNO_STAGGER : 0;
  if (skCO) queueAnno(skCO.x, skCO.z, skCO.ms, 'Cut-out ' + skCO.name + '<br>MS ' + formatMsVal(skCO.ms, 2), lowerStagger, true);
  var stgL = findWorstStringer(bayIndex, false);
  var stgLStagger = 0;
  if (stgL) {{
    if (skL && skCO) stgLStagger = ANNO_STAGGER * 2;
    else if (skL || skCO) stgLStagger = ANNO_STAGGER;
    queueAnno(stgL.x, stgL.z, stgL.ms, stgL.name + '<br>MS ' + formatMsVal(stgL.ms, 2), stgLStagger);
  }}

  // Spar caps inward; webs outboard of the web (away from wingbox center in X).
  ['Fwd', 'Aft'].forEach(function(label) {{
    var hasUpperCap = false;
    ['upper', 'lower'].forEach(function(capKey) {{
      var pt = findSparCapPoint(bay, label, capKey);
      if (!pt) return;
      var rec = getResultsRecord(bayIndex, 'spar_cap', label + '|' + capKey);
      var msInfo = minMsFromRecord(rec, 'spar_cap');
      if (!msInfo) return;
      if (capKey === 'upper') hasUpperCap = true;
      queueAnno(pt.x, pt.z, msInfo.ms, label + ' ' + capKey + ' cap<br>MS ' + formatMsVal(msInfo.ms, 2), 0);
    }});
    var ptW = findSparWebPoint(bay, label);
    if (ptW) {{
      var recW = getResultsRecord(bayIndex, 'spar_web', label);
      var msInfoW = minMsFromRecord(recW, 'spar_web');
      if (msInfoW) {{
        var webStagger = hasUpperCap ? ANNO_STAGGER : 0;
        queueAnno(ptW.x, ptW.z, msInfoW.ms, label + ' web<br>MS ' + formatMsVal(msInfoW.ms, 2), webStagger, true);
      }}
    }}
  }});

  // Longer arrows first (behind); short arrows / boxes on top.
  pending.sort(function(a, b) {{ return b.stagger - a.stagger; }});
  var built = pending.map(buildAnnoObj);
  // [AI] Remember where each box was placed before spreadResultsXsecAnnotations moves it,
  // so a re-run starts from the same layout instead of pushing the boxes further out
  // every time. Keyed on the text because clicking a box reorders the array.
  RESULTS_XSEC_ANNO_BASE = {{}};
  built.forEach(function(a) {{ RESULTS_XSEC_ANNO_BASE[a.text] = {{ ax: a.ax, ay: a.ay }}; }});
  return built;
}}

// [AI] ---- Spreading the cross-section callouts ----
// buildResultsXsecAnnotations places each box at a fixed pixel offset from the item it
// points at, which puts two boxes on top of each other whenever two items sit close
// together -- the upper and lower cap of a shallow spar are the usual pair. Plotly draws
// annotations in array order and does nothing about collisions, so overlapping boxes are
// pulled apart here, once the plot is drawn and the pixel geometry is known.
//
// Only boxes that actually overlap move, so the placement above stays the starting point
// and a section whose callouts already clear each other is left untouched. A small plot
// has nowhere to move them to: below the size thresholds every box is put back where
// buildAnnoObj put it and nothing is spread.

var RESULTS_XSEC_ANNO_BASE = {{}};   // annotation text -> its as-built {{ax, ay}}

// Plotting area, in px, below which the boxes are left alone. A callout is roughly
// 120 x 40 px and a section carries up to about a dozen of them, so under this much
// room the spread only trades one overlap for another and pushes boxes out over the
// plot title and the axis labels -- which is what it was measured doing at 120 px.
// Height of this plotting area, by window: 1366x768 -> 168, 1600x900 -> 258,
// 1920x1080 -> 380, 2560x1440 -> 734. The cut takes in every one of those, each
// checked to come out with all its callouts separated.
var ANNO_SPREAD_MIN_W = 620;
var ANNO_SPREAD_MIN_H = 160;

// Box size from its text: 12px Consolas runs about 7.3 px per character on 15 px lines,
// plus borderpad 3 and the 1 px border on each side.
function annoBoxSize(text) {{
  var lines = String(text).split('<br>');
  var chars = 0;
  lines.forEach(function(l) {{ chars = Math.max(chars, l.length); }});
  return {{ w: chars * 7.3 + 8, h: lines.length * 15 + 8 }};
}}

function spreadResultsXsecAnnotations(el) {{
  var fl = el && el._fullLayout;
  var anns = el && el.layout && el.layout.annotations;
  if (!fl || !fl.xaxis || !fl.yaxis || !anns || anns.length < 2) return;
  var xa = fl.xaxis, ya = fl.yaxis;
  if (!xa.l2p || !ya.l2p) return;

  // Every box starts from its as-built offset, so this is idempotent: running it again
  // after a resize re-spreads the original layout rather than the previous spread.
  var boxes = anns.map(function(a) {{
    var base = RESULTS_XSEC_ANNO_BASE[a.text] || {{ ax: a.ax, ay: a.ay }};
    var size = annoBoxSize(a.text);
    var anchorX = xa._offset + xa.l2p(a.x);
    var anchorY = ya._offset + ya.l2p(a.y);
    var refX = anchorX + base.ax;
    var refY = anchorY + base.ay;
    return {{
      w: size.w, h: size.h, anchorX: anchorX, anchorY: anchorY,
      xanchor: a.xanchor, yanchor: a.yanchor,
      // Top-left corner, backed out of the point xanchor / yanchor pin the box by.
      x0: a.xanchor === 'left' ? refX
        : (a.xanchor === 'right' ? refX - size.w : refX - size.w / 2),
      y0: a.yanchor === 'top' ? refY
        : (a.yanchor === 'bottom' ? refY - size.h : refY - size.h / 2)
    }};
  }});

  if (xa._length >= ANNO_SPREAD_MIN_W && ya._length >= ANNO_SPREAD_MIN_H) {{
    // Boxes may spill into the plot margins as well as the plotting area -- on a tall
    // plot that is where most of the free room is -- so they are clamped to the graph
    // div rather than to the axes.
    var pad = 2;
    var maxX = fl.width - pad, maxY = fl.height - pad;
    for (var pass = 0; pass < 60; pass++) {{
      var moved = false;
      for (var i = 0; i < boxes.length; i++) {{
        for (var j = i + 1; j < boxes.length; j++) {{
          var A = boxes[i], B = boxes[j];
          var ovX = Math.min(A.x0 + A.w, B.x0 + B.w) - Math.max(A.x0, B.x0);
          var ovY = Math.min(A.y0 + A.h, B.y0 + B.h) - Math.max(A.y0, B.y0);
          if (ovX <= 0 || ovY <= 0) continue;
          // Separate along whichever axis needs the smaller move: a pair stacked one
          // above the other slides apart vertically, a pair side by side sideways.
          var step = Math.min(ovX, ovY) / 2 + 0.5;
          if (ovY <= ovX) {{
            var dirY = (A.y0 + A.h / 2 <= B.y0 + B.h / 2) ? -1 : 1;
            A.y0 += dirY * step; B.y0 -= dirY * step;
          }} else {{
            var dirX = (A.x0 + A.w / 2 <= B.x0 + B.w / 2) ? -1 : 1;
            A.x0 += dirX * step; B.x0 -= dirX * step;
          }}
          moved = true;
        }}
      }}
      boxes.forEach(function(b) {{
        b.x0 = Math.max(pad, Math.min(maxX - b.w, b.x0));
        b.y0 = Math.max(pad, Math.min(maxY - b.h, b.y0));
      }});
      if (!moved) break;
    }}
  }}

  var upd = {{}};
  boxes.forEach(function(b, i) {{
    var refX = b.xanchor === 'left' ? b.x0
      : (b.xanchor === 'right' ? b.x0 + b.w : b.x0 + b.w / 2);
    var refY = b.yanchor === 'top' ? b.y0
      : (b.yanchor === 'bottom' ? b.y0 + b.h : b.y0 + b.h / 2);
    upd['annotations[' + i + '].ax'] = refX - b.anchorX;
    upd['annotations[' + i + '].ay'] = refY - b.anchorY;
  }});
  Plotly.relayout(el, upd);
}}

// [OAS] Where a margin came from. WingCalc points at the per-component CSVs its stress
// pass writes; OAS has no such files -- every margin on this tab is one element of the
// `vonmises` array against one allowable -- so each entry names the OAS quantity
// instead. (The source file also read `info.note`, which its own dict never defined;
// the notes are filled in here.)
var RESULTS_DETAIL_CSV = {{
  skin: {{
    file: 'vonmises[elem, 0] (upper) / vonmises[elem, 1] (lower)',
    note: 'Bending at the extreme skin fibre plus the axial term, with torsion shear, '
      + 'from openaerostruct/structures/vonmises_wingbox.py.'
  }},
  spar_web: {{
    file: 'vonmises[elem, 2] (front) / vonmises[elem, 3] (rear)',
    note: 'Chordwise bending at the extreme spar fibre plus the axial term, with '
      + 'torsion and vertical shear, from vonmises_wingbox.py.'
  }}
}};

function setResultsDetailCsvSource(kind) {{
  var el = document.getElementById('results-detail-source');
  if (!el) return;
  var info = RESULTS_DETAIL_CSV[kind];
  if (!info) {{
    el.classList.remove('is-visible');
    el.innerHTML = '';
    return;
  }}
  el.classList.add('is-visible');
  // [OAS] 'Source:' rather than a path under Outputs/<run>/ -- there is no file.
  el.innerHTML = 'Source: <code>' + escapeHtml(info.file) + '</code>. '
    + escapeHtml(info.note);
}}

function renderKvTable(el, record) {{
  if (!el) return;
  if (!record) {{
    el.innerHTML = '';
    return;
  }}
  var html = '<thead><tr><th>Property</th><th>Value</th></tr></thead><tbody>';
  Object.keys(record).forEach(function(k) {{
    html += '<tr><td>' + escapeHtml(k) + '</td><td>' + escapeHtml(record[k]) + '</td></tr>';
  }});
  html += '</tbody>';
  el.innerHTML = html;
}}

function restoreResultsPinnedDetail() {{
  if (resultsPinnedDetail) {{
    showResultsDetail(resultsPinnedDetail, true);
    return;
  }}
  var ph = document.getElementById('results-detail-placeholder');
  var tbl = document.getElementById('results-detail-table');
  if (ph) {{
    ph.style.display = '';
    ph.textContent = 'Hover a component for a preview; click to pin details here.';
  }}
  if (tbl) tbl.innerHTML = '';
  setResultsDetailCsvSource(null);
}}

function showResultsDetail(cd, pinned) {{
  var ph = document.getElementById('results-detail-placeholder');
  var tbl = document.getElementById('results-detail-table');
  if (!cd || !cd.length) {{
    if (pinned) {{
      resultsPinnedDetail = null;
      restoreResultsPinnedDetail();
    }}
    return;
  }}
  if (pinned) resultsPinnedDetail = cd.slice();
  var rec = getResultsRecord(cd[1], cd[0], cd[2]);
  if (!rec) {{
    if (ph) {{
      ph.style.display = '';
      ph.textContent = 'No critical-loadcase record for this component.';
    }}
    if (tbl) tbl.innerHTML = '';
    setResultsDetailCsvSource(null);
    return;
  }}
  if (ph) ph.style.display = 'none';
  setResultsDetailCsvSource(cd[0]);
  renderKvTable(tbl, rec);
}}

function initResults() {{
  var emptyEl = document.getElementById('results-empty');
  var bodyEl = document.getElementById('results-body');
  if (!RESULTS || !RESULTS.bays || !RESULTS.bays.length) {{
    if (emptyEl) emptyEl.style.display = '';
    if (bodyEl) bodyEl.style.display = 'none';
    return;
  }}
  if (emptyEl) emptyEl.style.display = 'none';
  if (bodyEl) bodyEl.style.display = 'flex';
  var meta = document.getElementById('results-meta');
  if (meta) meta.textContent = 'Generated: ' + RESULTS.generated_at;
  resultsSelectedBay = 0;
  resultsSelectedLoadcase = 'worst';
  buildResultsLoadcaseSelect();
  buildResultsOverallSummary();
  buildResultsSelectedLoadcaseSummary();
  buildResultsCrossSectionSummary();
  buildResultsBaySelect();
  initResultsOverview();
}}

function buildOverallResultsTableHtml(o) {{
  if (!o || o.bay_id == null) return '';
  var colgroup = '<colgroup>'
    + '<col class="results-cs-col-ms"><col class="results-cs-col-bay">'
    + '<col class="results-cs-col-loc"><col class="results-cs-col-lc">'
    + '<col class="results-cs-col-mode"></colgroup>';
  return '<table class="results-cs-table results-cs-table-overall">'
    + colgroup + '<tbody>'
    + '<tr class="results-cs-h"><th>Min MS</th><th>Bay</th>'
    + '<th>Location</th><th>Loadcase</th><th>Failure mode</th></tr>'
    + '<tr class="results-cs-d"><td><span class="results-cs-ms ' + msClass(o.min_ms) + '">'
    + formatMsVal(o.min_ms, 2) + '</span></td>'
    + '<td>' + escapeHtml(String(o.bay_id)) + '</td>'
    + '<td>' + escapeHtml(o.location) + '</td>'
    + '<td>' + escapeHtml(o.lc) + '</td>'
    + '<td>' + escapeHtml(o.failure_mode) + '</td></tr>'
    + '</tbody></table>';
}}

function buildResultsOverallSummary() {{
  var el = document.getElementById('results-overall-summary');
  if (!el || !RESULTS || !RESULTS.overall_results) {{
    if (el) el.innerHTML = '';
    return;
  }}
  el.innerHTML = buildOverallResultsTableHtml(RESULTS.overall_results);
}}

function resultsSelectedLoadcaseLabel() {{
  return resultsSelectedLoadcase === 'worst' ? 'Worst load case' : resultsSelectedLoadcase;
}}

function getSelectedLoadcaseOverallResults() {{
  if (!RESULTS) return null;
  if (resultsSelectedLoadcase === 'worst') {{
    return RESULTS.overall_results;
  }}
  var byLc = RESULTS.overall_results_by_loadcase;
  return byLc && byLc[resultsSelectedLoadcase] ? byLc[resultsSelectedLoadcase] : null;
}}

function buildResultsSelectedLoadcaseSummary() {{
  var titleEl = document.getElementById('results-selected-lc-title');
  var el = document.getElementById('results-selected-lc-summary');
  if (!RESULTS) {{
    if (titleEl) titleEl.textContent = 'Selected loadcase results';
    if (el) el.innerHTML = '';
    return;
  }}
  if (titleEl) {{
    titleEl.textContent = 'Selected loadcase results — ' + resultsSelectedLoadcaseLabel();
  }}
  if (!el) return;
  var html = buildOverallResultsTableHtml(getSelectedLoadcaseOverallResults());
  el.innerHTML = html;
}}

function buildResultsCrossSectionSummary() {{
  var el = document.getElementById('results-xsec-summary');
  var titleEl = document.getElementById('results-xsec-title');
  if (!el || !RESULTS || !RESULTS.bays || !RESULTS.bays.length) {{
    if (el) el.innerHTML = '';
    if (titleEl) titleEl.textContent = 'Cross section results';
    return;
  }}
  var bay = getResultsBayData(resultsSelectedBay);
  if (!bay || !bay.cross_section_summary) {{
    el.innerHTML = '';
    if (titleEl) titleEl.textContent = 'Cross section results';
    return;
  }}
  var lcSuffix = resultsSelectedLoadcase === 'worst'
    ? ' (worst load case)'
    : (' — ' + resultsSelectedLoadcase);
  if (titleEl) titleEl.textContent = 'Cross section results — Bay ' + bay.bay_id + lcSuffix;
  var s = bay.cross_section_summary;
  var html = '<div class="results-min-ms-line">Min MS in Bay ' + bay.bay_id + ": "
    + '<span class="results-cs-ms ' + msClass(s.min_ms) + '">'
    + formatMsVal(s.min_ms, 2) + '</span></div>';

  var colgroup = '<colgroup>'
    + '<col class="results-cs-col-ms"><col class="results-cs-col-loc">'
    + '<col class="results-cs-col-lc"><col class="results-cs-col-mode"></colgroup>';
  var rows = [
    ['Min skin MS', s.skin],
    ['Min stringer MS', s.stringer],
    ['Min spar cap MS', s.spar_cap],
    ['Min spar web MS', s.spar_web]
  ];
  html += '<table class="results-cs-table">' + colgroup + '<tbody>';
  rows.forEach(function(row) {{
    var msLabel = row[0], item = row[1];
    if (!item) return;
    html += '<tr class="results-cs-h"><th>' + escapeHtml(msLabel) + '</th>'
      + '<th>Location</th><th>Loadcase</th><th>Failure mode</th></tr>'
      + '<tr class="results-cs-d"><td><span class="results-cs-ms ' + msClass(item.ms) + '">'
      + formatMsVal(item.ms, 2) + '</span></td>'
      + '<td>' + escapeHtml(item.location) + '</td>'
      + '<td>' + escapeHtml(item.lc) + '</td>'
      + '<td>' + escapeHtml(item.failure_mode) + '</td></tr>';
  }});
  html += '</tbody></table>';
  el.innerHTML = html;
}}

function interpSemispanGeom(ss, ws) {{
  var bays = ss.bays;
  if (!bays || !bays.length) return {{ fwd: 0, aft: 0 }};
  if (ws <= bays[0].ws) return {{ fwd: bays[0].fwd, aft: bays[0].aft }};
  var last = bays[bays.length - 1];
  if (ws >= last.ws) return {{ fwd: last.fwd, aft: last.aft }};
  for (var j = 0; j < bays.length - 1; j++) {{
    var b0 = bays[j], b1 = bays[j + 1];
    if (ws >= b0.ws && ws <= b1.ws) {{
      var denom = b1.ws - b0.ws;
      var t = denom > 1e-9 ? (ws - b0.ws) / denom : 0;
      return {{
        fwd: b0.fwd + t * (b1.fwd - b0.fwd),
        aft: b0.aft + t * (b1.aft - b0.aft)
      }};
    }}
  }}
  return {{ fwd: bays[0].fwd, aft: bays[0].aft }};
}}

function buildBayMsAnnotationText(b, ms) {{
  ms = ms || b.ms;
  if (!ms) return '<b>Bay' + b.bay_id + ' MS</b>';
  return '<b>Bay' + b.bay_id + ' MS</b><br>'
    + 'Skn ' + formatMsVal(ms.skins, 2) + '<br>'
    + 'Stg ' + formatMsVal(ms.stringers, 2) + '<br>'
    + 'Spr ' + formatMsVal(ms.spars, 2);
}}

function isResultsTipBayIndex(i) {{
  // [OAS] Always false. WingCalc's outermost RESULTS.bays entry is the tip RIB
  // closing the last real bay, not a bay of its own -- its bay_edges span is
  // fabricated from the previous bay, so its MS callout would land on top of its
  // neighbour's and is suppressed. Every OAS entry is an FEM element between two
  // real mesh nodes, with its own edges, so none of them is a closing station and
  // suppressing the last one would hide a real result.
  return false;
}}

// [OAS] Minimum spacing between two MS callouts on the overview, as a fraction of
// semi-span. WingCalc draws 19 bays across the span and every box fits; OAS draws one
// per FEM element -- 27 of them, cosine-clustered, so the outboard ones are 4 in wide
// against 39 in at the root -- and a box per element outboard is a solid block of
// overlapping text. Callouts are therefore thinned to roughly WingCalc's own spacing,
// walking outboard and keeping a bay only once it clears the last one kept. The
// SELECTED bay is always kept, so nothing the reader has actually asked for is hidden,
// and every bay is still in the dropdown and in the tables. 0 disables the thinning.
var OVERVIEW_ANNO_MIN_GAP = 0.055;

function buildResultsOverviewAnnotations(selectedIndex) {{
  var ss = PLANFORM.semispan;
  if (!ss || !RESULTS || !RESULTS.bays) return [];
  var annotations = [];
  var span = ss.ws.length ? Math.max.apply(null, ss.ws) : 0;
  var minGap = OVERVIEW_ANNO_MIN_GAP * span;
  var lastWs = -Infinity;
  ss.bays.forEach(function(bp, i) {{
    if (isResultsTipBayIndex(i)) return;
    var edge = ss.bay_edges[i];
    var mid = (edge.ws_left + edge.ws_right) / 2;
    if (i !== selectedIndex && mid - lastWs < minGap) return;
    lastWs = mid;
    var b = RESULTS.bays[i];
    var wsMid = (edge.ws_left + edge.ws_right) / 2;
    var wsSpan = edge.ws_right - edge.ws_left;
    var geomMid = interpSemispanGeom(ss, wsMid);
    var bayY = (geomMid.fwd + geomMid.aft) / 2;
    // Inside wingbox: keep text compact so it fits each bay.
    var fontSize = Math.max(9, Math.min(11, 9 + wsSpan / 40));
    var selected = i === selectedIndex;
    annotations.push({{
      x: wsMid,
      y: bayY,
      xref: 'x',
      yref: 'y',
      text: buildBayMsAnnotationText(b, getResultsBayMs(i)),
      showarrow: false,
      align: 'center',
      xanchor: 'center',
      yanchor: 'middle',
      font: {{ size: fontSize, family: 'Consolas, monospace', color: '#2c3e50' }},
      bgcolor: selected ? 'rgba(232, 245, 255, 0.95)' : 'rgba(255, 255, 255, 0.92)',
      bordercolor: selected ? '#2c3e50' : '#bdc3c7',
      borderwidth: selected ? 2 : 1,
      borderpad: 2
    }});
  }});
  return annotations;
}}

function updateResultsOverviewSelection(bayIndex) {{
  var el = document.getElementById('results-overview-plot');
  if (!el || !el.layout) return;
  Plotly.relayout(el, {{ annotations: buildResultsOverviewAnnotations(bayIndex) }});
  var ssBays = PLANFORM.semispan && PLANFORM.semispan.bays;
  if (!ssBays || !ssBays.length) return;
  var fillColors = ssBays.map(function(_bp, i) {{
    return i === bayIndex
      ? 'rgba(44, 62, 80, 0.18)'
      : 'rgba(44, 62, 80, 0.06)';
  }});
  var bayTraceStart = 3 + ssBays.length;
  var traceIdx = [];
  for (var t = 0; t < ssBays.length; t++) traceIdx.push(bayTraceStart + t);
  Plotly.restyle(el, {{ fillcolor: fillColors }}, traceIdx);
}}

function buildResultsBaySelect() {{
  var sel = document.getElementById('results-bay-select');
  if (!sel || !RESULTS) return;
  sel.innerHTML = '';
  RESULTS.bays.forEach(function(b, i) {{
    var opt = document.createElement('option');
    opt.value = i;
    opt.textContent = 'Bay ' + b.bay_id + ", WS = " + Number(b.ws).toFixed(1) + "''";
    sel.appendChild(opt);
  }});
  sel.value = resultsSelectedBay;
}}

function onResultsBayChange() {{
  var sel = document.getElementById('results-bay-select');
  if (!sel) return;
  selectResultsBay(parseInt(sel.value));
}}

function buildResultsLoadcaseSelect() {{
  var sel = document.getElementById('results-lc-select');
  if (!sel || !RESULTS) return;
  sel.innerHTML = '';
  var worstOpt = document.createElement('option');
  worstOpt.value = 'worst';
  worstOpt.textContent = 'Worst load case';
  sel.appendChild(worstOpt);
  (RESULTS.load_cases || []).forEach(function(lc) {{
    var opt = document.createElement('option');
    opt.value = lc;
    opt.textContent = lc;
    sel.appendChild(opt);
  }});
  sel.value = resultsSelectedLoadcase;
}}

function onResultsLoadcaseChange() {{
  var sel = document.getElementById('results-lc-select');
  if (!sel) return;
  resultsSelectedLoadcase = sel.value;
  buildResultsSelectedLoadcaseSummary();
  buildResultsCrossSectionSummary();
  updateResultsOverviewSelection(resultsSelectedBay);
  drawResultsXsecPlot(resultsSelectedBay);
  resultsPinnedDetail = null;
  restoreResultsPinnedDetail();
}}

function selectResultsBay(bayIndex) {{
  resultsSelectedBay = bayIndex;
  // Keep the dropdown in sync; the overview plot also selects bays by click.
  var sel = document.getElementById('results-bay-select');
  if (sel) sel.value = bayIndex;
  buildResultsCrossSectionSummary();
  updateResultsOverviewSelection(bayIndex);
  drawResultsXsecPlot(bayIndex);
  resultsPinnedDetail = null;
  restoreResultsPinnedDetail();
}}

function initResultsOverview() {{
  var ss = PLANFORM.semispan;
  if (!ss || !ss.ws.length) return;

  var traces = [];

  // Wingbox planform fill (fwd spar → aft spar loop)
  var boxX = ss.ws.concat(ss.ws.slice().reverse());
  var boxY = ss.fwd.concat(ss.aft.slice().reverse());
  traces.push({{
    x: boxX, y: boxY, mode: 'lines', name: 'Wingbox',
    fill: 'toself', fillcolor: 'rgba(44, 62, 80, 0.08)',
    line: {{ color: '#2c3e50', width: 1.5 }},
    hoverinfo: 'skip', showlegend: false
  }});

  traces.push({{
    x: ss.ws, y: ss.fwd, mode: 'lines', name: 'Fwd spar',
    line: {{ color: '#d62728', width: 2 }}, hoverinfo: 'skip', showlegend: false
  }});
  traces.push({{
    x: ss.ws, y: ss.aft, mode: 'lines', name: 'Aft spar',
    line: {{ color: '#2ca02c', width: 2 }}, hoverinfo: 'skip', showlegend: false
  }});

  // Rib locations (vertical at each bay WS, fwd to aft spar)
  // [OAS] ss.ribs when the payload carries it. WingCalc puts a bay AT each rib
  // station, so ss.bays doubles as the rib list; an OAS bay is the element BETWEEN
  // two mesh nodes and its ws is the element mid-station, so drawing the rib lines
  // off ss.bays would put every one of them mid-bay.
  (ss.ribs || ss.bays).forEach(function(bp) {{
    traces.push({{
      x: [bp.ws, bp.ws],
      y: [bp.fwd, bp.aft],
      mode: 'lines',
      line: {{ color: '#555', width: 1 }},
      showlegend: false, hoverinfo: 'skip'
    }});
  }});

  var clickX = [], clickY = [], clickCd = [];
  ss.bays.forEach(function(bp, i) {{
    var edge = ss.bay_edges[i];
    var wsMid = (edge.ws_left + edge.ws_right) / 2;
    var gL = interpSemispanGeom(ss, edge.ws_left);
    var gR = interpSemispanGeom(ss, edge.ws_right);
    var gM = interpSemispanGeom(ss, wsMid);
    clickX.push(wsMid);
    clickY.push((gM.fwd + gM.aft) / 2);
    clickCd.push(i);
    traces.push({{
      x: [edge.ws_left, edge.ws_right, edge.ws_right, edge.ws_left, edge.ws_left],
      y: [gL.fwd, gR.fwd, gR.aft, gL.aft, gL.fwd],
      mode: 'lines', fill: 'toself',
      fillcolor: 'rgba(44,62,80,0.06)',
      line: {{ width: 0 }}, showlegend: false, hoverinfo: 'skip'
    }});
  }});

  traces.push({{
    x: clickX, y: clickY, mode: 'markers',
    marker: {{ size: 28, opacity: 0.01, color: '#2c3e50' }},
    showlegend: false,
    customdata: clickCd,
    text: clickCd.map(function(i) {{
      return 'Bay ' + RESULTS.bays[i].bay_id + ' (click to select)';
    }}),
    hovertemplate: '%{{text}}<extra></extra>'
  }});

  // Nacelle reference lines (semispan: positive WS only)
  var shapes = [];
  [
    {{ ws: PLANFORM.nacelle_ib_y, color: '#ff7f0e' }},
    {{ ws: PLANFORM.nacelle_ob_y, color: '#9467bd' }},
  ].forEach(function(spec) {{
    if (spec.ws == null || spec.ws <= 0) return;
    shapes.push({{
      type: 'line',
      x0: spec.ws, x1: spec.ws,
      y0: 0, y1: 1, yref: 'paper',
      line: {{ color: spec.color, width: 1.5, dash: 'dot' }}
    }});
  }});

  var layout = {{
    xaxis: {{ title: 'Spanwise WS (in)' }},
    yaxis: {{
      autorange: 'reversed',
      scaleanchor: 'x',
      scaleratio: 1,
      showticklabels: false,
      showline: false,
      ticks: '',
      title: {{ text: '' }},
      showgrid: false
    }},
    margin: {{ t: 20, b: 36, l: 12, r: 12 }},
    shapes: shapes,
    annotations: buildResultsOverviewAnnotations(0),
    showlegend: false,
    hovermode: 'closest'
  }};

  var el = document.getElementById('results-overview-plot');
  Plotly.newPlot(el, traces, layout, {{ responsive: true }}).then(function() {{
    if (!window.__resultsOverviewClickBound) {{
      el.on('plotly_click', function(ev) {{
        if (!ev.points || !ev.points.length) return;
        var pt = ev.points[0];
        if (pt.customdata !== undefined && pt.customdata !== null) {{
          selectResultsBay(pt.customdata);
        }}
      }});
      window.__resultsOverviewClickBound = true;
    }}
    selectResultsBay(resultsSelectedBay);
  }});
}}

function pushResultsSparGeom(traces, bay, bayIndex, sp, label, spec) {{
  if (!sp || !sp.draw) return;
  var d = sp.draw;
  function pushPoly(poly, lineColor, fillRgba) {{
    if (!poly || !poly.x || poly.x.length < 3) return;
    traces.push({{
      x: poly.x, y: poly.z,
      mode: 'lines', line: {{ color: lineColor, width: 1.5 }},
      fill: 'toself', fillcolor: fillRgba,
      showlegend: false, hoverinfo: 'skip'
    }});
  }}
  pushPoly(d.web, spec.webLine, spec.webFill);
  if (d.upper_skin_leg) pushPoly(d.upper_skin_leg, spec.capLine, spec.upperSkinFill);
  pushPoly(d.upper_web_leg, spec.capLine, spec.upperWebFill);
  if (d.lower_skin_leg) pushPoly(d.lower_skin_leg, spec.capLine, spec.lowerSkinFill);
  pushPoly(d.lower_web_leg, spec.capLine, spec.lowerWebFill);

  var wm = sp.web, uc = sp.upper_cap, lc = sp.lower_cap;
  var hoverX = [], hoverY = [], hoverText = [], hoverCd = [];

  function addCapHover(poly, capKey) {{
    if (!poly || !poly.x || poly.x.length < 3) return;
    var n = poly.x.length - 1, sx = 0, sz = 0;
    for (var i = 0; i < n; i++) {{ sx += poly.x[i]; sz += poly.z[i]; }}
    var rec = getResultsRecord(bayIndex, 'spar_cap', label + '|' + capKey);
    hoverX.push(sx / n); hoverY.push(sz / n);
    hoverText.push(msRecordHoverText(rec, label + ' spar, ' + capKey + ' cap', 'spar_cap'));
    hoverCd.push(['spar_cap', bayIndex, label + '|' + capKey]);
  }}
  addCapHover(d.upper_skin_leg, 'upper');
  addCapHover(d.lower_skin_leg, 'lower');
  if (d.web && d.web.x.length >= 3) {{
    var n = d.web.x.length - 1, sx = 0, sz = 0;
    for (var j = 0; j < n; j++) {{ sx += d.web.x[j]; sz += d.web.z[j]; }}
    var recW = getResultsRecord(bayIndex, 'spar_web', label);
    hoverX.push(sx / n); hoverY.push(sz / n);
    hoverText.push(msRecordHoverText(recW, label + ' spar web', 'spar_web'));
    hoverCd.push(['spar_web', bayIndex, label]);
  }}
  if (hoverX.length) {{
    traces.push({{
      x: hoverX, y: hoverY, mode: 'markers', name: label + ' MS',
      marker: {{ size: 12, opacity: 0.01, color: spec.webLine }},
      showlegend: false, text: hoverText, customdata: hoverCd,
      hovertemplate: '%{{text}}<extra></extra>'
    }});
  }}
}}

// Wingbox view framing for a bay: everything drawn in it (skin panels and their
// markers, stringers) padded a little. The fixed-scale option feeds the first bay's
// ranges to every bay so sections stay comparable in size.
function resultsXsecRange(bayIndex) {{
  var bay = BAYS[bayIndex];
  var geom = createXsecSurfaceHelpers(bay);
  var zMin = Infinity, zMax = -Infinity;
  function trackZ(z) {{
    if (z < zMin) zMin = z;
    if (z > zMax) zMax = z;
  }}
  (bay.upper_skin_panels || []).forEach(function(p) {{
    geom.skinPanelPolygon(p, true).z.forEach(trackZ);
    trackZ(p.z);
  }});
  (bay.lower_skin_panels || []).forEach(function(p) {{
    geom.skinPanelPolygon(p, false).z.forEach(trackZ);
    trackZ(p.z);
  }});
  (bay.upper_stringers || []).forEach(function(s) {{
    var g = geom.stringerPolygons(s, true);
    g.flange.z.forEach(trackZ); g.web.z.forEach(trackZ);
  }});
  (bay.lower_stringers || []).forEach(function(s) {{
    var g = geom.stringerPolygons(s, false);
    g.flange.z.forEach(trackZ); g.web.z.forEach(trackZ);
  }});

  var padX = 0.05 * Math.max(bay.box_chord, 1);
  var padZ = 0.08 * Math.max(zMax - zMin, 1);
  if (!isFinite(zMin)) {{ zMin = -5; zMax = 5; }}
  return {{
    x: [geom.xFwd - padX, geom.xAft + padX],
    y: [zMin - padZ, zMax + padZ]
  }};
}}

function onResultsScaleChange() {{
  drawResultsXsecPlot(resultsSelectedBay);
}}

function drawResultsXsecPlot(bayIndex) {{
  var bay = BAYS[bayIndex];
  var geom = createXsecSurfaceHelpers(bay);
  var chord = geom.chord, xFwd = geom.xFwd, xAft = geom.xAft;
  var traces = [];

  var upSk = bay.upper_skin_panels || [];
  upSk.forEach(function(p) {{
    var poly = geom.skinPanelPolygon(p, true);
    traces.push({{
      x: poly.x, y: poly.z, mode: 'lines',
      line: {{ color: '#1f77b4', width: 0.5 }},
      fill: 'toself', fillcolor: 'rgba(31,119,180,0.25)',
      showlegend: false, hoverinfo: 'skip'
    }});
  }});
  var loSk = bay.lower_skin_panels || [];
  loSk.forEach(function(p) {{
    var poly = geom.skinPanelPolygon(p, false);
    traces.push({{
      x: poly.x, y: poly.z, mode: 'lines',
      line: {{ color: '#1f77b4', width: 0.5 }},
      fill: 'toself', fillcolor: 'rgba(31,119,180,0.25)',
      showlegend: false, hoverinfo: 'skip'
    }});
  }});

  if (upSk.length) {{
    traces.push({{
      x: upSk.map(function(p) {{ return p.x; }}),
      y: upSk.map(function(p) {{ return p.z; }}),
      mode: 'markers', showlegend: false,
      marker: {{ size: 10, opacity: 0.01, color: '#d62728' }},
      text: upSk.map(function(p) {{
        return msRecordHoverText(getResultsRecord(bayIndex, 'skin', p.name), p.name, 'skin');
      }}),
      customdata: upSk.map(function(p) {{ return ['skin', bayIndex, p.name]; }}),
      hovertemplate: '%{{text}}<extra></extra>'
    }});
  }}
  if (loSk.length) {{
    traces.push({{
      x: loSk.map(function(p) {{ return p.x; }}),
      y: loSk.map(function(p) {{ return p.z; }}),
      mode: 'markers', showlegend: false,
      marker: {{ size: 10, opacity: 0.01, color: '#1f77b4' }},
      text: loSk.map(function(p) {{
        return msRecordHoverText(getResultsRecord(bayIndex, 'skin', p.name), p.name, 'skin');
      }}),
      customdata: loSk.map(function(p) {{ return ['skin', bayIndex, p.name]; }}),
      hovertemplate: '%{{text}}<extra></extra>'
    }});
  }}

  pushResultsSparGeom(traces, bay, bayIndex, bay.fwd_spar, 'Fwd', {{
    webLine: '#9e2a2b', webFill: 'rgba(214,39,40,0.55)',
    capLine: '#6d1f20',
    upperSkinFill: 'rgba(214,39,40,0.45)', upperWebFill: 'rgba(214,39,40,0.55)',
    lowerSkinFill: 'rgba(214,39,40,0.45)', lowerWebFill: 'rgba(214,39,40,0.55)',
  }});
  pushResultsSparGeom(traces, bay, bayIndex, bay.aft_spar, 'Aft', {{
    webLine: '#1f6b3d', webFill: 'rgba(39,174,96,0.55)',
    capLine: '#14532a',
    upperSkinFill: 'rgba(39,174,96,0.45)', upperWebFill: 'rgba(39,174,96,0.55)',
    lowerSkinFill: 'rgba(39,174,96,0.45)', lowerWebFill: 'rgba(39,174,96,0.55)',
  }});

  (bay.upper_stringers || []).forEach(function(s) {{
    var g = geom.stringerPolygons(s, true);
    traces.push({{
      x: g.flange.x, y: g.flange.z, mode: 'lines',
      line: {{ color: '#a51c30', width: 1 }}, fill: 'toself',
      fillcolor: 'rgba(165,28,48,0.45)', showlegend: false, hoverinfo: 'skip'
    }});
    traces.push({{
      x: g.web.x, y: g.web.z, mode: 'lines',
      line: {{ color: '#a51c30', width: 1 }}, fill: 'toself',
      fillcolor: 'rgba(165,28,48,0.55)', showlegend: false, hoverinfo: 'skip'
    }});
  }});
  var upStg = bay.upper_stringers || [];
  if (upStg.length) {{
    traces.push({{
      x: upStg.map(function(s) {{ return s.x; }}),
      y: upStg.map(function(s) {{ return s.z; }}),
      mode: 'markers', showlegend: false,
      marker: {{ size: 8, opacity: 0.01, color: '#d62728' }},
      text: upStg.map(function(s) {{
        return msRecordHoverText(getResultsRecord(bayIndex, 'stringer', s.name), s.name, 'stringer');
      }}),
      customdata: upStg.map(function(s) {{ return ['stringer', bayIndex, s.name]; }}),
      hovertemplate: '%{{text}}<extra></extra>'
    }});
  }}
  (bay.lower_stringers || []).forEach(function(s) {{
    var g = geom.stringerPolygons(s, false);
    traces.push({{
      x: g.flange.x, y: g.flange.z, mode: 'lines',
      line: {{ color: '#1565a8', width: 1 }}, fill: 'toself',
      fillcolor: 'rgba(21,101,168,0.45)', showlegend: false, hoverinfo: 'skip'
    }});
    traces.push({{
      x: g.web.x, y: g.web.z, mode: 'lines',
      line: {{ color: '#1565a8', width: 1 }}, fill: 'toself',
      fillcolor: 'rgba(21,101,168,0.55)', showlegend: false, hoverinfo: 'skip'
    }});
  }});
  var loStg = bay.lower_stringers || [];
  if (loStg.length) {{
    traces.push({{
      x: loStg.map(function(s) {{ return s.x; }}),
      y: loStg.map(function(s) {{ return s.z; }}),
      mode: 'markers', showlegend: false,
      marker: {{ size: 8, opacity: 0.01, color: '#1f77b4' }},
      text: loStg.map(function(s) {{
        return msRecordHoverText(getResultsRecord(bayIndex, 'stringer', s.name), s.name, 'stringer');
      }}),
      customdata: loStg.map(function(s) {{ return ['stringer', bayIndex, s.name]; }}),
      hovertemplate: '%{{text}}<extra></extra>'
    }});
  }}

  var bayRange = resultsXsecRange(bayIndex);
  var scale = document.getElementById('results-scale-select');
  var axisRange = (scale && scale.value === 'fixed' && bayIndex !== 0)
    ? resultsXsecRange(0) : bayRange;
  // Padding is symmetric, so the range midpoint is the unpadded section midpoint.
  var zMid = (bay.ZCG != null && isFinite(bay.ZCG))
    ? bay.ZCG : (bayRange.y[0] + bayRange.y[1]) / 2;

  var layout = {{
    xaxis: {{
      title: 'Y (in)',
      range: axisRange.x.slice()
    }},
    yaxis: {{
      title: 'Z (in)',
      range: axisRange.y.slice(),
      scaleanchor: 'x', scaleratio: 1
    }},
    // [AI mod] No chart title: it read 'Bay 1 | WS = 0.0 in', which is what the Bay
    // selector directly above the plot already says. Dropping it lets the top margin go
    // from the 50 px the title needed down to the few px the top tick label needs, and
    // this plot is height bound -- see the note on .results-zone1.
    margin: {{ t: 12, b: 50, l: 60, r: 20 }},
    showlegend: false,
    hovermode: 'closest',
    annotations: buildResultsXsecAnnotations(bayIndex, xFwd, xAft, zMid)
  }};

  var el = document.getElementById('results-xsec-plot');
  var cfg = {{ responsive: true }};
  // [AI mod] The callouts can only be spread once the plot is drawn and the pixel
  // geometry exists, so it hangs off the draw promise rather than the layout above.
  var drawn = (!el.data || el.data.length === 0)
    ? Plotly.newPlot(el, traces, layout, cfg).then(bindResultsXsecPlotHandlers)
    : Plotly.react(el, traces, layout, cfg);
  drawn.then(function() {{ spreadResultsXsecAnnotations(el); }});
}}

function bindResultsXsecPlotHandlers() {{
  if (window.__resultsXsecPlotBound) return;
  var el = document.getElementById('results-xsec-plot');
  if (!el) return;
  el.on('plotly_click', function(ev) {{
    if (!ev.points || !ev.points.length) return;
    var cd = ev.points[0].customdata;
    if (cd && cd.length) showResultsDetail(cd, true);
  }});
  el.on('plotly_hover', function(ev) {{
    if (!ev.points || !ev.points.length) return;
    var cd = ev.points[0].customdata;
    if (cd && cd.length) showResultsDetail(cd, false);
  }});
  el.on('plotly_unhover', function() {{
    restoreResultsPinnedDetail();
  }});
  // [AI] Where the section leaves no room to spread them, callouts still land on top of
  // one another. Clicking one brings it to the front: Plotly draws annotations in array
  // order, so the clicked box moves to the end of the array and goes fully opaque while
  // the rest go back to translucent.
  el.on('plotly_clickannotation', function(ev) {{
    var i = ev.index;
    var anns = (el.layout.annotations || []).map(function(a) {{
      return Object.assign({{}}, a, {{ bgcolor: 'rgba(255, 255, 255, 0.92)' }});
    }});
    if (i == null || i < 0 || i >= anns.length) return;
    var front = anns.splice(i, 1)[0];
    front.bgcolor = 'rgb(255, 255, 255)';
    anns.push(front);
    Plotly.relayout(el, {{ annotations: anns }});
  }});
  window.__resultsXsecPlotBound = true;
}}

// ---- Optimization ----
// [OAS] Dropped in full. WingCalc's trace is per-bay DE generations read from
// optimization_details.csv; OAS's is an SLSQP design-vector history with no bay
// axis, so there is nothing to feed these functions and nothing here is reachable.

// ---- Weight ----
function initNegativeMarginWarning() {{
  var el = document.getElementById('negative-margin-warning');
  if (!el) return;
  var overall = RESULTS && RESULTS.overall_results;
  var minMs = overall ? Number(overall.min_ms) : NaN;
  if (!isFinite(minMs) || minMs >= 0) {{
    el.style.display = 'none';
    el.innerHTML = '';
    return;
  }}
  var detail = [];
  if (overall.bay_id !== undefined && overall.bay_id !== null) detail.push('Bay ' + escapeHtml(overall.bay_id));
  if (overall.lc) detail.push('LC ' + escapeHtml(overall.lc));
  if (overall.location) detail.push(escapeHtml(overall.location));
  if (overall.failure_mode) detail.push(escapeHtml(overall.failure_mode));
  el.innerHTML = 'WARNING: Negative margin of safety detected.'
    + '<span class="warning-detail">Minimum MS '
    + escapeHtml(formatNumber(minMs, 3))
    + (detail.length ? ' (' + detail.join(', ') + ')' : '')
    + '</span>';
  el.style.display = 'block';
}}

function weightValueHtml(v, decimals, units) {{
  return escapeHtml(formatNumber(v, decimals)) + (units ? ' ' + escapeHtml(units) : '');
}}

function renderWeightSummaryCards() {{
  var el = document.getElementById('weight-summary-cards');
  if (!el || !WEIGHT || !WEIGHT.summary) return;
  var s = WEIGHT.summary;
  // [AI mod] The three parts and their total, so the row adds up as it is read:
  // W_wing = W_wingbox + W_secondary + misc, misc being what k_misc puts on the
  // secondary structure. The factor itself is on the breakdown table beside this.
  var cards = [
    ['Total wing weight', s.w_wing],
    ['Wingbox', s.w_wingbox],
    ['Secondary', s.w_secondary],
    ['Misc weight', s.w_misc]
  ];
  el.innerHTML = cards.map(function(card) {{
    return '<div class="weight-card">'
      + '<div class="weight-card-label">' + escapeHtml(card[0]) + '</div>'
      + '<div class="weight-card-value">' + weightValueHtml(card[1], 0, 'lbs') + '</div>'
      + '</div>';
  }}).join('');
}}

function renderWeightBreakdownTable() {{
  var el = document.getElementById('weight-breakdown-table');
  if (!el || !WEIGHT) return;
  var html = '<thead><tr><th>Group</th><th>Component</th><th>Value (lbs)</th>'
    + '<th>Reference</th></tr></thead><tbody>';
  function rows(group, items) {{
    (items || []).forEach(function(item) {{
      var cls = item.is_total ? ' class="weight-table-row-total"'
        : (item.is_subtotal ? ' class="weight-table-row-subtotal"' : '');
      // [AI] Whole lb for a weight; only the dimensionless factor asks for decimals.
      var decimals = item.decimals === undefined ? 0 : item.decimals;
      html += '<tr' + cls + '>'
        + '<td>' + escapeHtml(group) + '</td>'
        + '<td>' + escapeHtml(item.label) + '</td>'
        + '<td>' + escapeHtml(formatNumber(item.value, decimals)) + '</td>'
        + '<td>' + escapeHtml(item.ref || '') + '</td>'
        + '</tr>';
    }});
  }}
  rows('Wingbox', WEIGHT.wingbox);
  rows('Secondary', WEIGHT.secondary);
  rows('Final', WEIGHT.totals);
  html += '</tbody>';
  el.innerHTML = html;
}}

// [AI] ---- Weight Summary: spanwise distribution ----
// The chart and the table under it are two views of one payload, WEIGHT.spanwise, and
// are always drawn together through updateWeightPlot() so they cannot disagree.
//
// Two selectors drive them. "Scope" picks which categories are in: the wingbox alone,
// or the wingbox plus everything the wing carries on top of it. "Display" picks lb per
// station or lb per inch of span. Every weight in the payload is already factored, so
// the wingbox scope is the *full* wingbox -- not the basic structure. The non-optimum
// items are inside the structure categories; the nacelle and landing gear
// reinforcements have a category of their own.
//
// All of it is the right semispan, station by station, which is the grouping the
// per-bay weight breakdown is built on.

function weightPlotScope() {{
  var el = document.getElementById('weight-plot-scope');
  return (el && el.value === 'wingbox') ? 'wingbox' : 'wing';
}}

function weightPlotLinear() {{
  var el = document.getElementById('weight-plot-mode');
  return !!el && el.value === 'linear';
}}

// [AI] The categories the active scope shows, in stacking order. The wingbox ones are
// in both scopes; the secondary ones only in the whole-wing scope.
function weightPlotCategories() {{
  var all = (WEIGHT.spanwise && WEIGHT.spanwise.categories) || [];
  if (weightPlotScope() === 'wingbox') {{
    return all.filter(function(c) {{ return c.scope === 'wingbox'; }});
  }}
  return all;
}}

function weightStationTotal(st) {{
  return weightPlotScope() === 'wingbox' ? Number(st.w_wingbox) : Number(st.w_wing);
}}

// [AI] A weight per inch of span divides by the bay the station carries. The tip
// station has no bay, so it has no linear weight at all and comes back NaN, which
// Plotly and the table both leave blank rather than plotting a discrete weight as if
// it were spread over something.
function weightPlotValue(st, value) {{
  if (!weightPlotLinear()) return value;
  var pitch = Number(st.rib_pitch);
  return pitch > 0 ? value / pitch : NaN;
}}

function renderWeightStationTable() {{
  var el = document.getElementById('weight-station-table');
  if (!el || !WEIGHT || !WEIGHT.spanwise) return;
  var rows = WEIGHT.spanwise.stations || [];
  var cats = weightPlotCategories();
  var linear = weightPlotLinear();
  var decimals = linear ? 3 : 0;

  // [AI] The weight columns change unit with the display selector, so the units row is
  // built with the header rather than written into the labels.
  var wUnits = linear ? 'lbs/in' : 'lbs';
  var html = '<thead><tr><th>Bay</th><th>WS</th><th>Rib type</th>';
  cats.forEach(function(c) {{
    html += '<th>' + escapeHtml(c.short || c.label) + '</th>';
  }});
  html += '<th>Total half span</th></tr>'
    + '<tr class="weight-station-units"><td></td><td>in</td><td></td>'
    + cats.map(function() {{ return '<td>' + wUnits + '</td>'; }}).join('')
    + '<td>' + wUnits + '</td></tr></thead><tbody>';

  rows.forEach(function(st) {{
    html += '<tr>'
      + '<td>' + escapeHtml(st.bay_id) + '</td>'
      + '<td>' + escapeHtml(formatNumber(st.ws, 2)) + '</td>'
      + '<td>' + escapeHtml(st.rib_type_display) + '</td>';
    cats.forEach(function(c) {{
      html += '<td>' + weightStationCell(st, st.w[c.key], decimals) + '</td>';
    }});
    html += '<td>' + weightStationCell(st, weightStationTotal(st), decimals) + '</td>'
      + '</tr>';
  }});
  html += '</tbody>';
  el.innerHTML = html;
}}

// [AI mod] A zero is left blank rather than printed: most stations carry nothing at
// all under most categories -- no flap, no winglet, no reinforcement -- and a column of
// zeros hides the stations that do.
function weightStationCell(st, value, decimals) {{
  var shown = weightPlotValue(st, Number(value));
  if (!isFinite(shown) || shown === 0) return '';
  // [AI mod] A small negative -- a nacelle rib lighter than the plain rib under it --
  // rounds to "-0"; drop the sign.
  var text = formatNumber(shown, decimals);
  return escapeHtml(Number(text) === 0 ? text.replace('-', '') : text);
}}

function updateWeightPlot() {{
  drawWeightPlot();
  renderWeightStationTable();
  renderWeightPlotNote();
}}

// [AI] How the chart ties back to the whole-wing figures in the breakdown beside it.
function renderWeightPlotNote() {{
  var el = document.getElementById('weight-plot-note');
  if (!el || !WEIGHT || !WEIGHT.spanwise) return;
  var wingbox = weightPlotScope() === 'wingbox';
  var semispan = WEIGHT.spanwise.semispan_totals[wingbox ? 'wingbox' : 'wing'];
  var full = wingbox ? WEIGHT.summary.w_wingbox : WEIGHT.summary.w_wing;
  el.innerHTML = 'Halfspan total = ' + escapeHtml(formatNumber(semispan, 0))
    + ' lbs (full span = ' + escapeHtml(formatNumber(full, 0)) + ' lbs)'
    // [OAS] An optional sentence saying what the chart does NOT cover. There is no
    // room for it anywhere else on the panel and it belongs beside the total it
    // qualifies rather than in a tab of its own.
    + (WEIGHT.spanwise.note ? '. ' + escapeHtml(WEIGHT.spanwise.note) : '');
}}

function drawWeightPlot() {{
  var el = document.getElementById('weight-plot');
  if (!el || !WEIGHT) return;
  var rows = (WEIGHT.spanwise && WEIGHT.spanwise.stations) || [];
  if (!rows.length) {{
    el.innerHTML = '<p style="padding:32px;color:#888;">No station weight data available.</p>';
    return;
  }}
  var linear = weightPlotLinear();
  var yUnits = linear ? 'lbs/in' : 'lbs';
  var yTitle = linear ? 'Linear weight (lbs/in span)' : 'Weight (lbs)';
  var yHoverFormat = linear ? '.3f' : '.1f';
  var totalName = linear ? 'Station linear total' : 'Station total';

  // [AI] A bar is centred on the bay it covers. The tip station carries no bay, so it
  // sits on its own rib station.
  var ws = rows.map(function(st) {{
    var pitch = Number(st.rib_pitch);
    return Number(st.ws) + (pitch > 0 ? 0.5 * pitch : 0);
  }});
  var bayLabels = rows.map(function(st) {{ return 'Bay ' + st.bay_id; }});

  var traces = weightPlotCategories().map(function(cat, index) {{
    return {{
      x: ws,
      // [AI mod] A station carrying nothing under a category is left null rather than
      // plotted as zero, so the unified hover lists only what is actually there --
      // most stations have no flap, no reinforcement and no winglet.
      y: rows.map(function(st) {{
        var w = Number(st.w[cat.key]);
        return w === 0 ? null : weightPlotValue(st, w);
      }}),
      text: bayLabels,
      // [AI] The bay number is written into the skin segment only -- the first and
      // largest -- so each bar is labelled once. The tip station has no skin, so it is
      // named by its hover alone.
      textposition: index === 0 ? 'inside' : 'none',
      insidetextanchor: 'middle',
      type: 'bar',
      name: cat.label,
      marker: {{ color: cat.color }},
      hovertemplate: '<b>%{{text}}</b><br>WS=%{{x:.2f}} in<br>' + cat.label
        + '=%{{y:' + yHoverFormat + '}} ' + yUnits + '<extra></extra>'
    }};
  }});
  traces.push({{
    x: ws,
    y: rows.map(function(st) {{ return weightPlotValue(st, weightStationTotal(st)); }}),
    text: bayLabels,
    type: 'scatter',
    mode: 'lines+markers',
    name: totalName,
    line: {{ color: '#2c3e50', width: 2 }},
    marker: {{ size: 6 }},
    hovertemplate: '<b>%{{text}}</b><br>WS=%{{x:.2f}} in<br>' + totalName
      + '=%{{y:' + yHoverFormat + '}} ' + yUnits + '<extra></extra>'
  }});

  var layout = {{
    // [AI mod] 'relative' rather than 'stack' so a negative bar stacks downwards from
    // zero instead of being dropped: a nacelle rib can come out lighter than the plain
    // rib the model already carries at its station.
    barmode: 'relative',
    // [AI mod] The legend is a column in the right margin rather than a band under the
    // axis, so the right margin is what reserves the strip it stands in -- wide enough
    // for the longest name, 'Station linear total'. The whole-wing scope is the tall
    // case at 14 entries, about 280px at 11px; the 350px plot less the margins here
    // leaves 290px, which is what keeps Plotly from putting the legend in a scroll box.
    margin: {{ t: 12, b: 48, l: 70, r: 165 }},
    xaxis: {{ title: 'Wing Station (in)' }},
    yaxis: {{ title: yTitle }},
    legend: {{
      orientation: 'v',
      x: 1.01, xanchor: 'left',
      y: 1, yanchor: 'top',
      font: {{ size: 11 }}
    }},
    hovermode: 'x unified'
  }};
  Plotly.newPlot(el, traces, layout, {{ responsive: true }});
}}

function initWeight() {{
  var emptyEl = document.getElementById('weight-empty');
  var bodyEl = document.getElementById('weight-body');
  if (!WEIGHT || !WEIGHT.summary) {{
    if (emptyEl) emptyEl.style.display = '';
    if (bodyEl) bodyEl.style.display = 'none';
    return;
  }}
  if (emptyEl) emptyEl.style.display = 'none';
  if (bodyEl) bodyEl.style.display = 'block';
  renderWeightSummaryCards();
  renderWeightBreakdownTable();
  updateWeightPlot();
}}

// [AI] ---- Weight Visuals ----
// Where the wing's weight acts on the right wing: a top view (X against span) over
// the same planform the Planform tab draws, and a looking-aft view (Z against span)
// over the wingbox depth envelope. Only the station cgs are plotted -- clicking one
// lists the components that make it up in the side panel.
//
// A station is a bay or a rib, whichever grouping is picked in the toolbar. The two
// hold the same weight: grouped per bay a cg floats between two ribs, grouped per rib
// each rib takes the half bay either side of it so the cg lands at the rib. Every
// function below reads the active grouping through weightVisGroup(), so the two are
// one code path.

var WEIGHTVIS_KINDS = [
  {{ kind: 'Skin',      color: '#1a5f8a' }},
  {{ kind: 'Stringer',  color: '#2c7fb8' }},
  {{ kind: 'Spar cap',  color: '#e15759' }},
  {{ kind: 'Spar web',  color: '#76b7b2' }},
  {{ kind: 'Rib',       color: '#f28e2b' }}
];
var WEIGHTVIS_GROUPS = {{
  bay: {{
    name: 'Bay',
    trace: 'Bay cg',
    hover: 'cg (incl. rib)',
    note: 'Weight grouped into bays: a bay is the structure between two ribs, so its '
      + 'cg falls between them. The rib at the bay\\'s inboard station is counted in it.'
  }},
  rib: {{
    name: 'Rib',
    trace: 'Rib cg',
    hover: 'cg (half bay either side)',
    note: 'Weight grouped onto ribs: each rib carries the half bay either side of it '
      + 'plus its own plate, so the cg lands at the rib. The half bays are integrated '
      + 'over their own half of the span, so a tapering bay gives its heavy end to the '
      + 'right rib.'
  }}
}};
var weightVisSelectedStation = null;

// [AI] The grouping shown, and the payload behind it. The select is the only source of
// truth; a deck with no rib breakdown falls back to bays rather than blanking the tab.
function weightVisGroupMode() {{
  var el = document.getElementById('weightvis-group-mode');
  var mode = el ? el.value : 'bay';
  return (WEIGHTVIS.groupings && WEIGHTVIS.groupings[mode]) ? mode : 'bay';
}}

function weightVisGroup() {{
  return (WEIGHTVIS.groupings || {{}})[weightVisGroupMode()] || {{}};
}}

function weightVisGroupSpec() {{
  var spec = WEIGHTVIS_GROUPS[weightVisGroupMode()];
  // [OAS] The payload may override the wording. WingCalc's text explains that a bay
  // carries the rib at its inboard station, which is exactly wrong here -- OAS has no
  // ribs at all -- and the note sits right above the plots where it would be read as
  // a statement about what is in them.
  var over = WEIGHTVIS.group_spec_overrides && WEIGHTVIS.group_spec_overrides[weightVisGroupMode()];
  return over ? Object.assign({{}}, spec, over) : spec;
}}

function weightVisStations() {{
  return weightVisGroup().stations || [];
}}

// [AI] Chord fraction of a cg, blank where the payload has none (no local chord
// to divide by). Every x/c shown in this tab goes through here so the plots, the
// side panel and the readout agree on the format.
function weightVisXcText(xc) {{
  return (xc === null || xc === undefined) ? '—' : Number(xc).toFixed(3);
}}

function weightVisKindColor(kind) {{
  for (var i = 0; i < WEIGHTVIS_KINDS.length; i++) {{
    if (WEIGHTVIS_KINDS[i].kind === kind) return WEIGHTVIS_KINDS[i].color;
  }}
  return '#95a5a6';
}}

function weightVisStationMarkerSizes(stations, sizeByWeight) {{
  if (!sizeByWeight) return stations.map(function() {{ return 14; }});
  // [AI] Marker area proportional to station weight, so a heavy one reads as heavy.
  var wMax = 0;
  stations.forEach(function(s) {{ if (s.w > wMax) wMax = s.w; }});
  if (wMax <= 0) return stations.map(function() {{ return 14; }});
  return stations.map(function(s) {{
    return 10 + 18 * Math.sqrt(Math.max(s.w, 0) / wMax);
  }});
}}

function weightVisStationTrace(axis, sizeByWeight) {{
  // [AI] axis: 'x' for the top view (fuselage station), 'z' for the looking-aft view.
  var stations = weightVisStations();
  if (!stations.length) return null;
  var spec = weightVisGroupSpec();
  return {{
    x: stations.map(function(s) {{ return s.y; }}),
    y: stations.map(function(s) {{ return s[axis]; }}),
    mode: 'markers',
    type: 'scatter',
    name: spec.trace,
    marker: {{
      color: weightVisStationColors(stations),
      symbol: weightVisStationSymbols(stations),
      size: weightVisStationMarkerSizes(stations, sizeByWeight),
      line: {{ width: 2 }}
    }},
    customdata: stations.map(function(s) {{
      return [s.id, s.w, s.x, s.y, s.z, weightVisXcText(s.xc)];
    }}),
    hovertemplate:
      '<b>' + spec.name + ' %{{customdata[0]}} ' + spec.hover + '</b><br>'
      + 'W = %{{customdata[1]:.2f}} lbs<br>'
      + 'cg X = %{{customdata[2]:.2f}} in  (x/c = %{{customdata[5]}})<br>'
      + 'cg Y = %{{customdata[3]:.2f}} in<br>'
      + 'cg Z = %{{customdata[4]:.2f}} in'
      + '<br><i>click to list components</i><extra></extra>'
  }};
}}

// [AI] The selected station is drawn as a filled red diamond; the rest stay open
// outlines. Open symbols take their outline from marker.color, so it carries the
// colour in both states.
function weightVisStationColors(stations) {{
  return stations.map(function(s) {{
    return s.id === weightVisSelectedStation ? '#c0392b' : '#2c3e50';
  }});
}}

function weightVisStationSymbols(stations) {{
  return stations.map(function(s) {{
    return s.id === weightVisSelectedStation ? 'diamond' : 'diamond-open';
  }});
}}

function weightVisSeparateTrace(axis) {{
  // [AI] Wing weight that no station's total carries, drawn apart from the station
  // diamonds so it is visible as its own mass rather than folded into a station that
  // does not carry it. The flap and the aileron are resolved to the active grouping's
  // own stations, so these markers move with it.
  var items = weightVisGroup().separate || [];
  if (!items.length) return null;
  var station = weightVisGroupSpec().name.toLowerCase();
  return {{
    x: items.map(function(c) {{ return c.y; }}),
    y: items.map(function(c) {{ return c[axis]; }}),
    mode: 'markers',
    type: 'scatter',
    name: 'Separate item',
    marker: {{
      color: '#e67e22', symbol: 'square', size: 15,
      line: {{ color: '#fff', width: 1.2 }}
    }},
    customdata: items.map(function(c) {{
      return [c.label, c.w, c.x, c.y, c.z, weightVisXcText(c.xc)];
    }}),
    hovertemplate:
      '<b>%{{customdata[0]}}</b> (carried by no ' + station + ')<br>'
      + 'W = %{{customdata[1]:.2f}} lbs<br>'
      + 'cg X = %{{customdata[2]:.2f}} in  (x/c = %{{customdata[5]}})<br>'
      + 'cg Y = %{{customdata[3]:.2f}} in<br>'
      + 'cg Z = %{{customdata[4]:.2f}} in<extra></extra>'
  }};
}}

function weightVisTotalTrace(axis) {{
  // [AI] The whole semispan wing, stations and separate items together. Each grouping
  // carries its own, and the two differ a little: they agree on the weight and on the
  // wingbox cg exactly, and part on the smeared secondary items, which the rib
  // grouping places at the middle of each half bay rather than of the whole bay.
  var total = weightVisGroup().total;
  if (!total || !(total.w > 0)) return null;
  return {{
    x: [total.y],
    y: [total[axis]],
    mode: 'markers',
    type: 'scatter',
    name: 'Wing cg (semispan)',
    marker: {{
      color: '#8e44ad', symbol: 'star', size: 20,
      line: {{ color: '#fff', width: 1.2 }}
    }},
    hovertemplate:
      '<b>Semispan wing cg</b><br>'
      + 'W = ' + total.w.toFixed(2) + ' lbs'
      + ' (wingbox ' + Number(total.w_wingbox || 0).toFixed(2) + ')<br>'
      + 'cg X = ' + total.x.toFixed(2) + ' in  (x/c = '
      + weightVisXcText(total.xc) + ')<br>'
      + 'cg Y = ' + total.y.toFixed(2) + ' in<br>'
      + 'cg Z = ' + total.z.toFixed(2) + ' in<extra></extra>'
  }};
}}

function weightVisSemispanSurfaces(traces) {{
  // [AI] Same control-surface polygons as the Planform tab, right wing only.
  var surfaceSpecs = SURFACE_SPECS;
  surfaceSpecs.forEach(function(spec) {{
    var polygons = (PLANFORM.surfaces && PLANFORM.surfaces[spec.key]) || [];
    var shown = false;
    polygons.forEach(function(poly) {{
      // [AI] Polygons come mirrored about the centerline; keep the right-hand one.
      var onRight = poly.ws.every(function(v) {{ return v >= 0; }});
      if (!onRight) return;
      traces.push({{
        x: poly.ws, y: poly.x, mode: 'lines', name: spec.name,
        fill: 'toself', fillcolor: spec.color,
        line: {{ color: planformLineColor(spec.color), width: 1 }},
        showlegend: !shown,
        legendgroup: spec.name,
        hoverinfo: 'skip'
      }});
      shown = true;
    }});
  }});
}}

function weightVisSpanRange() {{
  // [AI] Both views share one span range so a cg can be read across the two plots.
  var maxWs = 0;
  (PLANFORM.semispan.ws || []).forEach(function(v) {{
    if (v > maxWs) maxWs = v;
  }});
  Object.keys(PLANFORM.surfaces || {{}}).forEach(function(key) {{
    (PLANFORM.surfaces[key] || []).forEach(function(poly) {{
      poly.ws.forEach(function(v) {{ if (v > maxWs) maxWs = v; }});
    }});
  }});
  var pad = 0.03 * (maxWs || 1);
  return [-pad, maxWs + pad];
}}

function weightVisNacelleShapes() {{
  var shapes = [];
  [
    {{ ws: PLANFORM.nacelle_ib_y, color: '#ff7f0e' }},
    {{ ws: PLANFORM.nacelle_ob_y, color: '#9467bd' }}
  ].forEach(function(spec) {{
    if (spec.ws == null || spec.ws <= 0) return;
    shapes.push({{
      type: 'line', x0: spec.ws, x1: spec.ws, y0: 0, y1: 1, yref: 'paper',
      line: {{ color: spec.color, width: 1.5, dash: 'dot' }}
    }});
  }});
  return shapes;
}}

function drawWeightVisualsTopPlot(sizeByWeight) {{
  var ss = PLANFORM.semispan;
  var traces = [];
  weightVisSemispanSurfaces(traces);

  traces.push(
    {{ x: ss.ws, y: ss.le, mode: 'lines', name: 'Leading Edge',
       line: {{ color: '#1f77b4', width: 2 }}, hoverinfo: 'skip' }},
    {{ x: ss.ws, y: ss.te, mode: 'lines', name: 'Trailing Edge',
       line: {{ color: '#1f77b4', width: 2 }}, hoverinfo: 'skip' }},
    {{ x: ss.ws, y: ss.fwd, mode: 'lines', name: 'Fwd Spar',
       line: {{ color: '#d62728', width: 1.5, dash: 'dash' }}, hoverinfo: 'skip' }},
    {{ x: ss.ws, y: ss.aft, mode: 'lines', name: 'Aft Spar',
       line: {{ color: '#2ca02c', width: 1.5, dash: 'dash' }}, hoverinfo: 'skip' }},
    {{ x: ss.ws, y: ss.quarter_chord, mode: 'lines', name: 'Quarter Chord',
       line: {{ color: '#ff7f0e', width: 2, dash: 'dot' }}, hoverinfo: 'skip' }},
    {{ x: ss.ws, y: ss.elastic_axis, mode: 'lines', name: 'Elastic Axis',
       line: {{ color: '#9467bd', width: 2, dash: 'dashdot' }}, hoverinfo: 'skip' }}
  );

  // [AI] Rib lines, fwd spar to aft spar at each station.
  // [OAS] ss.ribs when the payload carries it -- see initResultsOverview.
  (ss.ribs || ss.bays).forEach(function(bp) {{
    traces.push({{
      x: [bp.ws, bp.ws], y: [bp.fwd, bp.aft], mode: 'lines',
      line: {{ color: '#333', width: 0.5 }},
      showlegend: false, hoverinfo: 'skip'
    }});
  }});

  if (ss.bay_labels) {{
    // [AI] The Planform tab centres the bay number in the box, which is exactly where
    // the bay cg markers land here, so put it just ahead of the front spar instead
    // ('top' is forward: the X axis is reversed).
    traces.push({{
      x: ss.bay_labels.map(function(l) {{ return l.ws; }}),
      y: ss.bay_labels.map(function(l) {{
        return interpSemispanGeom(ss, l.ws).fwd;
      }}),
      text: ss.bay_labels.map(function(l) {{ return l.text; }}),
      mode: 'text', name: 'Bay Number',
      textposition: 'top center',
      textfont: {{ color: '#111', size: 12 }},
      showlegend: false, hoverinfo: 'skip'
    }});
  }}

  [weightVisStationTrace('x', sizeByWeight), weightVisSeparateTrace('x'),
   weightVisTotalTrace('x')].forEach(
    function(t) {{ if (t) traces.push(t); }}
  );

  var layout = {{
    xaxis: {{ title: 'Spanwise wing station Y (in)', range: weightVisSpanRange() }},
    yaxis: {{ title: 'Fuselage Station X (in)', autorange: 'reversed',
              scaleanchor: 'x', scaleratio: 1 }},
    legend: {{ orientation: 'h', x: 0.5, xanchor: 'center', y: -0.26,
               font: {{ size: 11 }} }},
    margin: {{ t: 20, b: 130, l: 80, r: 30 }},
    shapes: weightVisNacelleShapes(),
    hovermode: 'closest'
  }};
  Plotly.newPlot('weightvis-top-plot', traces, layout, {{ responsive: true }})
    .then(function() {{ bindWeightVisStationClick('weightvis-top-plot'); }});
}}

function drawWeightVisualsAftPlot(sizeByWeight, trueScale) {{
  var env = WEIGHTVIS.envelope || [];
  var traces = [];

  if (env.length) {{
    var ws = env.map(function(e) {{ return e.ws; }});
    var zTop = env.map(function(e) {{ return e.z_top; }});
    var zBot = env.map(function(e) {{ return e.z_bot; }});
    traces.push({{
      x: ws.concat(ws.slice().reverse()),
      y: zTop.concat(zBot.slice().reverse()),
      mode: 'lines', name: 'Wingbox envelope',
      fill: 'toself', fillcolor: 'rgba(44, 62, 80, 0.08)',
      line: {{ color: '#2c3e50', width: 1.5 }},
      hoverinfo: 'skip'
    }});
    traces.push({{
      x: ws, y: env.map(function(e) {{ return e.z_qc; }}),
      mode: 'lines', name: 'Quarter-chord height',
      line: {{ color: '#ff7f0e', width: 1.5, dash: 'dot' }},
      hoverinfo: 'skip'
    }});
    // [AI] Rib lines, box bottom to box top at each station.
    env.forEach(function(e) {{
      traces.push({{
        x: [e.ws, e.ws], y: [e.z_bot, e.z_top], mode: 'lines',
        line: {{ color: '#333', width: 0.5 }},
        showlegend: false, hoverinfo: 'skip'
      }});
    }});
  }}

  [weightVisStationTrace('z', sizeByWeight), weightVisSeparateTrace('z'),
   weightVisTotalTrace('z')].forEach(
    function(t) {{ if (t) traces.push(t); }}
  );

  var yaxis = {{ title: 'Height Z (in)' }};
  if (trueScale) {{
    yaxis.scaleanchor = 'x';
    yaxis.scaleratio = 1;
  }}
  var layout = {{
    xaxis: {{ title: 'Spanwise wing station Y (in)', range: weightVisSpanRange() }},
    yaxis: yaxis,
    legend: {{ orientation: 'h', x: 0.5, xanchor: 'center', y: -0.34,
               font: {{ size: 11 }} }},
    margin: {{ t: 20, b: 130, l: 80, r: 30 }},
    shapes: weightVisNacelleShapes(),
    hovermode: 'closest'
  }};
  Plotly.newPlot('weightvis-aft-plot', traces, layout, {{ responsive: true }})
    .then(function() {{ bindWeightVisStationClick('weightvis-aft-plot'); }});

  var note = document.getElementById('weightvis-aft-note');
  if (note) {{
    note.textContent = trueScale
      ? 'Drawn 1:1, so the box depth is to the same scale as the span.'
      : 'Z is exaggerated relative to span so the spread of the cgs through the box '
        + 'depth is readable. Switch to true scale to see the real proportions.';
  }}
}}

function drawWeightVisuals() {{
  if (!WEIGHTVIS) return;
  var sizeEl = document.getElementById('weightvis-size-mode');
  var scaleEl = document.getElementById('weightvis-aft-scale');
  var sizeByWeight = !sizeEl || sizeEl.value === 'weight';
  var trueScale = scaleEl && scaleEl.value === 'true';
  drawWeightVisualsTopPlot(sizeByWeight);
  drawWeightVisualsAftPlot(sizeByWeight, trueScale);
}}

// [AI] ---- Weight Visuals: station selection ----

function bindWeightVisStationClick(divId) {{
  var el = document.getElementById(divId);
  if (!el) return;
  // [AI] Plotly.newPlot drops previously registered handlers, so this rebinds on every
  // redraw. Selecting a station is idempotent, so a duplicate binding would be
  // harmless anyway.
  el.on('plotly_click', function(ev) {{
    if (!ev.points || !ev.points.length) return;
    var pt = ev.points[0];
    if (pt.data && pt.data.name === weightVisGroupSpec().trace && pt.customdata) {{
      selectWeightVisStation(pt.customdata[0]);
    }}
  }});
}}

function highlightWeightVisSelectedStation() {{
  var stations = weightVisStations();
  if (!stations.length) return;
  var traceName = weightVisGroupSpec().trace;
  var update = {{
    'marker.color': [weightVisStationColors(stations)],
    'marker.symbol': [weightVisStationSymbols(stations)]
  }};
  ['weightvis-top-plot', 'weightvis-aft-plot'].forEach(function(id) {{
    var el = document.getElementById(id);
    if (!el || !el.data) return;
    for (var i = 0; i < el.data.length; i++) {{
      if (el.data[i].name === traceName) {{
        Plotly.restyle(el, update, [i]);
        return;
      }}
    }}
  }});
}}

function selectWeightVisStation(stationId) {{
  weightVisSelectedStation = Number(stationId);
  var sel = document.getElementById('weightvis-station-select');
  if (sel) sel.value = String(weightVisSelectedStation);
  highlightWeightVisSelectedStation();
  renderWeightVisStationPanel();
}}

function onWeightVisStationSelect() {{
  var sel = document.getElementById('weightvis-station-select');
  if (sel) selectWeightVisStation(sel.value);
}}

// [AI] Switching grouping relabels the side panel, rebuilds the picker and redraws
// both plots. The station IDs are the same in either grouping -- rib n stands at the
// inboard end of bay n -- so whatever was selected stays selected.
function onWeightVisGroupChange() {{
  applyWeightVisGroupLabels();
  populateWeightVisStationSelect();
  renderWeightVisualsReadout();
  drawWeightVisuals();
  selectWeightVisStation(weightVisSelectedStation);
}}

function applyWeightVisGroupLabels() {{
  var spec = weightVisGroupSpec();
  var labels = [
    ['weightvis-side-title', spec.name + ' contributions'],
    ['weightvis-station-label', spec.name + ':'],
    ['weightvis-station-hint', 'or click a ' + spec.name.toLowerCase()
      + ' cg on a plot'],
    ['weightvis-group-note', spec.note]
  ];
  labels.forEach(function(pair) {{
    var el = document.getElementById(pair[0]);
    if (el) el.textContent = pair[1];
  }});
}}

function populateWeightVisStationSelect() {{
  var sel = document.getElementById('weightvis-station-select');
  if (!sel) return;
  var name = weightVisGroupSpec().name;
  sel.innerHTML = weightVisStations().map(function(s) {{
    return '<option value="' + s.id + '">' + name + ' ' + s.id
      + ' (WS ' + Number(s.ws).toFixed(1) + ')</option>';
  }}).join('');
}}

function renderWeightVisStationPanel() {{
  var summaryEl = document.getElementById('weightvis-station-summary');
  var tableEl = document.getElementById('weightvis-component-table');
  if (!summaryEl || !tableEl) return;

  var spec = weightVisGroupSpec();
  var group = weightVisGroup();
  var station = weightVisStations().filter(function(s) {{
    return s.id === weightVisSelectedStation;
  }})[0];
  if (!station) {{
    summaryEl.textContent = 'Select a ' + spec.name.toLowerCase()
      + ' to list its components.';
    tableEl.innerHTML = '';
    return;
  }}

  summaryEl.innerHTML =
    '<b>' + spec.name + ' ' + station.id + '</b> at WS '
    + Number(station.ws).toFixed(1) + ' in<br>'
    + 'W = ' + Number(station.w).toFixed(2) + ' lbs'
    + '  (wingbox ' + Number(station.w_wingbox || 0).toFixed(2) + ' lbs)<br>'
    + 'cg  X = ' + Number(station.x).toFixed(2)
    + '   Y = ' + Number(station.y).toFixed(2)
    + '   Z = ' + Number(station.z).toFixed(2) + ' in'
    + '   x/c = ' + weightVisXcText(station.xc);

  var comps = (group.components || []).filter(function(c) {{
    return c.id === weightVisSelectedStation;
  }});
  var wTotal = station.w || 0;

  // [AI] A rib carries half bays out of the bay on either side of it, and the two can
  // be the same panel of two neighbouring bays, so the source bay is worth a column
  // there. Grouped per bay it would only repeat the row's own station.
  var showFromBay = weightVisGroupMode() === 'rib';
  var html = '<thead><tr>'
    + '<th>Component</th>' + (showFromBay ? '<th>From bay</th>' : '')
    + '<th>Type</th><th>W (lbs)</th><th>% of ' + spec.name.toLowerCase() + '</th>'
    + '<th>X (in)</th><th>x/c</th><th>Y (in)</th><th>Z (in)</th>'
    + '</tr></thead><tbody>';
  comps.forEach(function(c) {{
    var pct = wTotal > 0 ? (100 * c.w / wTotal) : 0;
    html += '<tr>'
      + '<td style="text-align:left">'
      + '<span class="weightvis-kind-dot" style="background:'
      + weightVisKindColor(c.kind) + '"></span>'
      + escapeHtml(c.label) + '</td>'
      + (showFromBay ? '<td>' + c.from_bay + '</td>' : '')
      + '<td style="text-align:left">' + escapeHtml(c.kind) + '</td>'
      + '<td>' + c.w.toFixed(3) + '</td>'
      + '<td>' + pct.toFixed(1) + '</td>'
      + '<td>' + c.x.toFixed(2) + '</td>'
      + '<td>' + weightVisXcText(c.xc) + '</td>'
      + '<td>' + c.y.toFixed(2) + '</td>'
      + '<td>' + c.z.toFixed(2) + '</td>'
      + '</tr>';
  }});
  html += '<tr class="weight-table-row-total">'
    + '<td style="text-align:left">' + spec.name + ' ' + station.id + ' total</td>'
    + (showFromBay ? '<td></td>' : '')
    + '<td></td>'
    + '<td>' + Number(station.w).toFixed(3) + '</td>'
    + '<td>100.0</td>'
    + '<td>' + Number(station.x).toFixed(2) + '</td>'
    + '<td>' + weightVisXcText(station.xc) + '</td>'
    + '<td>' + Number(station.y).toFixed(2) + '</td>'
    + '<td>' + Number(station.z).toFixed(2) + '</td>'
    + '</tr></tbody>';
  tableEl.innerHTML = html;
}}

function renderWeightVisualsReadout() {{
  var el = document.getElementById('weightvis-cg-readout');
  if (!el) return;
  var t = weightVisGroup().total || {{}};
  el.textContent =
    'Semispan wing  W = ' + Number(t.w || 0).toFixed(1) + ' lbs'
    + ' (wingbox ' + Number(t.w_wingbox || 0).toFixed(1) + ')'
    + '   cg X = ' + Number(t.x || 0).toFixed(1)
    + '   Y = ' + Number(t.y || 0).toFixed(1)
    + '   Z = ' + Number(t.z || 0).toFixed(1) + ' in'
    + '   x/c = ' + weightVisXcText(t.xc);
}}

function initWeightVisuals() {{
  var emptyEl = document.getElementById('weightvis-empty');
  var bodyEl = document.getElementById('weightvis-body');
  var hasData = WEIGHTVIS
    && WEIGHTVIS.groupings
    && WEIGHTVIS.groupings.bay
    && (WEIGHTVIS.groupings.bay.stations || []).length
    && PLANFORM.semispan
    && PLANFORM.semispan.ws.length;
  if (!hasData) {{
    if (emptyEl) emptyEl.style.display = '';
    if (bodyEl) bodyEl.style.display = 'none';
    return;
  }}
  if (emptyEl) emptyEl.style.display = 'none';
  if (bodyEl) bodyEl.style.display = 'block';

  // [AI] A deck without a rib breakdown keeps the toolbar honest by losing the option
  // rather than offering one that draws nothing.
  var groupEl = document.getElementById('weightvis-group-mode');
  if (groupEl && !WEIGHTVIS.groupings.rib) {{
    groupEl.value = 'bay';
    groupEl.disabled = true;
  }}

  weightVisSelectedStation = weightVisStations()[0].id;
  onWeightVisGroupChange();
}}

// ---- Loading ----
// The two graphs on the left show the applied loads of ONE selected load case: the
// arrays loading/wing_loading.py integrated its VMT from. The two on the right show
// every load case for one selected VMT component, with the selected case drawn thick
// over the dimmed rest so it can be picked out of the bundle.

function loadingCase() {{
  return LOADING.cases[parseInt(document.getElementById('loading-lc-select').value)];
}}

// VMT datasets are named from sanitized filenames while LOADING carries the raw load
// case names, so match on alphanumerics only -- the same rule as orderedVmtIndices.
// Returns -1 for a load case with no VMT file, which leaves nothing highlighted.
function loadingVmtIndex() {{
  function key(s) {{ return String(s).toLowerCase().replace(/[^a-z0-9]/g, ''); }}
  var k = key(loadingCase().name);
  for (var i = 0; i < VMT_DATA.length; i++) {{
    if (key(VMT_DATA[i].name) === k) return i;
  }}
  return -1;
}}

// One colour per load case, taken from its position in VMT_DATA so the left-hand
// graphs and the thickened VMT line are drawn in the same colour.
function loadingColor() {{
  var i = loadingVmtIndex();
  if (i < 0) i = parseInt(document.getElementById('loading-lc-select').value);
  return COLORS[i % COLORS.length];
}}

function loadingPlot(id, traces, layout) {{
  var el = document.getElementById(id);
  if (el.data) Plotly.react(el, traces, layout, {{ responsive: true }});
  else Plotly.newPlot(el, traces, layout, {{ responsive: true }});
}}

// [OAS] ---- Native curve plus resampled markers ----
// The reason this report exists is to be read next to the WingCalc one, and the two
// do not share a spanwise grid: OAS integrates on its own 34-panel cosine-clustered
// mesh, WingCalc on 20 bay stations. Plotting only OAS's grid makes a station-by-
// station comparison a ruler exercise, and resampling OAS ONTO WingCalc's grid throws
// away the resolution that is the whole reason the two disagree outboard.
//
// So both are drawn: the LINE is OAS on its own grid, untouched, and the MARKERS are
// the same quantity linearly resampled onto WingCalc's 20 bay stations -- the numbers
// to read off against the other report. A case carries the markers as a '<key>_bay'
// array beside 'WS_bay'; where it carries neither, nothing extra is drawn and this is
// a no-op, which is what keeps the function safe to call from every plot.
function pushBayMarkers(traces, c, key, name, color) {{
  var xs = c.WS_bay, ys = c[key + '_bay'];
  if (!xs || !ys || !xs.length || xs.length !== ys.length) return;
  traces.push({{
    x: xs, y: ys, type: 'scatter', mode: 'markers',
    name: name + ' at WingCalc bay stations',
    marker: {{ color: color, size: 7, symbol: 'circle-open',
               line: {{ color: color, width: 1.6 }} }},
    hovertemplate: 'WS %{{x:.1f}} in<br>%{{y:.4f}}<extra>' + name + ' (bay station)</extra>'
  }});
}}

function initLoading() {{
  if (!LOADING || !LOADING.cases || LOADING.cases.length === 0) {{
    document.getElementById('loading-empty').style.display = 'block';
    return;
  }}
  document.getElementById('loading-body').style.display = 'block';

  var lcSel = document.getElementById('loading-lc-select');
  LOADING.cases.forEach(function(c, i) {{
    var opt = document.createElement('option');
    opt.value = i;
    // [AI mod] Spelled out on the selector because everything this tab draws is at
    // limit level: loading/wing_loading.py applies the case's own ult_factor when it
    // samples a station for margins, and nothing upstream of that has scaled these
    // arrays. Only the label carries the note -- c.name stays raw, since it is what
    // matches the VMT dataset.
    opt.textContent = c.name + ' - Limit Load';
    lcSel.appendChild(opt);
  }});

  // Torque on top, bending below: the two components this tab is opened for.
  [['a', 'Mx'], ['b', 'My']].forEach(function(slot) {{
    var sel = document.getElementById('loading-vmt-' + slot[0] + '-select');
    VMT_COMP.forEach(function(comp) {{
      var opt = document.createElement('option');
      opt.value = comp[0];
      opt.textContent = comp[0] + ' - ' + comp[1] + ' (' + comp[2] + ')';
      sel.appendChild(opt);
    }});
    sel.value = slot[1];
  }});

  drawLoading();
}}

function drawLoading() {{
  drawLoadingAero();
  drawLoadingWeight();
  drawLoadingVmt('a');
  drawLoadingVmt('b');
}}

// What the loading distribution dropdown selects: the running lift or drag in
// lbs/in, or the section Cl or Cd shape -- the running load divided by the local
// chord, normalised to its peak. See _extract_loading_data for why the dynamic
// pressure drops out of a normalised coefficient but the chord does not.
var LOADING_AERO_MODES = {{
  lift: {{ key: 'lift', title: 'Running lift (lbs/in)',
          empty: 'This load case carries no lift.' }},
  drag: {{ key: 'drag', title: 'Running drag (lbs/in), positive aft',
          empty: 'This load case carries no wing drag.' }},
  cl:   {{ key: 'cl',   title: 'Cl norm. - l/c shape, peak = 1',
          empty: 'This load case carries no lift.' }},
  cd:   {{ key: 'cd',   title: 'Cd norm. - d/c shape, peak = 1',
          empty: 'This load case carries no wing drag.' }}
}};

function drawLoadingAero() {{
  var c = loadingCase();
  var mode = LOADING_AERO_MODES[document.getElementById('loading-aero-select').value];
  var y = c[mode.key];

  var peak = 0;
  y.forEach(function(v) {{ if (Math.abs(v) > peak) peak = Math.abs(v); }});

  // An all-zero distribution is a real answer for a ground case or a case with no
  // wing drag, so say so on the plot rather than leaving a flat line to be read as
  // missing data.
  var annotations = [];
  if (peak === 0) {{
    annotations.push({{
      text: mode.empty, showarrow: false,
      xref: 'paper', yref: 'paper', x: 0.5, y: 0.5,
      font: {{ size: 13, color: '#8a949e' }}
    }});
  }}

  var aeroTraces = [{{
    x: c.WS, y: y, type: 'scatter', mode: 'lines',
    name: c.name,
    line: {{ color: loadingColor(), width: 2.2 }},
    fill: 'tozeroy', fillcolor: 'rgba(44,62,80,0.07)'
  }}];
  // [OAS] Same curve resampled onto WingCalc's bay stations -- see pushBayMarkers.
  pushBayMarkers(aeroTraces, c, mode.key, mode.title, '#c0392b');

  // [OAS] The aero runs the FULL span, winglet included, because the VLM loads the
  // winglet and the FEM carries that load -- about 1.4% of half-wing lift and 3.1% of
  // the root bending moment. The structure stops short of it, at the winglet root, so
  // the two grids genuinely differ and the plot says where. Cutting the lift curve at
  // the structural station instead would draw a wing being handed free tip lift.
  if (c.cut_ws_in) {{
    annotations.push({{
      x: c.cut_ws_in, y: 1.0, xref: 'x', yref: 'paper',
      text: 'structure ends ' + c.cut_ws_in.toFixed(1) + ' in',
      showarrow: false, xanchor: 'right', yanchor: 'top',
      font: {{ size: 11, color: '#7f8c8d' }}
    }});
  }}

  loadingPlot('loading-aero-plot', aeroTraces, {{
    margin: {{ t: 8, b: 46, l: 78, r: 18 }},
    xaxis: {{ title: 'Wing Station (in)', zeroline: false }},
    yaxis: {{ title: mode.title, zeroline: true, zerolinecolor: '#bbb' }},
    annotations: annotations,
    shapes: c.cut_ws_in ? [{{
      type: 'line', x0: c.cut_ws_in, x1: c.cut_ws_in, y0: 0, y1: 1,
      xref: 'x', yref: 'paper',
      line: {{ color: '#95a5a6', width: 1.2, dash: 'dot' }}
    }}] : [],
    showlegend: false,
    hovermode: 'x unified'
  }});
}}

// Load from punctual loading: the distributed inertia of the structure and the fuel
// as curves, and everything that is really one force -- each nacelle and the gear
// reaction -- as a single labelled arrow. Fz is the vertical component of the inertia
// load; the gX-driven Fx of the same masses is chordwise and belongs on the VMT
// graphs, not here.
function drawLoadingWeight() {{
  var c = loadingCase();
  var series = [
    {{ key: 'w_dist',   name: 'Distributed total', color: '#c0392b', width: 2.4 }},
    {{ key: 'w_struct', name: 'Structure',         color: '#2c3e50', width: 1.5 }},
    {{ key: 'w_fuel',   name: 'Fuel',              color: '#16a085', width: 1.5 }}
  ];
  // [OAS] The weight curves live on the STRUCTURAL grid, which stops at the winglet
  // root, while the aero curves on the other plot run the full span. Drawing both off
  // c.WS would stretch the structural arrays over stations they have no value at.
  var wsW = c.WS_struct || c.WS;
  var traces = series.map(function(sr) {{
    return {{
      x: wsW, y: c[sr.key], type: 'scatter', mode: 'lines',
      name: sr.name, line: {{ color: sr.color, width: sr.width }}
    }};
  }});
  // [OAS] The distributed structural weight is the one curve on this plot that is a
  // direct OAS-vs-WingCalc comparison -- it is the sized box, station by station -- so
  // it is the one that carries the bay-station markers. See pushBayMarkers.
  pushBayMarkers(traces, c, 'w_struct', 'Structure', '#2c3e50');

  var yMin = 0, yMax = 0;
  series.forEach(function(sr) {{
    c[sr.key].forEach(function(v) {{
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }});
  }});
  var peak = Math.max(Math.abs(yMin), Math.abs(yMax)) || 1;

  // Arrows at every Nth station only: the load grid is one station per inch, so the
  // full set would be a solid block of ink. Stubs below 2% of the peak are dropped
  // because the arrowhead alone would be taller than the line under it.
  var annotations = [];
  var step = Math.max(1, Math.round(wsW.length / 40));
  for (var i = 0; i < wsW.length; i += step) {{
    var v = c.w_dist[i];
    if (Math.abs(v) < peak * 0.02) continue;
    annotations.push({{
      x: wsW[i], y: v, ax: wsW[i], ay: 0,
      xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
      showarrow: true, arrowhead: 2, arrowsize: 1.1, arrowwidth: 1.1,
      arrowcolor: 'rgba(192,57,43,0.5)', text: ''
    }});
  }}

  // One arrow per point load, drawn from the zero line so its direction is the load's
  // own. These are lbs on a lbs/in axis and the W2F and gear reactions sit orders of
  // magnitude
  // above the distributed load, so they are scaled against each other -- the largest
  // at 85% of the distributed peak -- and kept out of the traces so they cannot drive
  // the autoscale. Each carries its magnitude, and the axis title says they are not
  // to scale.
  var pmax = 0;
  c.points.forEach(function(p) {{ if (Math.abs(p.fz) > pmax) pmax = Math.abs(p.fz); }});
  if (pmax > 0) {{
    c.points.forEach(function(p) {{
      var len = p.fz / pmax * peak * 0.85;
      if (len > yMax) yMax = len;
      if (len < yMin) yMin = len;
      annotations.push({{
        x: p.ws, y: len, ax: p.ws, ay: 0,
        xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
        showarrow: true, arrowhead: 2, arrowsize: 1.2, arrowwidth: 2.4,
        arrowcolor: '#8e44ad', text: ''
      }});
      annotations.push({{
        x: p.ws, y: len, xref: 'x', yref: 'y', showarrow: false,
        text: p.label + '<br>' + (p.fz > 0 ? '+' : '-') +
              Math.abs(Math.round(p.fz)).toLocaleString() + ' lbs',
        font: {{ size: 10, color: '#8e44ad' }},
        xanchor: 'left', yanchor: p.fz > 0 ? 'bottom' : 'top', xshift: 5
      }});
    }});
  }}

  var pad = (yMax - yMin) * 0.18 || 1;
  loadingPlot('loading-weight-plot', traces, {{
    margin: {{ t: 8, b: 46, l: 84, r: 20 }},
    xaxis: {{ title: 'Wing Station (in)', zeroline: false }},
    yaxis: {{
      title: 'Distributed load (lbs/in)<br><i>arrows labelled in lbs, not to scale</i>',
      range: [yMin - pad, yMax + pad],
      zeroline: true, zerolinecolor: '#666', zerolinewidth: 1.5
    }},
    annotations: annotations,
    legend: {{ x: 1.02, xanchor: 'left', y: 0.5, yanchor: 'middle',
              font: {{ size: 12 }} }},
    hovermode: 'x unified'
  }});
}}

// One VMT component for every load case. The selected case keeps its normal trace so
// the legend and the hover readout stay in deck order, and gets a second thick trace
// drawn on top of the bundle.
function drawLoadingVmt(slot) {{
  var id = 'loading-vmt-' + slot + '-plot';
  if (VMT_DATA.length === 0) {{
    document.getElementById(id).innerHTML =
      '<p style="padding:40px;color:#888;">No VMT data available.</p>';
    return;
  }}
  var col = document.getElementById('loading-vmt-' + slot + '-select').value;
  var comp = null;
  VMT_COMP.forEach(function(cc) {{ if (cc[0] === col) comp = cc; }});
  var selIdx = loadingVmtIndex();

  // Every case solid at full opacity: the selected one is picked out by width
  // alone, which keeps the unselected colours readable against the legend.
  // [AI mod] Legend in the Stress Results load case order -- the order the load case
  // dropdown above these graphs is built in -- rather than the order the VMT files
  // happened to be read in. The colour stays keyed on the VMT index and not on the
  // position in this list, so a case keeps one colour across every tab.
  var traces = orderedVmtIndices().map(function(vi) {{
    var ds = VMT_DATA[vi];
    return {{
      x: ds.WS, y: ds[col], type: 'scatter', mode: 'lines',
      name: ds.name,
      line: {{ color: COLORS[vi % COLORS.length], width: 1.4 }}
    }};
  }});
  if (selIdx >= 0) {{
    var sel = VMT_DATA[selIdx];
    traces.push({{
      x: sel.WS, y: sel[col], type: 'scatter', mode: 'lines',
      name: sel.name, showlegend: false, hoverinfo: 'skip',
      line: {{ color: COLORS[selIdx % COLORS.length], width: 4 }}
    }});
    // [OAS] The selected case's VMT resampled onto WingCalc's bay stations, so the
    // two reports' VMT can be read against each other at the same stations rather
    // than off two different grids. See pushBayMarkers.
    pushBayMarkers(traces, sel, col, comp[0], '#2c3e50');
  }}

  loadingPlot(id, traces, {{
    margin: {{ t: 8, b: 46, l: 84, r: 20 }},
    xaxis: {{ title: 'Wing Station (in)', zeroline: false }},
    yaxis: {{
      title: comp[0] + ' - ' + comp[1] + ' (' + comp[2] + ')',
      zeroline: true, zerolinecolor: '#bbb'
    }},
    // Legend down the right-hand side: the load case names are too long to fit on
    // one row under the plot, and Plotly's default margin.autoexpand makes room for
    // it, so margin.r stays a minimum rather than a reserved strip.
    legend: {{ x: 1.02, xanchor: 'left', y: 0.5, yanchor: 'middle',
              font: {{ size: 12 }} }},
    hovermode: 'x unified'
  }});
}}

// ---- Init on load ----
window.addEventListener('load', function() {{
  initNegativeMarginWarning();
  syncChromeHeight();
  initPlanform();
  plotInitialized.planform = true;
}});
window.addEventListener('resize', function() {{
  syncChromeHeight();
  resizeCurrentTabPlots();
}});
</script>
</body>
</html>"""

    # The bundle is spliced in after formatting, not interpolated: minified JavaScript is
    # nothing but braces, and an f-string would read every one of them as a field.
    return html.replace(PLOTLY_PLACEHOLDER, plotly_script_block())
