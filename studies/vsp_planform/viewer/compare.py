"""A two-pane page that puts the WingCalc and OAS reports side by side.

The two reports are deliberately drawn by the same frontend (see
``wingcalc_frontend``), so anything that looks different between them is a
modelling difference rather than a plotting one. This page is what makes that
comparison a glance instead of two windows and a memory.

TAB SYNC IS BEST-EFFORT, AND SAYS SO
------------------------------------
Driving both panes from one tab bar means reaching into each iframe and calling
its ``switchTab``. Whether that is allowed is the browser's call, not ours:
Chrome gives every ``file://`` document its own opaque origin, so the reach
throws a SecurityError and the panes can only be driven by their own tab bars.
Firefox has historically been laxer, and anything served over http:// from one
origin works. Rather than wire a control that silently does nothing on the most
likely setup, the page probes once on load and then either shows a working tab
bar or says plainly that the panes are independent and why. A control that lies
about what it does is worse than no control.

Only the six tabs both reports carry are offered. WingCalc's Optimization tab
has no OAS counterpart -- its per-bay differential-evolution trace and OAS's
global SLSQP history are different objects -- so it is reachable from the
WingCalc pane's own tab bar and not from here.
"""

from __future__ import annotations

import html as html_lib
import json
import os
from datetime import datetime
from pathlib import Path

# The six tabs both reports carry, as (data-tab value, button label). The values
# are the frontend's own, so they are what ``switchTab`` expects.
COMMON_TABS = [
    ("planform", "Planform"),
    ("xsec", "Cross section &amp; VMT"),
    ("loading", "Loading"),
    ("results", "Stress Results"),
    ("weightvis", "Weight Visuals"),
    ("weight", "Weight Summary"),
]

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{ --chrome-h: 92px; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f5f5f5;
  overflow: hidden;
}}
#header {{
  background: #2c3e50; color: white;
  padding: 12px 24px; font-size: 20px; font-weight: 600;
  display: flex; align-items: center; gap: 32px;
}}
.header-title {{ display: flex; flex-direction: column; line-height: 1.1; }}
.header-sub {{ font-size: 12px; font-weight: 400; opacity: 0.85; margin-top: 3px; }}
.tabs {{ display: flex; gap: 4px; flex-wrap: wrap; }}
.tab-btn {{
  background: rgba(255,255,255,0.15); color: white; border: none;
  padding: 8px 20px; cursor: pointer; font-size: 14px; border-radius: 6px 6px 0 0;
  transition: background 0.15s;
}}
.tab-btn:hover {{ background: rgba(255,255,255,0.25); }}
.tab-btn.active {{ background: white; color: #2c3e50; font-weight: 600; }}
#sync-note {{
  padding: 8px 24px; font-size: 13px;
  background: #fffaf0; color: #8a6d3b; border-bottom: 1px solid #f0e0c0;
}}
#sync-note code {{ font-family: 'Consolas', 'Courier New', monospace; }}
.panes {{ display: flex; height: calc(100vh - var(--chrome-h)); }}
.pane {{ display: flex; flex-direction: column; min-width: 0; }}
.pane-head {{
  flex: 0 0 auto; display: flex; align-items: baseline; gap: 12px;
  padding: 6px 12px; background: #fff; border-bottom: 1px solid #ddd;
  font-size: 13px; color: #2c3e50; font-weight: 600;
}}
.pane-head .sub {{ font-weight: 400; color: #7f8c8d; font-size: 12px; }}
.pane-head a {{ margin-left: auto; font-weight: 400; font-size: 12px; color: #2980b9; }}
.pane iframe {{ flex: 1 1 auto; width: 100%; border: none; background: #f5f5f5; }}
.pane.missing .placeholder {{
  flex: 1 1 auto; display: flex; align-items: center; justify-content: center;
  padding: 32px; text-align: center; color: #7f8c8d; font-size: 14px; line-height: 1.6;
}}
#splitter {{
  flex: 0 0 6px; cursor: col-resize; background: #dfe6e9;
  border-left: 1px solid #cfd8dc; border-right: 1px solid #cfd8dc;
}}
#splitter:hover {{ background: #b2bec3; }}
</style>
</head>
<body>

<div id="header">
  <div class="header-title">
    <span>{heading}</span>
    {subtitle_html}
  </div>
  <div class="tabs" id="tab-bar" style="display:none">{tab_buttons}</div>
</div>

<div id="sync-note"></div>

<div class="panes" id="panes">
  <div class="pane" id="pane-left" style="flex: 1 1 50%">
    <div class="pane-head">
      <span>{left_label}</span><span class="sub">{left_sub}</span>
      <a href="{left_src}" target="_blank">open alone &#8599;</a>
    </div>
    {left_body}
  </div>
  <div id="splitter"></div>
  <div class="pane" id="pane-right" style="flex: 1 1 50%">
    <div class="pane-head">
      <span>{right_label}</span><span class="sub">{right_sub}</span>
      <a href="{right_src}" target="_blank">open alone &#8599;</a>
    </div>
    {right_body}
  </div>
</div>

<script>
var FRAME_IDS = {frame_ids};

function frames_() {{
  return FRAME_IDS.map(function (id) {{ return document.getElementById(id); }})
                  .filter(function (f) {{ return f; }});
}}

/* One probe, once, on load: can we actually reach inside both panes? Chrome says
   no for file:// (every such document is its own opaque origin) and throws a
   SecurityError on contentDocument. Whatever the answer, the page then tells the
   truth about which controls it has. */
function syncAvailable() {{
  var fs = frames_();
  if (!fs.length) return false;
  for (var i = 0; i < fs.length; i++) {{
    try {{
      var w = fs[i].contentWindow;
      if (!w || !w.document || typeof w.switchTab !== 'function') return false;
    }} catch (e) {{
      return false;
    }}
  }}
  return true;
}}

function switchBoth(name, btn) {{
  frames_().forEach(function (f) {{
    try {{
      var w = f.contentWindow;
      var b = w.document.querySelector('.tab-btn[data-tab="' + name + '"]');
      if (b) w.switchTab(name, b);
    }} catch (e) {{ /* pane went away or turned opaque; the others still switch */ }}
  }});
  var bar = document.getElementById('tab-bar');
  Array.prototype.forEach.call(bar.querySelectorAll('.tab-btn'), function (b) {{
    b.classList.remove('active');
  }});
  if (btn) btn.classList.add('active');
}}

function syncChromeHeight() {{
  var h = document.getElementById('header').offsetHeight
        + document.getElementById('sync-note').offsetHeight;
  document.documentElement.style.setProperty('--chrome-h', h + 'px');
}}

var splitting = false;
document.getElementById('splitter').addEventListener('mousedown', function (e) {{
  splitting = true;
  e.preventDefault();
  /* While dragging, the iframes must stop swallowing mousemove -- without this the
     pointer crosses into a pane and the drag dies there. */
  frames_().forEach(function (f) {{ f.style.pointerEvents = 'none'; }});
}});
document.addEventListener('mousemove', function (e) {{
  if (!splitting) return;
  var panes = document.getElementById('panes');
  var frac = (e.clientX - panes.getBoundingClientRect().left) / panes.offsetWidth;
  frac = Math.min(0.9, Math.max(0.1, frac));
  document.getElementById('pane-left').style.flex = '1 1 ' + (100 * frac) + '%';
  document.getElementById('pane-right').style.flex = '1 1 ' + (100 * (1 - frac)) + '%';
}});
document.addEventListener('mouseup', function () {{
  if (!splitting) return;
  splitting = false;
  frames_().forEach(function (f) {{ f.style.pointerEvents = ''; }});
}});

window.addEventListener('load', function () {{
  var note = document.getElementById('sync-note');
  if (syncAvailable()) {{
    document.getElementById('tab-bar').style.display = 'flex';
    note.innerHTML = 'Tab buttons above drive <b>both</b> panes. Drag the divider to '
                   + 'resize. Each pane also keeps its own tab bar.';
  }} else {{
    note.innerHTML = '<b>Panes are independent.</b> This browser treats each report as '
                   + 'its own origin (normal for <code>file://</code> in Chrome), so one '
                   + 'tab bar cannot drive both &mdash; use each pane\\'s own tabs. Drag '
                   + 'the divider to resize. Serving this folder over http:// from one '
                   + 'origin re-enables the shared tab bar.';
  }}
  syncChromeHeight();
}});
window.addEventListener('resize', syncChromeHeight);
</script>
</body>
</html>
"""


def _rel(target: Path, start: Path) -> str:
    """A browser-usable relative href from ``start``'s folder to ``target``.

    ``os.path.relpath`` is what handles the two reports sitting in different
    folders; the separator is forced to ``/`` because a Windows backslash in an
    href is not a path separator to a browser.
    """
    return os.path.relpath(str(target), str(start.parent)).replace(os.sep, "/")


def _pane(src: str | None, missing_msg: str) -> str:
    if src is None:
        return f'<div class="placeholder">{html_lib.escape(missing_msg)}</div>'
    return f'<iframe id="{{id}}" src="{html_lib.escape(src)}"></iframe>'


def write_compare_page(
    out_path,
    wingcalc_html=None,
    oas_html=None,
    heading="WingCalc ↔ OAS",
    subtitle_lines=(),
    wingcalc_sub="",
    oas_sub="",
    created_at=None,
):
    """Write the side-by-side page.

    Either report may be ``None`` -- a run whose WingCalc sizing failed still has
    an OAS report worth looking at, and the missing side says what is missing
    rather than showing an empty frame.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    created_at = created_at or datetime.now()

    left_src = _rel(Path(wingcalc_html), out_path) if wingcalc_html else None
    right_src = _rel(Path(oas_html), out_path) if oas_html else None

    frame_ids = []
    left_body = _pane(left_src, "No WingCalc report for this run. The sizing either "
                                "failed or was not run; the run log says which.")
    if left_src:
        left_body = left_body.format(id="frame-wc")
        frame_ids.append("frame-wc")
    right_body = _pane(right_src, "No OAS report for this run.")
    if right_src:
        right_body = right_body.format(id="frame-oas")
        frame_ids.append("frame-oas")

    subs = list(subtitle_lines) + [created_at.strftime("%Y-%m-%d %H:%M")]
    subtitle_html = "\n    ".join(
        f'<span class="header-sub">{html_lib.escape(str(s))}</span>' for s in subs if s
    )
    tab_buttons = "".join(
        f'<button class="tab-btn{" active" if i == 0 else ""}" data-tab="{key}" '
        f"onclick=\"switchBoth('{key}', this)\">{label}</button>"
        for i, (key, label) in enumerate(COMMON_TABS)
    )

    html = _PAGE.format(
        title=html_lib.escape(f"{heading} - side by side"),
        heading=html_lib.escape(heading),
        subtitle_html=subtitle_html,
        tab_buttons=tab_buttons,
        frame_ids=json.dumps(frame_ids),
        left_label="WingCalc", left_sub=html_lib.escape(wingcalc_sub),
        left_src=html_lib.escape(left_src or ""), left_body=left_body,
        right_label="OpenAeroStruct", right_sub=html_lib.escape(oas_sub),
        right_src=html_lib.escape(right_src or ""), right_body=right_body,
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# A study's front page
# ---------------------------------------------------------------------------
_INDEX = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>{title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f5f5f5; color: #2c3e50; }}
#header {{ background: #2c3e50; color: white; padding: 12px 24px; font-size: 20px;
          font-weight: 600; }}
.header-sub {{ display: block; font-size: 12px; font-weight: 400; opacity: 0.85; margin-top: 3px; }}
main {{ padding: 20px 24px; }}
h3 {{ font-size: 15px; border-bottom: 2px solid #2c3e50; padding-bottom: 6px; margin: 0 0 12px; }}
table {{ border-collapse: collapse; background: #fff; }}
th {{ text-align: left; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em;
     color: #7f8c8d; padding: 6px 10px 8px; border-bottom: 2px solid #dfe6e9; }}
td {{ padding: 6px 10px; font-size: 14px; border-bottom: 1px solid #f0f0f0;
     font-family: 'Consolas', 'Courier New', monospace; }}
td.num {{ text-align: right; }}
td.lab {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-weight: 600; }}
td a {{ color: #2980b9; margin-right: 10px; font-family: -apple-system, 'Segoe UI', sans-serif; }}
.err {{ color: #c0392b; font-size: 12px; font-family: -apple-system, 'Segoe UI', sans-serif; }}
.note {{ margin-top: 14px; font-size: 13px; color: #555; line-height: 1.5; max-width: 980px; }}
code {{ font-family: 'Consolas', 'Courier New', monospace; }}
</style></head><body>
<div id="header">{heading}<span class="header-sub">{sub}</span></div>
<main>
<h3>Results</h3>
<table><tr><th>Design</th><th>Drag N</th><th>W_wing OAS lb</th><th>W_wing WingCalc lb</th>
<th>Range nmi</th><th>Lift centroid % semi</th><th>Tip twist deg</th><th>Reports</th></tr>
{rows}
</table>
<div class="note">{note}</div>
</main></body></html>
"""


def write_study_index(out_path, title, rows, source_json=None, created_at=None):
    """One page per study: the results table, and links to every design's reports.

    ``rows`` are dicts with ``label``, ``result`` (the design's JSON record), the
    three report paths (any may be ``None``), ``wc_w_wing_lb`` and ``error``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    created_at = created_at or datetime.now()

    def link(p, text):
        return f'<a href="{html_lib.escape(_rel(Path(p), out_path))}">{text}</a>' if p else ""

    def num(v, fmt):
        return f'<td class="num">{format(v, fmt) if v is not None else "&mdash;"}</td>'

    body = []
    for r in rows:
        res = r["result"]
        links = (link(r.get("compare_html"), "side by side") + link(r.get("oas_html"), "OAS")
                 + link(r.get("wingcalc_html"), "WingCalc"))
        err = f'<div class="err">{html_lib.escape(r["error"])}</div>' if r.get("error") else ""
        body.append(
            f'<tr><td class="lab">{html_lib.escape(r["label"])}</td>'
            + num(res.get("drag_N"), ",.1f") + num(res.get("W_wing_lb"), ",.1f")
            + num(r.get("wc_w_wing_lb"), ",.1f") + num(res.get("R_nmi"), ".2f")
            + num(100 * res["y_cl_frac_semi"] if "y_cl_frac_semi" in res else None, ".3f")
            + num(res.get("twist_tip_deg"), ".2f")
            + f"<td>{links}{err}</td></tr>")
    src = (f"The numbers above are read from <code>{html_lib.escape(str(source_json))}</code>, "
           "the study's result file." if source_json else "")
    note = (src + " W_wing OAS is WingCalc's weight build-up computed by OAS on its own box; "
            "W_wing WingCalc is WingCalc sizing the same design. Range uses the OAS weight.")
    out_path.write_text(_INDEX.format(
        title=html_lib.escape(title), heading=html_lib.escape(title),
        sub=html_lib.escape(created_at.strftime("%Y-%m-%d %H:%M")),
        rows="\n".join(body), note=note), encoding="utf-8")
    return out_path
