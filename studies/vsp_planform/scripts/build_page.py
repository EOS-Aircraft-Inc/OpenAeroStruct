"""Assemble the render gallery into a single self-contained HTML page."""

import base64
import os

HERE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "figures"
)


def data_uri(filename):
    with open(os.path.join(HERE, filename), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


FIGURES = [
    (
        "plan_l_geometry.png",
        "Plan_L, as built",
        "29 x 53 camber nodes. Region A is only &plusmn;1.27 m of the 17.63 m half-span, so the planform is "
        "almost entirely the tapered panel. The section overlay shows camber growing steadily outboard as the "
        "chord falls from 3.53 m to 0.51 m.",
    ),
    (
        "const_chord_geometry.png",
        "ConstChord, as built",
        "29 x 189 camber nodes. The constant-chord bay runs out to &plusmn;9.19 m before the taper starts, and "
        "the thickness ratio falls 0.178 &rarr; 0.100 along the way. Note the dihedral break inside region A "
        "itself, at roughly &plusmn;1.3 m.",
    ),
    (
        "plan_l_resampling.png",
        "Plan_L, resampled for the VLM",
        "Native mesh in grey, the 9 x 35 VLM mesh in red. 728 panels per half become 272. The winglet close-up "
        "is where the re-loft has to work: arc-length parameterization keeps the knee from being smeared.",
    ),
    (
        "const_chord_resampling.png",
        "ConstChord, resampled for the VLM",
        "The harder case: 2632 panels per half down to 272, a 10x cut. The convergence panel is the evidence "
        "for N_CHORDWISE = 9 &mdash; the knee of the curve, before the returns flatten out.",
    ),
]

SPECS = [
    ("Spanwise sections (per half)", "27", "95"),
    ("Camber nodes per section", "29", "29"),
    ("Half-span", "694.111 in / 17.630 m", "708.000 in / 17.983 m"),
    ("Root chord", "138.90 in / 3.53 m", "105.00 in / 2.67 m"),
    ("Tip chord", "20.00 in / 0.51 m", "14.00 in / 0.36 m"),
    ("Thickness ratio, root &rarr; tip", "0.177 &rarr; 0.177", "0.178 &rarr; 0.100"),
    ("Twist, root &rarr; tip", "2.117&deg; &rarr; 0.133&deg;", "4.000&deg; &rarr; -0.926&deg;"),
    ("Region A ends at", "section 2, y = 50.0 in", "section 51, y = 361.7 in"),
    ("Winglet starts at", "section 22, y = 661.7 in", "section 90, y = 674.9 in"),
    ("Region B leading-edge sweep", "4.868&deg;", "8.023&deg;"),
    ("Region B dihedral", "4.390&deg;", "4.390&deg;"),
]

ERRORS = [
    ("5", "136", "0.323", "0.090", "1.154", "0.274"),
    ("7", "204", "0.104", "0.026", "0.532", "0.124"),
    ("9", "272", "0.042", "0.014", "0.272", "0.058"),
    ("13", "408", "0.036", "0.010", "0.138", "0.019"),
]


def spec_rows():
    return "\n".join(
        f"          <tr><th scope=\"row\">{label}</th><td>{a}</td><td>{b}</td></tr>" for label, a, b in SPECS
    )


def error_rows():
    out = []
    for nx, panels, pmax, prms, cmax, crms in ERRORS:
        mark = ' class="picked"' if nx == "9" else ""
        label = f"{nx} &nbsp;<span class=\"tag\">chosen</span>" if nx == "9" else nx
        out.append(
            f"          <tr{mark}><th scope=\"row\">{label}</th><td>{panels}</td>"
            f"<td>{pmax}</td><td>{prms}</td><td>{cmax}</td><td>{crms}</td></tr>"
        )
    return "\n".join(out)


def figure_blocks():
    blocks = []
    for filename, title, caption in FIGURES:
        blocks.append(
            f"""      <figure class="plate">
        <div class="paper"><img src="{data_uri(filename)}" alt="{title}" /></div>
        <figcaption><b>{title}.</b> {caption}</figcaption>
      </figure>"""
        )
    return "\n".join(blocks)


HTML = f"""<title>Plan_L Wing Meshes</title>
<style>
  :root {{
    --ground: #f2f3f5;
    --panel: #ffffff;
    --paper: #fcfcfd;
    --ink: #171b21;
    --ink-soft: #4d5763;
    --ink-faint: #78828f;
    --rule: #d6dae0;
    --rule-soft: #e6e9ed;
    --region-a: #4c72b0;
    --region-b: #c9713f;
    --region-c: #3f8a5c;
    --accent: #2f5d99;
  }}

  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ground: #14171c;
      --panel: #1c2027;
      --paper: #f3f4f6;
      --ink: #e6e9ee;
      --ink-soft: #a8b1bd;
      --ink-faint: #79838f;
      --rule: #2e343d;
      --rule-soft: #252b33;
      --region-a: #7fa3d8;
      --region-b: #e2946a;
      --region-c: #6fb98c;
      --accent: #8fb2e2;
    }}
  }}

  :root[data-theme="dark"] {{
    --ground: #14171c;
    --panel: #1c2027;
    --paper: #f3f4f6;
    --ink: #e6e9ee;
    --ink-soft: #a8b1bd;
    --ink-faint: #79838f;
    --rule: #2e343d;
    --rule-soft: #252b33;
    --region-a: #7fa3d8;
    --region-b: #e2946a;
    --region-c: #6fb98c;
    --accent: #8fb2e2;
  }}

  * {{ box-sizing: border-box; }}

  body {{
    margin: 0;
    background: var(--ground);
    color: var(--ink);
    font-family: ui-sans-serif, system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }}

  .wrap {{
    max-width: 1180px;
    margin: 0 auto;
    padding: clamp(28px, 5vw, 64px) clamp(18px, 4vw, 44px) 96px;
    display: flex;
    flex-direction: column;
    gap: 48px;
  }}

  .mono {{
    font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  }}

  header .eyebrow {{
    font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin: 0 0 14px;
  }}

  header h1 {{
    margin: 0;
    font-size: clamp(30px, 4.6vw, 46px);
    line-height: 1.1;
    font-weight: 620;
    letter-spacing: -0.02em;
    text-wrap: balance;
  }}

  header p.lede {{
    margin: 16px 0 0;
    max-width: 66ch;
    font-size: 17px;
    color: var(--ink-soft);
  }}

  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px 22px;
    margin-top: 26px;
    padding-top: 22px;
    border-top: 1px solid var(--rule);
  }}

  .legend span {{
    display: inline-flex;
    align-items: center;
    gap: 9px;
    font-size: 13.5px;
    color: var(--ink-soft);
  }}

  .legend i {{
    width: 22px;
    height: 3px;
    border-radius: 2px;
    background: currentColor;
  }}

  .swatch-a {{ color: var(--region-a); }}
  .swatch-b {{ color: var(--region-b); }}
  .swatch-c {{ color: var(--region-c); }}

  section {{ display: flex; flex-direction: column; gap: 18px; }}

  h2 {{
    margin: 0;
    font-size: 13px;
    font-weight: 640;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--ink-faint);
    font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  }}

  .scroller {{ overflow-x: auto; }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14.5px;
    font-variant-numeric: tabular-nums;
    background: var(--panel);
  }}

  caption {{
    caption-side: bottom;
    text-align: left;
    padding-top: 12px;
    font-size: 13.5px;
    color: var(--ink-faint);
  }}

  th, td {{
    text-align: left;
    padding: 10px 16px;
    border-bottom: 1px solid var(--rule-soft);
    white-space: nowrap;
  }}

  thead th {{
    border-bottom: 1px solid var(--rule);
    font-weight: 620;
    font-size: 12.5px;
    letter-spacing: 0.04em;
    color: var(--ink-soft);
  }}

  tbody th {{
    font-weight: 480;
    color: var(--ink-soft);
    white-space: normal;
  }}

  tbody td {{
    font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 13.5px;
  }}

  tbody tr:last-child th, tbody tr:last-child td {{ border-bottom: 0; }}

  tr.picked th, tr.picked td {{ background: color-mix(in srgb, var(--accent) 11%, transparent); }}

  .tag {{
    font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 10.5px;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--accent);
    border: 1px solid color-mix(in srgb, var(--accent) 45%, transparent);
    border-radius: 3px;
    padding: 1px 6px;
    white-space: nowrap;
  }}

  .plates {{ display: flex; flex-direction: column; gap: 40px; }}

  .plate {{ margin: 0; display: flex; flex-direction: column; gap: 14px; }}

  /* The figures are drawn on white; keep them on paper in both themes rather
     than letting a dark ground cut a bright rectangle out of the page. */
  .paper {{
    background: var(--paper);
    border: 1px solid var(--rule);
    padding: 10px;
    overflow-x: auto;
  }}

  .paper img {{ display: block; width: 100%; height: auto; min-width: 620px; }}

  figcaption {{
    font-size: 14.5px;
    color: var(--ink-soft);
    max-width: 78ch;
  }}

  figcaption b {{ color: var(--ink); font-weight: 620; }}

  .notes {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
    gap: 1px;
    background: var(--rule-soft);
    border: 1px solid var(--rule);
  }}

  .note {{ background: var(--panel); padding: 20px 22px; }}

  .note h3 {{
    margin: 0 0 8px;
    font-size: 14px;
    font-weight: 640;
    letter-spacing: 0.01em;
  }}

  .note p {{ margin: 0; font-size: 14px; color: var(--ink-soft); }}

  code {{
    font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.92em;
    color: var(--accent);
  }}

  footer {{
    border-top: 1px solid var(--rule);
    padding-top: 20px;
    font-size: 13px;
    color: var(--ink-faint);
  }}

  a {{ color: var(--accent); }}
  a:focus-visible, .paper:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 3px; }}
</style>

<div class="wrap">
  <header>
    <p class="eyebrow">OpenVSP DegenGeom &rarr; OpenAeroStruct</p>
    <h1>Plan_L wing meshes</h1>
    <p class="lede">
      Two baseline wings, read straight out of their OpenVSP DegenGeom CSV exports and rebuilt as
      OpenAeroStruct camber-surface meshes &mdash; no OpenVSP install in the loop. Each wing is a
      constant-chord bay, a straight tapered panel, and a winglet; the region breakpoints below were
      detected from the geometry, not hardcoded.
    </p>
    <div class="legend mono">
      <span class="swatch-a"><i></i>Region A &mdash; constant chord</span>
      <span class="swatch-b"><i></i>Region B &mdash; taper</span>
      <span class="swatch-c"><i></i>Region C &mdash; winglet</span>
    </div>
  </header>

  <section>
    <h2>Baselines</h2>
    <div class="scroller">
      <table>
        <thead>
          <tr><th scope="col">&nbsp;</th><th scope="col">Plan_L</th><th scope="col">ConstChord</th></tr>
        </thead>
        <tbody>
{spec_rows()}
        </tbody>
        <caption>Section indices are native DegenStick indices for the surf_index 0 half, root to tip.</caption>
      </table>
    </div>
  </section>

  <section>
    <h2>Renders</h2>
    <div class="plates">
{figure_blocks()}
    </div>
  </section>

  <section>
    <h2>Choosing the chordwise count</h2>
    <div class="scroller">
      <table>
        <thead>
          <tr>
            <th scope="col">Nodes</th><th scope="col">Panels / half</th>
            <th scope="col">Plan_L max</th><th scope="col">Plan_L rms</th>
            <th scope="col">ConstChord max</th><th scope="col">ConstChord rms</th>
          </tr>
        </thead>
        <tbody>
{error_rows()}
        </tbody>
        <caption>
          Camber-surface error against the native 29-node sections, as a percentage of local chord,
          at 35 spanwise stations. ConstChord is the binding case.
        </caption>
      </table>
    </div>
  </section>

  <section>
    <h2>How these were built</h2>
    <div class="notes">
      <div class="note">
        <h3>Camber surface</h3>
        <p>
          <code>x + nCamber_x &middot; zCamber</code> and its y and z counterparts, with the same
          <code>flip_normal</code> handling and left/right join that OpenAeroStruct's own VSP path uses, so a
          mesh built from CSV is interchangeable with one built through the live API.
        </p>
      </div>
      <div class="note">
        <h3>Spanwise re-loft</h3>
        <p>
          PCHIP in cumulative arc length along the leading edge, not in y. The winglet turns through about
          45&deg; of dihedral, where y stops tracking real distance; arc length does not. Stations are cosine
          clustered toward the tip, with 20% of them spent inside the winglet.
        </p>
      </div>
      <div class="note">
        <h3>Chordwise re-loft</h3>
        <p>
          PCHIP in normalized camber-line arc length, cosine clustered toward both edges &mdash; the leading
          edge for the suction peak, the trailing edge for the Kutta condition. Leading- and trailing-edge
          nodes are carried over exactly rather than smoothed.
        </p>
      </div>
    </div>
  </section>

  <footer>
    Rendered from <span class="mono">Plan_L_DegenGeom.csv</span> and
    <span class="mono">Plan_L_ConstChord_DegenGeom.csv</span> via
    <span class="mono">studies/vsp_planform/mesh.py</span>. Dimensions are native inches where marked;
    all mesh coordinates are metres.
  </footer>
</div>
"""

with open(os.path.join(HERE, "wing_renders.html"), "w") as f:
    f.write(HTML)

print(os.path.join(HERE, "wing_renders.html"), os.path.getsize(os.path.join(HERE, "wing_renders.html")))
