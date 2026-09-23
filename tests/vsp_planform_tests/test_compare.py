"""The side-by-side page is one template and one string format, so what breaks it
is the template, not the physics.

``compare._PAGE`` carries a stylesheet and a script inside a ``str.format``
template, which means every CSS and JS brace in it is doubled. Miss one and
``format`` either raises or -- far worse -- silently substitutes something into
the middle of the JavaScript and writes a page that opens to a blank frame. The
tests here are therefore about the OUTPUT text: no leftover doubled braces, the
frame sources resolve to files that exist, and the JS quoting survived.

They also pin the two behaviours that are easy to regress into a lie: a missing
report must produce a stated placeholder rather than an empty frame, and the tab
bar must start hidden, because whether it can drive both panes is the browser's
decision at load time and not something this page may assume.
"""

import re
import unittest
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory

from studies.vsp_planform.viewer import compare


class _WellFormed(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
            "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack:
            self.errors.append(f"</{tag}> with nothing open")
        elif self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes <{self.stack[-1]}>")
        else:
            self.stack.pop()


class _Pair:
    """A folder holding two stand-in reports, laid out as a real run leaves them."""

    def __enter__(self):
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.wc = self.dir / "Wing_Reportdeck_arcA_optimal_e694.html"
        self.oas = self.dir / "OAS_Wing_Report.html"
        for p in (self.wc, self.oas):
            p.write_text("<!DOCTYPE html><html><body>x</body></html>", encoding="utf-8")
        self.out = self.dir / "Compare.html"
        return self

    def __exit__(self, *exc):
        self._tmp.cleanup()


class TestComparePage(unittest.TestCase):
    def _write(self, pair, **kw):
        compare.write_compare_page(pair.out, wingcalc_html=pair.wc,
                                   oas_html=pair.oas, **kw)
        return pair.out.read_text(encoding="utf-8")

    def test_no_leftover_doubled_braces(self):
        """A doubled brace in the output is a brace `format` never unescaped."""
        with _Pair() as p:
            html = self._write(p)
        self.assertNotIn("{{", html)
        self.assertNotIn("}}", html)

    def test_no_unsubstituted_placeholders(self):
        """`{name}` surviving into the page means a field the template forgot."""
        with _Pair() as p:
            html = self._write(p)
        # Real CSS/JS braces are followed by a newline, a space or a quote; a live
        # placeholder is a bare identifier wrapped in braces.
        self.assertEqual([], re.findall(r"\{[a-z_]+\}", html))

    def test_html_is_well_formed(self):
        with _Pair() as p:
            html = self._write(p)
        w = _WellFormed()
        w.feed(html)
        self.assertEqual([], w.errors)
        self.assertEqual([], w.stack)

    def test_frame_sources_resolve_to_real_files(self):
        """The hrefs are relative, so they are only right relative to the page."""
        with _Pair() as p:
            html = self._write(p)
            srcs = re.findall(r'<iframe id="[^"]+" src="([^"]+)"', html)
            self.assertEqual(2, len(srcs))
            for src in srcs:
                self.assertNotIn("\\", src, "a backslash is not a path separator to a browser")
                self.assertTrue((p.out.parent / src).is_file(), f"{src} does not exist")

    def test_frame_ids_match_the_frames_present(self):
        """FRAME_IDS drives every sync call; an id with no frame breaks them all."""
        with _Pair() as p:
            html = self._write(p)
        declared = re.search(r"var FRAME_IDS = (\[[^\]]*\]);", html).group(1)
        present = re.findall(r'<iframe id="([^"]+)"', html)
        for fid in present:
            self.assertIn(f'"{fid}"', declared)
        self.assertEqual(len(present), declared.count('"') // 2)

    def test_missing_report_gives_a_placeholder_not_an_empty_frame(self):
        with _Pair() as p:
            compare.write_compare_page(p.out, wingcalc_html=None, oas_html=p.oas)
            html = p.out.read_text(encoding="utf-8")
        self.assertEqual(1, html.count("<iframe"))
        self.assertIn("No WingCalc report", html)
        self.assertIn('var FRAME_IDS = ["frame-oas"]', html)

    def test_tab_bar_starts_hidden(self):
        """Whether one bar can drive both panes is decided at load, by the browser.

        Shipping it visible would show a control that does nothing on the most
        likely setup -- a file:// open in Chrome, where each report is its own
        opaque origin.
        """
        with _Pair() as p:
            html = self._write(p)
        self.assertIn('id="tab-bar" style="display:none"', html)
        self.assertIn("syncAvailable()", html)

    def test_offers_only_tabs_both_reports_have(self):
        """Optimization exists in WingCalc's report and not in OAS's."""
        with _Pair() as p:
            html = self._write(p)
        keys = re.findall(r'class="tab-btn[^"]*" data-tab="([a-z]+)"', html)
        self.assertEqual([k for k, _ in compare.COMMON_TABS], keys)
        self.assertNotIn("optimization", keys)

    def test_subtitles_and_heading_are_escaped(self):
        with _Pair() as p:
            html = self._write(p, heading="arc <A> & co",
                               subtitle_lines=["box <3,370> lb"])
        self.assertIn("arc &lt;A&gt; &amp; co", html)
        self.assertIn("box &lt;3,370&gt; lb", html)
        self.assertNotIn("<A>", html)

    def test_javascript_string_quoting_survived(self):
        """The sync note contains an apostrophe inside a single-quoted JS string."""
        with _Pair() as p:
            html = self._write(p)
        script = html.split("<script>")[-1].split("</script>")[0]
        self.assertIn(r"pane\'s own tabs", script)
        # An unescaped apostrophe would leave an odd number of quotes on its line.
        for line in script.splitlines():
            bare = re.sub(r"\\'", "", line)
            self.assertEqual(0, bare.count("'") % 2, f"unbalanced quote: {line}")


class TestPairShim(unittest.TestCase):
    """``pair`` restates an arc run in the vocabulary the snapshot already reads."""

    RESULT = {
        "arc": "A", "profile": "optimal", "airfoil": "e694",
        "twist_cp": [-0.36, 2.44, 0.10, 5.95, -0.07],
        "alpha": 0.7386, "success": True, "w_wing_lb": 6816.41,
    }

    def test_shim_carries_every_key_the_snapshot_reads(self):
        from studies.vsp_planform.viewer import pair

        case = pair._shim_case(self.RESULT, "arc A / optimal / e694")
        for key in ("label", "twist_lower_deg", "twist_cp_deg", "alpha_deg",
                    "objective", "success"):
            self.assertIn(key, case)
        self.assertEqual(self.RESULT["twist_cp"], case["twist_cp_deg"])
        self.assertAlmostEqual(self.RESULT["alpha"], case["alpha_deg"])

    def test_shim_does_not_claim_a_sized_box(self):
        """arc_optimal_toc sizes in WingCalc, so there is no OAS box yet.

        Zeros here would read as a sized box of zero thickness rather than as an
        absent one, and the snapshot would skip the sizing stage it needs to run.
        """
        from studies.vsp_planform.viewer import pair

        case = pair._shim_case(self.RESULT, "x")
        self.assertNotIn("skin_cp_m", case)
        self.assertNotIn("spar_cp_m", case)

    def test_objective_is_drag(self):
        """Whatever else changes, an arc_optimal_toc point is a drag optimum."""
        from studies.vsp_planform.viewer import pair

        self.assertEqual("drag", pair._shim_case(self.RESULT, "x")["objective"])


class TestThicknessCacheKey(unittest.TestCase):
    """The sizing cache must miss when the design moves.

    ``pair`` writes its shim under one fixed name holding one point, so a key of
    "file name + index" is the same string on every run. A re-run whose twist or
    alpha had moved would load the previous run's box and draw new aero over old
    structure -- a report that looks right and describes two different wings.
    """

    CASE = {"label": "x", "twist_lower_deg": -1.0, "objective": "drag",
            "twist_cp_deg": [-0.36, 2.44, 0.10, 5.95, -0.07], "alpha_deg": 0.7386}

    def _seed(self, d, taper=0.4644):
        p = Path(d) / "seed.json"
        p.write_text(f'{{"taper_B": {taper}, "t_over_c_cp": [0.25, 0.145]}}',
                     encoding="utf-8")
        return p

    def test_same_design_hits(self):
        from studies.vsp_planform.viewer import oas_snapshot as osnap

        with TemporaryDirectory() as d:
            seed = self._seed(d)
            self.assertEqual(osnap._design_digest(self.CASE, seed),
                             osnap._design_digest(dict(self.CASE), seed))

    def test_moved_twist_misses(self):
        from studies.vsp_planform.viewer import oas_snapshot as osnap

        with TemporaryDirectory() as d:
            seed = self._seed(d)
            moved = dict(self.CASE, twist_cp_deg=[-0.36, 2.44, 0.10, 5.95, -0.08])
            self.assertNotEqual(osnap._design_digest(self.CASE, seed),
                                osnap._design_digest(moved, seed))

    def test_moved_alpha_misses(self):
        from studies.vsp_planform.viewer import oas_snapshot as osnap

        with TemporaryDirectory() as d:
            seed = self._seed(d)
            moved = dict(self.CASE, alpha_deg=0.7387)
            self.assertNotEqual(osnap._design_digest(self.CASE, seed),
                                osnap._design_digest(moved, seed))

    def test_moved_frozen_geometry_misses(self):
        """The seed carries taper and t/c, which the box is sized on."""
        from studies.vsp_planform.viewer import oas_snapshot as osnap

        with TemporaryDirectory() as d:
            a = osnap._design_digest(self.CASE, self._seed(d, taper=0.4644))
            b = osnap._design_digest(self.CASE, self._seed(d, taper=0.4700))
            self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
