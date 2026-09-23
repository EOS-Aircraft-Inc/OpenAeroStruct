# Vendored third-party assets

## plotly-basic.min.js

| | |
|---|---|
| Package | `plotly.js` (basic build, minified) |
| Version | 2.35.2 |
| Source | https://cdn.plot.ly/plotly-basic-2.35.2.min.js |
| SHA-256 | `138c2e81014b979dc00867a93da55b7605a17495ee78dd7afb433b7f021dfcfa` |
| License | MIT (Copyright 2012-2024, Plotly, Inc.) — header retained at the top of the file |

Both HTML reports inline this file verbatim rather than linking a CDN copy. A linked
copy leaves the report blank, with no message, on any machine with no route out to the
internet, and makes an archived report depend on plot.ly still serving that one URL
years from now — which a report kept as a design record cannot rely on. The `basic`
build holds scatter, bar and pie; scatter and bar are every trace the reports draw.

The digest above is pinned as `_PLOTLY_SHA256` in `viewer/wing_viewer.py` and checked
every time a report is written, so a swapped, patched or truncated bundle stops the run
instead of riding along into an engineering record.

### Upgrading

1. Download the new bundle from the URL above with the version changed.
2. `sha256sum viewer/vendor/plotly-basic.min.js`
3. Update `_PLOTLY_SHA256` in `viewer/wing_viewer.py` and the version and digest here.
4. Run `python main.py run` and open a report — the version bump is not verified by the
   digest check, only that the file is the one that was pinned.

Commit the bundle and the digest in the same commit: a digest without the file it
describes cannot be checked, and a file without its digest is not verified.
