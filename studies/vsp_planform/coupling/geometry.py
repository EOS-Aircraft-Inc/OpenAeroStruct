"""Write the OpenVSP station export that WingCalc's geometry provider reads.

WingCalc takes its planform from an "airfoil metadata CSV" plus one Selig ``.dat``
contour per station (``io/openvsp.py``: ``read_openvsp_geometry``). Each metadata
block needs only four keys -- ``Airfoil File Name``, ``Leading Edge Point``,
``Trailing Edge Point``, ``Chord`` -- and ``ws`` is taken from the LE's y.

Section *shape* cannot come from OAS: its mesh is the camber surface and it
reduces a section to t/c and c_max_t. So shape comes from the baseline
DegenGeom's plate (``zCamber`` and ``t``, normalized by chord) and is rescaled to
the OAS ``t_over_c`` at that station. That is exactly OAS's own assumption --
t/c is not a design variable in this study -- so nothing is invented here.

Stations outboard of the winglet junction are dropped: ``ws`` is the LE y, which
stops increasing through the winglet, and a non-monotonic ws would corrupt the
provider's spanwise interpolation.
"""

import numpy as np

SCALE = 0.0254  # m per inch


def database_profile(name, x_grid):
    """Camber and thickness of a database section, normalized by chord, on x_grid.

    Split rather than returned as upper/lower because the two are scaled
    differently: the thickness carries the design's t/c, the camber does not.
    """
    import aerosandbox as asb
    af = asb.Airfoil(name)
    t = np.array([float(af.local_thickness(x_over_c=float(x))) for x in x_grid])
    cam = np.array([float(af.local_camber(x_over_c=float(x))) for x in x_grid])
    return cam, t


def blended_profile(blend, x_grid, semi_in=708.0):
    """Camber and thickness of a SPANWISE section pair, as a function of y (inches)."""
    cam_i, t_i = database_profile(blend["inboard"], x_grid)
    cam_o, t_o = database_profile(blend["outboard"], x_grid)
    f0, f1 = float(blend["f_start"]), float(blend["f_end"])

    def at(y_in):
        w = float(np.clip((abs(y_in) / semi_in - f0) / (f1 - f0), 0.0, 1.0))
        return (1.0 - w) * cam_i + w * cam_o, (1.0 - w) * t_i + w * t_o

    return at


def write_spanload(path, name, y_in, width_in, lift_lb_per_in, note=""):
    """Write the ``*_lift_mtow.csv`` companion: the 1 g spanload at MTOW.

    WingCalc reads ``y_in``, ``width_in`` and ``lift_lb_per_in`` and uses only the
    SHAPE, rescaling it to each load case's ``Nz_lift * AC_Weight``. Without this
    file it falls back to an elliptical spanload measured from the fuselage side --
    a different load on the same wing, and silently so.
    """
    y = np.asarray(y_in, dtype=float)
    w = np.asarray(width_in, dtype=float)
    q = np.asarray(lift_lb_per_in, dtype=float)
    with path.open("w", encoding="utf-8") as fh:
        say = lambda t: print(t, file=fh)
        say(f"# {name} -- spanwise lift distribution, 1 g, TRIMMED, at MTOW.")
        say("# Written by studies/vsp_planform/coupling/geometry.py from the same")
        say("# trimmed OAS state the drag was taken from, so load and geometry agree.")
        if note:
            say(f"# {note}")
        say(f"# Integrated over the half wing: {float((q * w).sum()):.2f} lb")
        say("y_in,width_in,lift_lb_per_in")
        for yi, wi, qi in zip(y, w, q):
            say(f"{yi:.4f},{wi:.6f},{qi:.6f}")
    return path


def normalized_sections(plate, stick):
    """Baseline section shapes as (span_frac, x/c, upper/c, lower/c)."""
    n_sec = plate.num_secs
    span = stick.le[:, 1]
    frac = (span - span[0]) / (span[-1] - span[0])

    x_n, up_n, lo_n = [], [], []
    for i in range(n_sec):
        chord = float(stick.chord[i])
        x = (plate.x[i] - float(stick.le[i, 0])) / chord
        camber = plate.zCamber[i] / chord
        half_t = 0.5 * plate.t[i] / chord
        order = np.argsort(x)
        x_n.append(x[order])
        up_n.append((camber + half_t)[order])
        lo_n.append((camber - half_t)[order])
    return frac, x_n, up_n, lo_n


def _resample(x_src, y_src, x_dst):
    return np.interp(x_dst, x_src, y_src)


def section_at(frac_target, frac, x_n, up_n, lo_n, x_grid):
    """Blend the two bracketing baseline sections onto a common x/c grid."""
    j = int(np.clip(np.searchsorted(frac, frac_target) - 1, 0, len(frac) - 2))
    f0, f1 = frac[j], frac[j + 1]
    w = 0.0 if f1 == f0 else (frac_target - f0) / (f1 - f0)

    up = (1 - w) * _resample(x_n[j], up_n[j], x_grid) + w * _resample(x_n[j + 1], up_n[j + 1], x_grid)
    lo = (1 - w) * _resample(x_n[j], lo_n[j], x_grid) + w * _resample(x_n[j + 1], lo_n[j + 1], x_grid)
    return up, lo


def write_dat(path, x_grid, upper, lower, header):
    """Selig order: TE along the upper surface to LE, then lower back to TE."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for x, y in zip(x_grid[::-1], upper[::-1]):
            fh.write(f"{x:.10f} {y:.10f}\n")
        for x, y in zip(x_grid[1:], lower[1:]):
            fh.write(f"{x:.10f} {y:.10f}\n")


def export(mesh_m, toc_panel, plate, stick, out_dir, name="OAS_export", max_ws_in=None, n_x=201,
           front_pct=None, rear_schedule=None, airfoil=None, spanload=None):
    """Write ``out_dir/<name>.csv`` and one ``.dat`` per station.

    mesh_m     : OAS mesh, (nx, ny, 3), metres
    toc_panel  : t/c per spanwise *panel* from OAS (len ny-1) or per node (len ny)
    plate/stick: baseline DegenGeom, for section shape
    max_ws_in  : drop stations beyond this ws (the winglet junction)
    front_pct  : box front edge, fraction of chord. With ``rear_schedule``,
                 also writes ``<name>_spar.csv``.
    rear_schedule : ``((y_in, x/c), ...)`` breakpoints for the rear spar.
    airfoil    : the section the design is actually built on -- a database name, or
                 ``{"inboard", "outboard", "f_start", "f_end"}`` for a spanwise blend.
                 WITHOUT IT THE BASELINE SECTION IS SUBSTITUTED, which is not a
                 cosmetic difference: measured near the aileron the baseline peaks at
                 x/c 0.317 and keeps 0.56 of its thickness at a 0.74c spar, where the
                 e694/goe16k blend peaks at 0.499 and keeps 0.82. Depth is
                 ``retention * t/c * chord``, so sizing on the wrong one understates
                 the box by ~30% while every reported number still says 7.00 in.
    spanload   : ``(y_in, width_in, lift_lb_per_in)`` for the ``*_lift_mtow.csv``
                 companion. Omitted, WingCalc falls back to an elliptical spanload.

    THE SPAR FILE IS NOT OPTIONAL ON A V3.6 DECK. Those decks carry no
    'Aft spar chord ratio' row in planformIn.csv, so WingCalc resolves the
    box from ``OpenVSP/*_spar.csv`` first and falls back to
    ``AlternativeInputs/sparRatios.csv``. The shipped fallback is a DIFFERENT
    construction (Arc A's offset spar, 0.750c rising to 0.8044c), so a deck
    exported without a spar file is silently sized on the wrong box.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    le = mesh_m[0] / SCALE      # (ny, 3) inches
    te = mesh_m[-1] / SCALE
    ny = le.shape[0]

    # OAS meshes run tip -> root in y; make it root -> tip.
    if le[0, 1] > le[-1, 1]:
        le, te = le[::-1], te[::-1]
        toc_panel = np.asarray(toc_panel)[::-1]

    toc = np.asarray(toc_panel, dtype=float)
    toc_node = toc if len(toc) == ny else np.interp(
        np.arange(ny), np.arange(len(toc)) + 0.5, toc)

    chord = np.linalg.norm(te - le, axis=1)
    ws = le[:, 1]

    keep = np.ones(ny, dtype=bool)
    if max_ws_in is not None:
        keep = ws <= max_ws_in + 1e-6
        nxt = np.flatnonzero(~keep)
        if nxt.size:                       # one bracketing station past the cut
            keep[nxt[0]] = True
    # ws must be strictly increasing for the provider's interpolation
    keep &= np.r_[True, np.diff(ws) > 1e-6]

    frac, x_n, up_n, lo_n = normalized_sections(plate, stick)
    x_grid = 0.5 * (1 - np.cos(np.linspace(0.0, np.pi, n_x)))   # cosine, LE-clustered

    prof_at = None
    if airfoil is not None:
        if isinstance(airfoil, str):
            _cam, _t = database_profile(airfoil, x_grid)
            prof_at = lambda _y: (_cam, _t)          # noqa: E731 -- one section, no y
        else:
            prof_at = blended_profile(airfoil, x_grid)

    span0, span1 = ws[keep][0], ws[keep][-1]
    blocks = []
    for idx in np.flatnonzero(keep):
        if prof_at is None:
            f = 0.0 if span1 == span0 else (ws[idx] - span0) / (span1 - span0)
            up, lo = section_at(f, frac, x_n, up_n, lo_n, x_grid)
            t_now = float(np.max(up - lo))
            if t_now > 1e-9:
                scale = float(toc_node[idx]) / t_now
                up, lo = up * scale, lo * scale
        else:
            # Thickness carries the design's t/c; camber does not. Scaling the two
            # together (which the baseline branch above does) would change the
            # camber line as a side effect of a thickness change.
            cam, thk = prof_at(ws[idx])
            t_now = float(np.max(thk))
            if t_now > 1e-9:
                thk = thk * (float(toc_node[idx]) / t_now)
            up, lo = cam + 0.5 * thk, cam - 0.5 * thk
        dat = f"{name}_{idx}.dat"
        write_dat(out_dir / dat, x_grid, up, lo, f"# {name} station {idx}, ws {ws[idx]:.4f} in")
        blocks.append((dat, idx, le[idx], te[idx], chord[idx]))

    csv_path = out_dir / f"{name}.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("# AIRFOIL METADATA CSV FILE\n\n")
        fh.write("Airfoil File Directory, ./\n\n")
        for dat, idx, lep, tep, c in blocks:
            fh.write("#" * 40 + "\n")
            fh.write(f"Airfoil File Name, {dat}\n")
            fh.write(f"Geom Name, {name}\n")
            fh.write("Geom ID, OASEXPORT\n")
            fh.write(f"Airfoil Index, {idx}\n")
            fh.write(f"Leading Edge Point, {lep[0]:.6f}, {lep[1]:.6f}, {lep[2]:.6f}\n")
            fh.write(f"Trailing Edge Point, {tep[0]:.6f}, {tep[1]:.6f}, {tep[2]:.6f}\n")
            fh.write(f"Chord, {c:.6f}\n")
            fh.write("#" * 40 + "\n\n")
    lift_path = None
    if spanload is not None:
        lift_path = write_spanload(out_dir / f"{name}_lift_mtow.csv", name, *spanload)

    spar_path = None
    if front_pct is not None and rear_schedule is not None:
        spar_path = _write_spar(out_dir / f"{name}_spar.csv", name,
                                [b[1] for b in blocks], ws, chord,
                                float(front_pct), rear_schedule)
    return csv_path, len(blocks), spar_path, lift_path


def _write_spar(path, name, idxs, ws, chord, front_pct, rear_schedule):
    """Write the ``*_spar.csv`` companion WingCalc reads for the box.

    Only ``y_in``, ``front_pct`` and ``rear_pct`` are read back
    (``io/openvsp.py:find_openvsp_spar_ratios``); the rest is provenance for a
    human. The tool then requires, in ``io/inputs.py:_validate_spar_ratios``:
    strictly increasing y, ``0 < fwd < aft < 1``, and -- because both spars are
    frozen straight inside the fuselage -- every row inboard of W2F BL carrying
    the value interpolated AT W2F. ``rear_spar_fraction`` holds its end values
    flat outside the breakpoints, so an inboard-most breakpoint at or outboard of
    W2F satisfies the last rule by construction.
    """
    from studies.vsp_planform.param import rear_spar_fraction

    y = np.asarray(ws, dtype=float)[idxs]
    c = np.asarray(chord, dtype=float)[idxs]
    rear = np.asarray(rear_spar_fraction(y, rear_schedule), dtype=float)

    bad = np.flatnonzero(~((0.0 < front_pct) & (front_pct < rear) & (rear < 1.0)))
    if bad.size:
        raise ValueError(
            f"{path.name}: front {front_pct} and rear {rear[bad[0]]} at y "
            f"{y[bad[0]]:.2f} in do not satisfy 0 < fwd < aft < 1, which WingCalc "
            f"rejects. Check the rear-spar schedule.")

    with path.open("w", encoding="utf-8") as fh:
        say = lambda t: print(t, file=fh)
        say(f"# {name} -- wingbox spar chord fractions, one row per exported station.")
        say("# Written by studies/vsp_planform/coupling/geometry.py from the OAS design")
        say("# that was optimized, so the box sized here is the box OAS was constrained on.")
        sched = tuple((float(a), float(b)) for a, b in rear_schedule)
        say(f"# Front spar held at {front_pct:.4f}c. Rear spar schedule (y_in, x/c): {sched}")
        say("# box_width_in is (rear_pct - front_pct) * chord_in; provenance only, not read back.")
        say("y_in,chord_in,front_pct,rear_pct,box_width_in")
        for yi, ci, ri in zip(y, c, rear):
            say(f"{yi:.4f},{ci:.6f},{front_pct:.4f},{ri:.4f},{(ri - front_pct) * ci:.4f}")
    return path
