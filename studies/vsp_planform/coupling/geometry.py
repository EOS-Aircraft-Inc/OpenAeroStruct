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


def export(mesh_m, toc_panel, plate, stick, out_dir, name="OAS_export", max_ws_in=None, n_x=201):
    """Write ``out_dir/<name>.csv`` and one ``.dat`` per station.

    mesh_m     : OAS mesh, (nx, ny, 3), metres
    toc_panel  : t/c per spanwise *panel* from OAS (len ny-1) or per node (len ny)
    plate/stick: baseline DegenGeom, for section shape
    max_ws_in  : drop stations beyond this ws (the winglet junction)
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

    span0, span1 = ws[keep][0], ws[keep][-1]
    blocks = []
    for idx in np.flatnonzero(keep):
        f = 0.0 if span1 == span0 else (ws[idx] - span0) / (span1 - span0)
        up, lo = section_at(f, frac, x_n, up_n, lo_n, x_grid)
        t_now = float(np.max(up - lo))
        if t_now > 1e-9:
            scale = float(toc_node[idx]) / t_now
            up, lo = up * scale, lo * scale
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
    return csv_path, len(blocks)
