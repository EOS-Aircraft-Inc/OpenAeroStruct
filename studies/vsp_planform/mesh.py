"""Camber-surface meshes from DegenGeom data, and spanwise/chordwise re-lofting.

The mesh construction reproduces ``generate_vsp_surfaces``
(``openaerostruct/geometry/utils.py:504-537``) exactly, so a mesh built here
from a CSV export is interchangeable with one built through the live OpenVSP
API: same camber reconstruction, same ``flip_normal`` handling, same chordwise
and spanwise ordering, same left/right join.

Mesh convention (see ``tests/geometry_tests/test_vsp_mesh.py``)::

    mesh[0, :, :]   leading edge
    mesh[-1, :, :]  trailing edge
    mesh[:, 0, :]   most negative y (left tip, or the root for a half mesh)
    mesh[:, -1, :]  most positive y (right tip)

A note on ``nx``
----------------
OAS computes ``nx = (plate.num_pnts + 1) // 2`` because the OpenVSP API reports
the *surface* point count on the plate object (57 here) while storing only the
(57+1)/2 = 29 camber points. The DegenGeom CSV instead reports the camber point
count directly in its ``PLATE,<nXsecs>,<nPnts>`` header, so ``num_pnts`` is
already 29 and ``nx = plate.num_pnts``. Halving it again would silently throw
away the aft half of every airfoil.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

from studies.vsp_planform import config
from studies.vsp_planform.degen_csv import lifting_surfaces, read_degen_csv

# Chordwise nodes to keep for the VLM. The DegenGeom plates carry 29, which is
# far more than a VLM needs and costs quadratically: the aerodynamic influence
# matrix goes as the square of the panel count, so 29 nodes (28 panels) at 35
# spanwise stations is 952 panels per half against 272 at 9 nodes, a factor 12
# in AIC work.
#
# Round-trip camber error against the native 29-node sections, as a fraction of
# local chord (worst node / rms over all nodes, both baselines, 35 spanwise):
#
#     nx   Plan_L max    Plan_L rms   ConstChord max   ConstChord rms
#      5     0.32%         0.090%          1.15%           0.27%
#      7     0.10%         0.026%          0.53%           0.12%
#      9     0.042%        0.014%          0.27%           0.058%
#     13     0.036%        0.010%          0.14%           0.019%
#
# 9 is the knee: 7 -> 9 halves the ConstChord error for 1.65x the panels, while
# 9 -> 13 halves it again for 2.1x the panels and lands well below the accuracy
# of the VLM itself. The error is concentrated at mid-chord, where the camber
# crest sits and cosine clustering is deliberately sparse.
N_CHORDWISE = 9


def plate_to_mesh(component):
    """Build the camber-surface mesh of one symmetric half, in native units.

    Parameters
    ----------
    component : DegenComponent
        A LIFTING_SURFACE component holding exactly one plate.

    Returns
    -------
    ndarray
        ``(nx, ny, 3)`` with ``nx = plate.num_pnts`` chordwise nodes running
        leading edge to trailing edge, and ``ny = plate.num_secs`` spanwise
        sections running from most-negative to most-positive y.
    """
    plate = component.plate

    nx = plate.num_pnts
    ny = plate.num_secs
    mesh = np.zeros((nx, ny, 3))

    # Camber surface: openaerostruct/geometry/utils.py:509-511.
    coords = list(plate.camber_surface())

    # Make sure the mesh is ordered in the right direction: utils.py:513-521.
    # The flip is along the section axis, so it is the mirrored half (the one
    # with flip_normal False) that gets reversed and ends up running tip to root.
    if not component.flip_normal:
        coords = [np.flipud(c) for c in coords]

    for i, coord in enumerate(coords):
        mesh[:, :, i] = np.flipud(coord.T)

    return mesh


def _components(path_or_components):
    """Accept either a CSV path or an already-parsed component list."""
    if isinstance(path_or_components, (str, bytes)) or hasattr(path_or_components, "__fspath__"):
        return read_degen_csv(path_or_components)
    return list(path_or_components)


def _halves(path_or_components):
    """Return the two halves of the single lifting surface, by surf_index."""
    groups = lifting_surfaces(_components(path_or_components))
    if len(groups) != 1:
        raise ValueError(f"expected exactly one lifting surface, found {len(groups)}")
    (comps,) = groups.values()
    return {comp.surf_index: comp for comp in comps}


def half_mesh(path_or_components, scale=config.SCALE):
    """Mesh and stick of the ``surf_index == 0`` half, scaled to metres.

    Parameters
    ----------
    path_or_components : str or list
        Path to a DegenGeom CSV, or the parsed component list.
    scale : float
        Multiplies the mesh coordinates. The VSP models are in inches.

    Returns
    -------
    (ndarray, DegenStick)
        The ``(nx, ny, 3)`` mesh and the matching stick. The stick is *not*
        scaled -- it stays in native units, since region detection is scale
        invariant and the raw numbers are easier to check against VSP.
    """
    comp = _halves(path_or_components)[0]
    return plate_to_mesh(comp) * scale, comp.stick


def full_mesh(path_or_components, scale=config.SCALE):
    """Both halves joined into a single mesh, scaled to metres.

    The join is ``np.hstack((left[:, :-1, :], right))`` from
    ``openaerostruct/geometry/utils.py:536``: the shared root section is dropped
    from the left half so it is not duplicated. ``surf_index == 0`` is the right
    half, ``surf_index == 1`` the left.
    """
    halves = _halves(path_or_components)
    right = plate_to_mesh(halves[0])
    left = plate_to_mesh(halves[1])
    return np.hstack((left[:, :-1, :], right)) * scale


def spanwise_stations(y_native, n_total, y_c_start, winglet_fraction=config.WINGLET_STATION_FRACTION):
    """Target spanwise stations for re-lofting one half.

    Stations are cosine-clustered toward the tip across regions A and B, which
    puts resolution where the loading gradient is, and ``winglet_fraction`` of
    the total is spent inside the winglet, which is a small span fraction but
    carries all of the curvature.

    Parameters
    ----------
    y_native : ndarray
        Native spanwise stations, root to tip, used only for the endpoints.
    n_total : int
        Number of stations to return.
    y_c_start : float
        Spanwise station where the winglet begins; it is always a node.
    winglet_fraction : float
        Fraction of ``n_total`` placed strictly outboard of ``y_c_start``.

    Returns
    -------
    ndarray
        ``n_total`` strictly increasing stations from root to tip.
    """
    y_root = float(y_native[0])
    y_tip = float(y_native[-1])

    if not y_root < y_c_start < y_tip:
        raise ValueError(f"winglet start {y_c_start} is outside the span [{y_root}, {y_tip}]")

    n_c = min(max(round(winglet_fraction * n_total), 1), n_total - 2)
    n_ab = n_total - n_c

    # sin(pi/2 * t) has its widest spacing at the root and closes up toward the
    # tip, i.e. cosine clustering on the outboard end only. n_ab includes both
    # the root and the y_c_start node.
    t = np.linspace(0.0, 1.0, n_ab)
    y_ab = y_root + (y_c_start - y_root) * np.sin(0.5 * np.pi * t)

    # Uniform in y through the winglet. Its dihedral grows to ~45 deg, so this
    # is mildly stretched in arc length, but the winglet is short enough that
    # the extra clustering from winglet_fraction dominates.
    y_c = np.linspace(y_c_start, y_tip, n_c + 1)[1:]

    return np.concatenate([y_ab, y_c])


def chordwise_stations(nx_new, clustering="cosine"):
    """Normalized chordwise parameter for ``nx_new`` nodes, leading edge first.

    Parameters
    ----------
    nx_new : int
        Number of chordwise nodes, at least 2.
    clustering : str
        ``"cosine"`` clusters toward *both* edges: the leading edge carries the
        suction peak and the trailing edge sets the Kutta condition, so a VLM
        wants nodes at both. ``"uniform"`` is available for comparison.

    Returns
    -------
    ndarray
        ``nx_new`` values increasing from exactly 0.0 (leading edge) to exactly
        1.0 (trailing edge).
    """
    if nx_new < 2:
        raise ValueError("need at least two chordwise nodes")

    u = np.linspace(0.0, 1.0, nx_new)
    if clustering == "cosine":
        return 0.5 * (1.0 - np.cos(np.pi * u))
    if clustering == "uniform":
        return u
    raise ValueError(f"unknown clustering {clustering!r}")


def _camber_arclength(mesh):
    """Normalized cumulative arc length along each section's camber line.

    Returns ``(nx, ny)``, running 0 at the leading edge to 1 at the trailing
    edge for every section.
    """
    steps = np.linalg.norm(np.diff(mesh, axis=0), axis=2)
    s = np.vstack([np.zeros((1, mesh.shape[1])), np.cumsum(steps, axis=0)])
    return s / s[-1]


def resample_chordwise(mesh, nx_new=N_CHORDWISE, clustering="cosine"):
    """Re-discretize each section's camber line onto ``nx_new`` chordwise nodes.

    The interpolation parameter is normalized arc length along the camber line,
    not x. On a cambered section the camber line is longer than its chord and the
    leading-edge region turns fastest, so an x-parameterization would place too
    few nodes exactly where the curvature is; arc length spreads them evenly
    along the actual curve before the cosine clustering pulls them to the edges.

    PCHIP again: shape preserving, so a coarse chordwise cut cannot bulge the
    camber line past the original surface.

    The leading- and trailing-edge nodes are the parameter endpoints, where PCHIP
    reproduces its data exactly, so they are carried over rather than smoothed.

    Parameters
    ----------
    mesh : ndarray
        ``(nx, ny, 3)`` mesh, ``mesh[0]`` the leading edge.
    nx_new : int
        Number of chordwise nodes to keep.
    clustering : str
        Passed to :func:`chordwise_stations`.

    Returns
    -------
    (ndarray, dict)
        The ``(nx_new, ny, 3)`` mesh, and a residual dict with keys ``max``,
        ``rms``, ``max_relative`` and ``rms_relative``. As for the spanwise
        residual this is a round trip onto the original chordwise stations; the
        relative figures normalize each node by its own local chord.
    """
    nx, ny = mesh.shape[0], mesh.shape[1]
    if nx_new > nx:
        raise ValueError(f"cannot refine {nx} chordwise nodes to {nx_new}; resampling only coarsens")

    s = _camber_arclength(mesh)
    t_new = chordwise_stations(nx_new, clustering)

    mesh_new = np.zeros((nx_new, ny, 3))
    for j in range(ny):
        mesh_new[:, j, :] = PchipInterpolator(s[:, j], mesh[:, j, :], axis=0)(t_new)

    # Round trip: interpolate the coarse section back onto the original nodes.
    s_back = _camber_arclength(mesh_new)
    mesh_back = np.zeros_like(mesh)
    for j in range(ny):
        mesh_back[:, j, :] = PchipInterpolator(s_back[:, j], mesh_new[:, j, :], axis=0)(s[:, j])

    error = np.linalg.norm(mesh_back - mesh, axis=2)
    chord = np.linalg.norm(mesh[-1, :, :] - mesh[0, :, :], axis=1)
    relative = error / chord
    residual = {
        "max": float(error.max()),
        "rms": float(np.sqrt(np.mean(error**2))),
        "max_relative": float(relative.max()),
        "rms_relative": float(np.sqrt(np.mean(relative**2))),
    }

    return mesh_new, residual


def resample(mesh, y_new, nx_new=N_CHORDWISE, clustering="cosine"):
    """Resample a mesh spanwise and chordwise in one call.

    Spanwise first, then chordwise, so the chordwise error is measured on the
    sections that actually survive into the VLM.

    Returns
    -------
    (ndarray, dict)
        The ``(nx_new, len(y_new), 3)`` mesh and a residual dict
        ``{"spanwise": {...}, "chordwise": {...}}``. The two are kept separate
        so that a cheap chordwise cut cannot hide an expensive spanwise one, or
        the other way round.
    """
    spanwise_mesh, spanwise_residual = resample_spanwise(mesh, y_new)
    mesh_new, chordwise_residual = resample_chordwise(spanwise_mesh, nx_new, clustering)
    return mesh_new, {"spanwise": spanwise_residual, "chordwise": chordwise_residual}


def _leading_edge_arclength(mesh):
    """Cumulative arc length along the leading edge, one value per section."""
    le = mesh[0, :, :]
    steps = np.linalg.norm(np.diff(le, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def resample_spanwise(mesh, y_new):
    """Re-loft a mesh onto new spanwise stations.

    Each chordwise index line is interpolated as a function of *arc length along
    the leading edge*, not of y. The winglet turns through ~45 deg of dihedral,
    so y advances only ~0.7 m per metre of span there and a y-parameterization
    compresses the winglet knee into a near-vertical segment, which is exactly
    where an interpolant is least forgiving. Arc length stays proportional to
    real distance through the turn.

    The interpolant is PCHIP: cubic, C1, and shape preserving, so it does not
    overshoot at the knee the way a natural cubic spline does.

    Parameters
    ----------
    mesh : ndarray
        ``(nx, ny, 3)`` mesh of one half, sections ordered root to tip.
    y_new : ndarray
        Target spanwise stations, strictly increasing, within the original span.

    Returns
    -------
    (ndarray, dict)
        The ``(nx, len(y_new), 3)`` mesh, and a residual dict with keys ``max``,
        ``rms`` and ``max_relative``. The residual is a round trip: the
        resampled surface is interpolated back onto the *original* stations and
        compared node by node with the original mesh, so it measures how much
        geometry the new station count actually loses.
    """
    y_new = np.asarray(y_new, dtype=float)
    s = _leading_edge_arclength(mesh)
    y = mesh[0, :, 1]

    if not np.all(np.diff(y) > 0):
        raise ValueError("sections must be ordered by strictly increasing y")

    # Map the requested y stations onto the arc-length parameter, so the new
    # sections sit exactly at y_new.
    mesh_new = _interp_sections(mesh, s, _arclength_at_y(s, y, y_new))

    # Round trip: back onto the original stations.
    s_back = _leading_edge_arclength(mesh_new)
    mesh_back = _interp_sections(mesh_new, s_back, _arclength_at_y(s_back, mesh_new[0, :, 1], y))

    error = np.linalg.norm(mesh_back - mesh, axis=2)
    reference = np.linalg.norm(mesh[0, -1, :] - mesh[0, 0, :])
    residual = {
        "max": float(error.max()),
        "rms": float(np.sqrt(np.mean(error**2))),
        "max_relative": float(error.max() / reference),
    }

    return mesh_new, residual


def _arclength_at_y(s, y, y_new):
    """Arc lengths at which the leading edge crosses the stations ``y_new``.

    Interpolating s(y) is only a first guess: the curve is parameterized by s, so
    the station that actually lands at ``y_new`` is the root of ``y(s) - y_new``.
    Newton converges in a couple of steps here because y(s) is monotone with a
    derivative bounded away from zero (the winglet reaches ~45 deg of dihedral,
    so dy/ds stays above 0.7).
    """
    y_of_s = PchipInterpolator(s, y)
    dy_ds = y_of_s.derivative()

    s_new = PchipInterpolator(y, s)(y_new)
    for _ in range(20):
        residual = y_of_s(s_new) - y_new
        if np.max(np.abs(residual)) < 1e-12 * max(1.0, s[-1]):
            break
        s_new = np.clip(s_new - residual / dy_ds(s_new), s[0], s[-1])
    return s_new


def _interp_sections(mesh, s, s_new):
    """Interpolate every chordwise index line of ``mesh`` from ``s`` to ``s_new``."""
    nx = mesh.shape[0]
    out = np.zeros((nx, s_new.size, 3))
    for i in range(nx):
        out[i] = PchipInterpolator(s, mesh[i, :, :], axis=0)(s_new)
    return out
