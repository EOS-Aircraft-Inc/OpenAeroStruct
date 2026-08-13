"""Parser for OpenVSP DegenGeom CSV exports.

OpenAeroStruct's own VSP path (``openaerostruct.geometry.utils.generate_vsp_surfaces``)
drives the OpenVSP Python API in-process. That API is not pip-installable and is
not present here, but the DegenGeom *CSV* export carries the same DegenPlate and
DegenStick data, so we read it directly and skip OpenVSP entirely.

The camber-surface reconstruction below mirrors ``generate_vsp_surfaces``
(``openaerostruct/geometry/utils.py:509-511``) exactly, so meshes produced from a
CSV match meshes produced through the live API.

File structure
--------------
Comment lines begin with ``#`` and name the columns of the block that follows.
Each component is introduced by a type/name header line, then a sequence of
blocks::

    SURFACE_NODE,<nXsecs>,<nPnts>   nXsecs*nPnts rows of  x,y,z,u,w
    SURFACE_FACE,...                (nXsecs-1)*(nPnts-1) rows of  nx,ny,nz,area
    PLATE,<nXsecs>,<nPnts>          nXsecs rows of  nx,ny,nz   (plate normals)
                                    then nXsecs*nPnts rows of
                                    x,y,z,zCamber,t,nCamberx,...
    STICK_NODE,<nXsecs>             nXsecs rows of  lex,...,toc,tLoc,chord,...
    STICK_FACE,<nXsecs-1>           sweeple,sweepte,areaTop,areaBot
    POINT                           one row of volume/area/inertia properties
"""

from dataclasses import dataclass, field

import numpy as np

# Type keywords that introduce a data block, mapped to how many integer sizes
# follow the keyword on the same line.
_BLOCK_SIZES = {
    "SURFACE_NODE": 2,
    "SURFACE_FACE": 2,
    "PLATE": 2,
    "STICK_NODE": 1,
    "STICK_FACE": 1,
    "POINT": 0,
}

_COMPONENT_TYPES = ("LIFTING_SURFACE", "BODY", "DISK", "MESH")


@dataclass
class DegenPlate:
    """The camber-plate representation of one surface.

    Attributes are (nXsecs, nPnts) arrays matching OpenVSP's DegenPlate.
    """

    num_secs: int
    num_pnts: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    zCamber: np.ndarray
    t: np.ndarray
    nCamber_x: np.ndarray
    nCamber_y: np.ndarray
    nCamber_z: np.ndarray
    u: np.ndarray
    wTop: np.ndarray
    wBot: np.ndarray

    def camber_surface(self):
        """Return (x, y, z) of the mean camber surface, each (nXsecs, nPnts).

        Same expression as ``openaerostruct/geometry/utils.py:509-511``.
        """
        return (
            self.x + self.nCamber_x * self.zCamber,
            self.y + self.nCamber_y * self.zCamber,
            self.z + self.nCamber_z * self.zCamber,
        )


@dataclass
class DegenStick:
    """Per-section properties from the STICK_NODE block."""

    num_secs: int
    columns: dict = field(repr=False)

    def __getitem__(self, key):
        return self.columns[key]

    @property
    def chord(self):
        return self.columns["chord"]

    @property
    def toc(self):
        """Thickness-to-chord ratio per section."""
        return self.columns["toc"]

    @property
    def tLoc(self):
        """Chordwise location of maximum thickness, as a chord fraction."""
        return self.columns["tLoc"]

    @property
    def le(self):
        """Leading-edge coordinates, (nXsecs, 3)."""
        return np.column_stack([self.columns[k] for k in ("lex", "ley", "lez")])

    @property
    def te(self):
        """Trailing-edge coordinates, (nXsecs, 3)."""
        return np.column_stack([self.columns[k] for k in ("tex", "tey", "tez")])

    @property
    def twist(self):
        """Geometric twist per section in degrees, from the chord line."""
        le, te = self.le, self.te
        return np.degrees(np.arctan2(-(te[:, 2] - le[:, 2]), te[:, 0] - le[:, 0]))

    @property
    def quarter_chord(self):
        """Quarter-chord line, (nXsecs, 3)."""
        le, te = self.le, self.te
        return le + 0.25 * (te - le)

    def le_sweep(self):
        """Leading-edge sweep in degrees for each panel, length nXsecs-1."""
        le = self.le
        return np.degrees(np.arctan2(np.diff(le[:, 0]), np.diff(le[:, 1])))

    def dihedral(self):
        """Dihedral in degrees for each panel, length nXsecs-1."""
        le = self.le
        return np.degrees(np.arctan2(np.diff(le[:, 2]), np.diff(le[:, 1])))


@dataclass
class DegenComponent:
    """One geometry component (one symmetric half of one VSP geom)."""

    type: str
    name: str
    surf_index: int
    geom_id: str
    main_surf_index: int
    sym_copy_index: int
    flip_normal: bool
    plates: list = field(default_factory=list, repr=False)
    sticks: list = field(default_factory=list, repr=False)

    @property
    def plate(self):
        """The single plate, for components that have exactly one."""
        if len(self.plates) != 1:
            raise ValueError(f"{self.name} has {len(self.plates)} plates; index self.plates directly")
        return self.plates[0]

    @property
    def stick(self):
        if len(self.sticks) != 1:
            raise ValueError(f"{self.name} has {len(self.sticks)} sticks; index self.sticks directly")
        return self.sticks[0]


def _header_columns(line):
    """Split a ``# a,b,c`` comment line into column names."""
    return [c.strip() for c in line.lstrip("#").split(",") if c.strip()]


def _read_rows(lines, start, count, ncols=None):
    """Read ``count`` comma-separated float rows beginning at ``lines[start]``."""
    rows = []
    i = start
    while len(rows) < count:
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        values = [float(v) for v in line.split(",") if v.strip()]
        if ncols is not None:
            values = values[:ncols]
        rows.append(values)
    return np.array(rows), i


def read_degen_csv(path):
    """Parse a DegenGeom CSV export.

    Returns a list of :class:`DegenComponent`, one per symmetric copy, in file
    order. For a symmetric wing this is two entries with ``surf_index`` 0 and 1.
    """
    with open(path) as f:
        lines = f.read().splitlines()

    components = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Comment lines name the columns of the block that follows, but each
        # block re-reads its own header when it needs the names, so they can be
        # skipped here.
        if not line or line.startswith("#"):
            i += 1
            continue

        fields = [f.strip() for f in line.split(",")]
        keyword = fields[0]

        if keyword in _COMPONENT_TYPES:
            components.append(
                DegenComponent(
                    type=keyword,
                    name=fields[1],
                    surf_index=int(fields[2]),
                    geom_id=fields[3],
                    main_surf_index=int(fields[4]),
                    sym_copy_index=int(fields[5]),
                    flip_normal=bool(int(fields[6])),
                )
            )
            i += 1
            continue

        if keyword in _BLOCK_SIZES:
            nsize = _BLOCK_SIZES[keyword]
            sizes = [int(v) for v in fields[1 : 1 + nsize]]
            i += 1

            if keyword == "PLATE":
                num_secs, num_pnts = sizes
                # Plate normals, one per section, then the camber data.
                _, i = _read_rows(lines, i, num_secs)
                # Advance to the data column header so we can name the columns.
                while not lines[i].strip().startswith("#"):
                    i += 1
                columns = _header_columns(lines[i])
                data, i = _read_rows(lines, i + 1, num_secs * num_pnts, ncols=len(columns))
                col = {name: data[:, j].reshape(num_secs, num_pnts) for j, name in enumerate(columns)}
                components[-1].plates.append(
                    DegenPlate(
                        num_secs=num_secs,
                        num_pnts=num_pnts,
                        x=col["x"],
                        y=col["y"],
                        z=col["z"],
                        zCamber=col["zCamber"],
                        t=col["t"],
                        nCamber_x=col["nCamberx"],
                        nCamber_y=col["nCambery"],
                        nCamber_z=col["nCamberz"],
                        u=col["u"],
                        wTop=col["wTop"],
                        wBot=col["wBot"],
                    )
                )

            elif keyword == "STICK_NODE":
                (num_secs,) = sizes
                while not lines[i].strip().startswith("#"):
                    i += 1
                columns = _header_columns(lines[i])
                data, i = _read_rows(lines, i + 1, num_secs, ncols=len(columns))
                components[-1].sticks.append(
                    DegenStick(
                        num_secs=num_secs,
                        columns={name: data[:, j] for j, name in enumerate(columns)},
                    )
                )

            else:
                # Blocks we do not consume: skip their rows so parsing stays aligned.
                counts = {
                    "SURFACE_NODE": lambda s: s[0] * s[1],
                    "SURFACE_FACE": lambda s: (s[0] - 1) * (s[1] - 1),
                    "STICK_FACE": lambda s: s[0],
                    "POINT": lambda s: 1,
                }
                _, i = _read_rows(lines, i, counts[keyword](sizes))

            continue

        # Anything else (subsurfaces, hinge lines, blank sections) is skipped.
        i += 1

    return components


def lifting_surfaces(components):
    """Group LIFTING_SURFACE components by geom id, in surf_index order."""
    groups = {}
    for comp in components:
        if comp.type != "LIFTING_SURFACE":
            continue
        groups.setdefault(comp.geom_id, []).append(comp)
    for comps in groups.values():
        comps.sort(key=lambda c: c.surf_index)
    return groups
