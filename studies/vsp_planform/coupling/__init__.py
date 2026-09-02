"""OpenAeroStruct <-> WingCalc coupling.

The aero study and the structural sizing tool model the same wing and, until
this package existed, could not be compared: OAS reduces a section to t/c and
c_max_t and carries no structure at all, while WingCalc reads real airfoil
contours and sizes real laminates but takes its loads from a hardcoded elliptical
distribution.

  geometry.py  write the OpenVSP station export WingCalc's provider reads, from
               an OAS design. Chord and t/c round-trip to 4+ digits.
  deck.py      build a WingCalc input deck from an OAS design and run the sizer.
  mission.py   MTOW / battery / electric-range bookkeeping shared by the drivers.

Validated against an independent WingCalc run of the same constant-chord wing
(V3.5.4): 20 bays closed, wing weight within 0.8%.
"""

from studies.vsp_planform.coupling import geometry, mission  # noqa: F401
