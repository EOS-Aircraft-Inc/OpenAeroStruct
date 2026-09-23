"""The OpenAeroStruct wing report -- WingCalc's report, drawn from OAS data.

Five modules, split on what each one is allowed to know about:

``wingcalc_frontend``
    A vendored copy of ``Structures-WingCalc_Tool/viewer/wing_viewer.py``'s
    renderer. Pure: dicts in, HTML out. It knows nothing about OpenAeroStruct.
``oas_snapshot``
    Runs (or re-runs) the OAS model at a converged design point and flattens
    everything the report needs into a plain JSON-serializable dict, in the
    report's units. It knows nothing about the report's shapes.
``oas_report``
    Turns a snapshot into the WingCalc-shaped payload dicts and writes the page.
    The public entry point ``generate_oas_viewer`` and the CLI live here.

``compare``
    A two-pane page framing WingCalc's report and OAS's. It knows nothing about
    either one beyond where the file is.
``pair``
    What an arc run calls at the end: restates the run in the vocabulary
    ``oas_snapshot`` reads, then writes both the OAS report and the compare page
    into WingCalc's own output folder, so the two land beside each other.

The seam between ``oas_snapshot`` and ``oas_report`` is deliberate: a snapshot
can be cached and the page re-rendered from it in under a second, with no
OpenMDAO, no aerosandbox and no FEM solve -- which is what makes iterating on
the report bearable.
"""

from studies.vsp_planform.viewer.compare import write_compare_page
from studies.vsp_planform.viewer.oas_report import generate_oas_viewer

__all__ = ["generate_oas_viewer", "write_compare_page"]
