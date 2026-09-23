"""Every number the weight build-up reads, from the deck WingCalc reads it from.

The build-up is only WingCalc's method if it is also WingCalc's INPUTS. So this
reads the same Baseline deck ``coupling.deck.write_deck`` copies (``WC_DECK``,
under ``WINGCALC_ROOT``) and applies the same two overrides that function applies
before WingCalc ever sees the deck:

* ``Total wingbox span`` becomes ``deck.WINGBOX_SPAN_IN`` (1,356 in);
* each load case's ``AC_Weight`` becomes ``deck.case_weight_lb`` -- MTOW less the
  unfuelled wing tanks -- the SAME function the deck writer calls, so the two can
  never be cast in different aircraft weights.

Everything else is read as the deck has it. Nothing here is a design variable; it
is the installation, the rib rules, the control-surface layout and the load cases.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from studies.vsp_planform.coupling import deck as wcdeck
from studies.vsp_planform.coupling import mission


def _name_value_rows(path: Path) -> dict[str, str]:
    """``Name,Value,...`` rows as a dict of raw strings, first column as the key.

    planformIn.csv is ``Name,Catergory,Value,Units`` and wingLoadingIn.csv is
    ``Name,Value,Units``; the caller says which column the value is in.
    """
    return {
        row[0].strip(): row
        for row in csv.reader(path.open(newline="", encoding="utf-8-sig"))
        if row and row[0].strip()
    }


@dataclass(frozen=True)
class Material:
    density: float      # lb/in3
    ply_t: float        # in, 0 for a metal


@dataclass(frozen=True)
class WeightDeck:
    """The deck, as the weight build-up needs it. Inches, pounds."""

    source: str
    # planformIn.csv
    half_wingbox_span: float
    inboard_nacelle_Y: float
    outboard_nacelle_Y: float
    W2F_BL: float
    winglet_span: float
    winglet_tip_chord: float
    rib_pitch_inbrd_nacelle: float
    rib_pitch_outbrd_nacelle: float
    avg_pitch_inbrd: float
    avg_pitch_mid: float
    avg_pitch_outbrd: float
    fuse_diameter: float
    flap_Ksup: float
    flap_Kslot: float
    flap_span_ratio: float
    aileron_Kbal: float
    aileron_span_ratio: float
    wing_splices: int
    splice_bay: int
    splice_length: float
    mfs: tuple[tuple[float, float], ...]
    skin_material: str
    web_material: str
    # wingLoadingIn.csv, every row, as numbers
    wing_loading: dict[str, float] = field(repr=False)
    # loadCasesIn.csv, every row, with AC_Weight overridden
    load_cases: tuple[dict[str, str], ...] = field(repr=False)
    # Materials.csv
    materials: dict[str, Material] = field(repr=False)

    def wl(self, key: str) -> float:
        """One wingLoadingIn.csv value. Missing is an error, as it is in WingCalc."""
        if key not in self.wing_loading:
            raise KeyError(f"wingLoadingIn.csv in {self.source} has no '{key}' row")
        return self.wing_loading[key]


def _materials(path: Path) -> dict[str, Material]:
    """Materials.csv holds a composite block then a metallic block.

    The two blocks put density and ply thickness in different columns, so each is
    read against its own header row rather than by position.
    """
    out: dict[str, Material] = {}
    header = None
    for row in csv.reader(path.open(newline="", encoding="utf-8-sig")):
        if not row or not any(c.strip() for c in row):
            header = None
            continue
        if row[0].strip() == "Material Name":
            header = [c.strip() for c in row]
            continue
        if header is None:
            continue
        rec = dict(zip(header, (c.strip() for c in row)))
        dens = next(v for k, v in rec.items() if k.lower().startswith("density"))
        ply = next((v for k, v in rec.items() if k.lower().startswith("ply thickness")), "")
        out[rec["Material Name"]] = Material(float(dens), float(ply) if ply else 0.0)
    return out


def read_weight_deck(deck_dir: Path | None = None,
                     mtow_lb: float = mission.MTOW_LB) -> WeightDeck:
    """Read the deck the weight build-up is cast in. Defaults to ``WC_DECK``."""
    deck_dir = Path(deck_dir) if deck_dir is not None else wcdeck.WC_DECK
    pf = _name_value_rows(deck_dir / "planformIn.csv")
    wl_rows = _name_value_rows(deck_dir / "wingLoadingIn.csv")

    def p(key: str) -> float:
        return float(pf[key][2])

    def ps(key: str) -> str:
        return pf[key][2].strip()

    wing_loading = {}
    for key, row in wl_rows.items():
        try:
            wing_loading[key] = float(row[1])
        except (IndexError, ValueError):
            continue            # the header row

    rows = list(csv.DictReader((deck_dir / "loadCasesIn.csv").open(newline="", encoding="utf-8-sig")))
    cases = []
    for r in rows:
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        if not r.get("Load_case"):
            continue
        r["AC_Weight"] = repr(wcdeck.case_weight_lb(deck_dir, mtow_lb, float(r["Fuel_fraction"])))
        cases.append(r)

    return WeightDeck(
        source=str(deck_dir),
        half_wingbox_span=0.5 * wcdeck.WINGBOX_SPAN_IN,
        inboard_nacelle_Y=p("Inboard Nacelle Y"),
        outboard_nacelle_Y=p("Outboard Nacelle Y"),
        W2F_BL=p("W2F BL"),
        winglet_span=p("Winglet Span"),
        winglet_tip_chord=p("Winglet tip chord"),
        rib_pitch_inbrd_nacelle=p("Rib pitch inbrd nacelle"),
        rib_pitch_outbrd_nacelle=p("Rib pitch outbrd nacelle"),
        avg_pitch_inbrd=p("AVG Pitch inbrd"),
        avg_pitch_mid=p("AVG Pitch mid"),
        avg_pitch_outbrd=p("AVG Pitch outbrd"),
        fuse_diameter=p("Fuse diameter"),
        flap_Ksup=p("flap_Ksup"),
        flap_Kslot=p("flap_Kslot"),
        flap_span_ratio=p("Flap span ratio"),
        aileron_Kbal=p("Aileron_Kbal"),
        aileron_span_ratio=p("Aileron span ratio"),
        wing_splices=int(round(p("wing_splices"))),
        splice_bay=int(round(p("splice_bay"))),
        splice_length=p("splice_length"),
        mfs=((p("mfs1_start"), p("mfs1_end")), (p("mfs2_start"), p("mfs2_end"))),
        skin_material=ps("Skin Material"),
        web_material=ps("Spar Web Material"),
        wing_loading=wing_loading,
        load_cases=tuple(cases),
        materials=_materials(deck_dir / "Materials.csv"),
    )
