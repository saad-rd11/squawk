"""
ASRS Anomaly → ADREP Occurrence Category Mapping

Single source of truth consumed by three consumers:
  1. Prefix builder   → plain-language expansion for embedding (dense + sparse)
  2. Payload normalizer → canonical ADREP code for filter payload
  3. Schema docs       → ASRS→ADREP bridge for cross-database alignment

Adding a new mapping: edit this dict. That's it. The three consumers
stay in sync automatically because they all read from this one constant.

Coverage: 98.1% of all anomaly instances in the ASRS 2020-2022 dataset.
The remaining 1.9% are null entries (incidents with no anomaly recorded).
"""

ASRS_TO_ADREP: dict[str, tuple[str, str]] = {
    # ── Midair / Near Midair Collision ──
    "Conflict NMAC": ("MAC", "midair near midair collision"),
    "Conflict Airborne Conflict": ("MAC", "midair near midair collision"),
    # ── Controlled Flight Into Terrain ──
    "Inflight Event / Encounter CFTT / CFIT": (
        "CFIT",
        "controlled flight into terrain",
    ),
    # ── Loss of Control ──
    "Inflight Event / Encounter Loss Of Aircraft Control": (
        "LOC-I",
        "loss of control inflight",
    ),
    "Ground Event / Encounter Loss Of Aircraft Control": (
        "LOC-G",
        "loss of control ground",
    ),
    # ── System / Component Failure (Non-Powerplant) ──
    "Aircraft Equipment Problem Critical": ("SCF-NP", "system component failure"),
    "Aircraft Equipment Problem Less Severe": ("SCF-NP", "system component failure"),
    "Critical": ("SCF-NP", "system component failure"),
    "Less Severe": ("SCF-NP", "system component failure"),
    # ── Fire / Smoke (Non-Impact) ──
    "Flight Deck / Cabin / Aircraft Event Smoke / Fire / Fumes / Odor": (
        "F-NI",
        "fire smoke non impact",
    ),
    # ── Turbulence ──
    "Inflight Event / Encounter Weather / Turbulence": ("TURB", "turbulence encounter"),
    # ── Fuel Related ──
    "Inflight Event / Encounter Fuel Issue": ("FUEL", "fuel related"),
    "Ground Event / Encounter Fuel Issue": ("FUEL", "fuel related"),
    # ── Birdstrike ──
    "Inflight Event / Encounter Bird / Animal": ("BIRD", "birdstrike"),
    # ── Air Traffic Management / CNS ──
    "ATC Issue All Types": ("ATM", "air traffic management"),
    "Deviation - Track / Heading All Types": ("ATM", "air traffic management"),
    "Deviation - Altitude Excursion From Assigned Altitude": (
        "ATM",
        "air traffic management",
    ),
    "Deviation - Altitude Overshoot": ("ATM", "air traffic management"),
    "Deviation - Altitude Undershoot": ("ATM", "air traffic management"),
    "Deviation - Altitude Crossing Restriction Not Met": (
        "ATM",
        "air traffic management",
    ),
    "Deviation - Speed All Types": ("ATM", "air traffic management"),
    "Deviation / Discrepancy - Procedural Clearance": ("ATM", "air traffic management"),
    "Airspace Violation All Types": ("ATM", "air traffic management"),
    "Deviation / Discrepancy - Procedural Unauthorized Flight Operations (UAS)": (
        "ATM",
        "air traffic management",
    ),
    # ── Ground Collision ──
    "Conflict Ground Conflict": ("GCOL", "ground collision"),
    "Ground Event / Encounter Ground Strike - Aircraft": ("GCOL", "ground collision"),
    # ── Runway Excursion ──
    "Ground Excursion Runway": ("RE", "runway excursion"),
    "Ground Excursion Taxiway": ("RE", "runway excursion"),
    "Ground Excursion Ramp": ("RE", "runway excursion"),
    # ── Runway Incursion (Vehicle, Aircraft, Person) ──
    "Ground Incursion Runway": ("RI-VAP", "runway incursion vehicle aircraft person"),
    "Ground Incursion Taxiway": ("RI-VAP", "runway incursion vehicle aircraft person"),
    "Ground Incursion Ramp": ("RI-VAP", "runway incursion vehicle aircraft person"),
    # ── Ground Handling ──
    "Ground Event / Encounter Ground Equipment Issue": ("RAMP", "ground handling"),
    "Ground Event / Encounter Vehicle": ("RAMP", "ground handling"),
    "Ground Event / Encounter Object": ("RAMP", "ground handling"),
    "Ground Event / Encounter Person / Animal / Bird": ("RAMP", "ground handling"),
    "Ground Event / Encounter FOD": ("RAMP", "ground handling"),
    "Ground Event / Encounter Jet Blast": ("RAMP", "ground handling"),
    "Ground Event / Encounter Aircraft": ("RAMP", "ground handling"),
    "Ground Event / Encounter Other / Unknown": ("RAMP", "ground handling"),
    # ── Cabin Safety ──
    "Flight Deck / Cabin / Aircraft Event Illness / Injury": ("CABIN", "cabin safety"),
    "Flight Deck / Cabin / Aircraft Event Passenger Misconduct": (
        "CABIN",
        "cabin safety",
    ),
    "Flight Deck / Cabin / Aircraft Event Passenger Electronic Device": (
        "CABIN",
        "cabin safety",
    ),
    "Flight Deck / Cabin / Aircraft Event Other / Unknown": ("CABIN", "cabin safety"),
    # ── Unintended Flight in IMC ──
    "Inflight Event / Encounter VFR In IMC": ("UIMC", "unintended flight in imc"),
    # ── Inflight Object Encounter ──
    "Inflight Event / Encounter Object": ("OTHR", "other"),
    # ── Undershoot / Overshoot ──
    "Ground Event / Encounter Gear Up Landing": ("USOS", "undershoot overshoot"),
    # ── Other ──
    "Deviation / Discrepancy - Procedural Published Material / Policy": (
        "OTHR",
        "other",
    ),
    "Deviation / Discrepancy - Procedural FAR": ("OTHR", "other"),
    "Deviation / Discrepancy - Procedural Maintenance": ("OTHR", "other"),
    "Deviation / Discrepancy - Procedural Hazardous Material Violation": (
        "OTHR",
        "other",
    ),
    "Deviation / Discrepancy - Procedural Weight And Balance": ("OTHR", "other"),
    "Deviation / Discrepancy - Procedural MEL / CDL": ("OTHR", "other"),
    "Deviation / Discrepancy - Procedural Landing Without Clearance": ("OTHR", "other"),
    "Deviation / Discrepancy - Procedural Other / Unknown": ("OTHR", "other"),
    "Deviation / Discrepancy - Procedural Security": ("OTHR", "other"),
    "Inflight Event / Encounter Other / Unknown": ("OTHR", "other"),
    "Inflight Event / Encounter Unstabilized Approach": ("OTHR", "other"),
    "Inflight Event / Encounter Wake Vortex Encounter": ("OTHR", "other"),
    "Inflight Event / Encounter Aircraft": ("OTHR", "other"),
    "Inflight Event / Encounter Fly Away (UAS)": ("OTHR", "other"),
    "Inflight Event / Encounter Laser": ("OTHR", "other"),
    "Ground Event / Encounter Weather / Turbulence": ("OTHR", "other"),
    "Ground Event / Encounter Loss Of VLOS (UAS)": ("OTHR", "other"),
    # ── No Specific Anomaly ──
    "No Specific Anomaly Occurred Unwanted Situation": ("UNK", "no specific anomaly"),
    "No Specific Anomaly Occurred All Types": ("UNK", "no specific anomaly"),
    # ── Rare / Catch-All ──
    "Other uav report": ("OTHR", "other"),
    "Other Fatigue": ("OTHR", "other"),
    "Other Scheduling changes": ("OTHR", "other"),
    "Other Service Road signs needed": ("OTHR", "other"),
}

# Reverse lookup: ADREP code → plain language (uses first entry per code)
ADREP_CODE_TO_PLAIN: dict[str, str] = {}
for _, (code, plain) in ASRS_TO_ADREP.items():
    if code not in ADREP_CODE_TO_PLAIN:
        ADREP_CODE_TO_PLAIN[code] = plain

# All known ADREP occurrence category codes
ALL_ADREP_CODES: list[str] = sorted(ADREP_CODE_TO_PLAIN.keys())


def validate_export(anomaly_series: "pd.Series") -> None:
    """Pre-ingestion check: verify all anomaly strings in an export are mappable.

    Call this once over the entire dataframe *before* the row-by-row ingestion
    loop. If unmapped strings exist, ingestion refuses to start and hands you
    the complete list — no debugging mid-run.

    Args:
        anomaly_series: The anomaly column from the raw export
                        (a pandas Series of semicolon-joined strings).

    Raises:
        ValueError: With the full list of unmapped strings if any are found.
    """
    import pandas as pd  # noqa: F811

    if not isinstance(anomaly_series, pd.Series):
        raise TypeError("anomaly_series must be a pandas Series")

    all_strings = anomaly_series.dropna().str.split("; ").explode().str.strip().unique()
    result = verify_coverage(set(all_strings))
    if result["unmapped_count"] > 0:
        raise ValueError(
            f"Export has {result['unmapped_count']} unmapped anomaly string(s).\n"
            f"Add these to ASRS_TO_ADREP in anomaly_map.py before ingesting:\n"
            f"  {chr(10).join('  - ' + s for s in result['unmapped_values'])}"
        )


def map_anomaly(asrs_string: str | None) -> tuple[str, str] | None:
    """Map a single ASRS anomaly string to (ADREP_code, plain_language).

    Returns None for null/empty input.
    """
    if not asrs_string:
        return None
    key = asrs_string.strip()
    return ASRS_TO_ADREP.get(key, None)


def expand_anomalies(
    asrs_list: list[str],
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Expand a list of ASRS anomaly strings into ADREP codes and plain text.

    Args:
        asrs_list: Raw ASRS anomaly strings from the ;-joined field.
        strict: If True, raises ValueError on any unmapped string.
                If False, routes unmapped to OTHR and logs a warning.

    Returns:
        (codes, plain_list) — deduplicated, in order of first appearance.

    Raises:
        ValueError: In strict mode, if any ASRS string has no mapping.
    """
    import logging

    logger = logging.getLogger(__name__)
    codes: list[str] = []
    plain: list[str] = []
    for raw in asrs_list:
        mapped = map_anomaly(raw)
        if mapped:
            c, p = mapped
            if c not in codes:
                codes.append(c)
                plain.append(p)
        elif strict:
            raise ValueError(
                f"Unmapped ASRS anomaly string: {raw!r}. "
                f"Add it to ASRS_TO_ADREP in anomaly_map.py before ingesting."
            )
        else:
            logger.warning("Unmapped ASRS anomaly string %r → routed to OTHR", raw)
            if "OTHR" not in codes:
                codes.append("OTHR")
                plain.append("other")
    return codes, plain


def verify_coverage(asrs_strings: set[str]) -> dict:
    """Check coverage of the mapping against a set of observed ASRS anomaly strings.

    Returns dict with total, mapped, unmapped counts and the unmapped values.
    """
    mapped = sum(1 for s in asrs_strings if s in ASRS_TO_ADREP)
    unmapped = [s for s in asrs_strings if s not in ASRS_TO_ADREP]
    return {
        "total": len(asrs_strings),
        "mapped": mapped,
        "unmapped_count": len(unmapped),
        "unmapped_values": sorted(unmapped),
        "coverage_pct": round(mapped / len(asrs_strings) * 100, 1)
        if asrs_strings
        else 0,
    }
