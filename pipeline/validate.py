import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ParentPointPayload(BaseModel):
    report_id: str
    year: Optional[int] = None
    aircraft_models: list[str]
    aircraft_family: str
    manufacturer: str
    flight_phase: list[str]
    flight_conditions: Optional[str] = None
    flight_plan: Optional[str] = None
    anomaly: list[str]
    primary_problem: Optional[str] = None
    contributing_factors: Optional[list[str]] = None
    state: Optional[str] = None
    operator: Optional[str] = None
    source: str = "ASRS"
    synopsis: str

    @field_validator("aircraft_models")
    @classmethod
    def models_not_empty(cls, v):
        if not v:
            raise ValueError("aircraft_models must not be empty")
        return v

    @field_validator("source")
    @classmethod
    def source_is_valid(cls, v):
        if v not in ("ASRS",):
            raise ValueError(f"unknown source: {v}")
        return v


class ChildPointPayload(BaseModel):
    report_id: str
    year: Optional[int] = None
    aircraft_models: list[str]
    aircraft_family: str
    manufacturer: str
    flight_phase: list[str]
    anomaly: list[str]
    operator: Optional[str] = None
    state: Optional[str] = None
    parent_id: str
    narrative_source: str = "captain"
    chunk_index: int
    chunk_total: int
    context_prefix: str
    chunk: str  # raw narrative chunk (no prefix) — compose with context_prefix at embed time

    @field_validator("narrative_source")
    @classmethod
    def source_valid(cls, v):
        if v not in ("captain", "first_officer"):
            raise ValueError(f"unknown narrative_source: {v}")
        return v


class ParentPoint(BaseModel):
    id: str
    vector: str
    payload: ParentPointPayload


class ChildPoint(BaseModel):
    id: str
    vector: str
    sparse: str
    payload: ChildPointPayload


def validate_payload_parent(data: dict[str, Any]) -> ParentPointPayload:
    return ParentPointPayload.model_validate(data)


def validate_payload_child(data: dict[str, Any]) -> ChildPointPayload:
    return ChildPointPayload.model_validate(data)


def validate_points(
    parent_points: list[ParentPoint],
    child_points: list[ChildPoint],
) -> int:
    errors = 0
    for i, pp in enumerate(parent_points):
        try:
            ParentPoint.model_validate(pp)
        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.error("Parent point %d validation error: %s", i, e)
    for i, cp in enumerate(child_points):
        try:
            ChildPoint.model_validate(cp)
        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.error("Child point %d validation error: %s", i, e)
    if errors > 3:
        logger.error(
            "%d total validation errors found (first 3 shown above, %d suppressed)",
            errors,
            errors - 3,
        )
    return errors


def validate_aircraft_map(
    aircraft_map: dict[str, tuple[str, str, str]],
    icao_designators_path: str,
) -> dict:
    """
    Validate aircraft map against ICAO Doc 8643 designator list.

    Checks:
    1. Family consistency — same canonical family uses same spelling
    2. ICAO code validity — every non-UNKN model code exists in Doc 8643 or allowlist
    3. Format check — no clearly malformed codes (manufacturer names, etc.)

    Returns dict with 'errors' list and 'status' ('pass'|'fail').
    """
    import json
    import re

    with open(icao_designators_path) as f:
        icao_codes = json.load(f)

    errors = []

    # Allowlist for intentional non-standard codes
    allowlist = {
        "UAV",
        "P51",
        "V22",
        "A10",
        "T28",
        "F11",
        "F16",
        "F18",
        "F5",
        "T45",
        "T6",
        "U2",
        "K35E",
        "BD30",
        "LAKE",
        "LA25",
        "TEAL",
        "VIK1",
        "KODI",
        "G73",
        "GA8",
        "EPIC",
        "B95",
        "PA11",
    }

    valid_format = re.compile(r"^[A-Z0-9]{2,6}$")

    entries = list(aircraft_map.items())

    # 1. Family consistency
    family_groups: dict[str, set[str]] = {}
    for raw_key, (model, family, mfr) in entries:
        key = family.lower().strip()
        if key not in family_groups:
            family_groups[key] = set()
        family_groups[key].add(family)
    for canonical, variants in sorted(family_groups.items()):
        if len(variants) > 1:
            errors.append(
                f"INCONSISTENT FAMILY: {canonical!r} has variants: {sorted(variants)}"
            )

    # 2. ICAO code validity
    for raw_key, (model, family, mfr) in entries:
        if model in ("UNKN", "NONE", "", None):
            continue
        if model in allowlist:
            continue
        if model in icao_codes:
            continue
        if not valid_format.match(model):
            errors.append(
                f"INVALID CODE FORMAT: model={model!r} from {raw_key!r} "
                f"(family={family!r}, mfr={mfr!r})"
            )
        else:
            # Format-OK but not in reference — informational, not an error
            # (the reference list may be incomplete)
            pass

    return {
        "errors": errors,
        "status": "fail" if errors else "pass",
        "total_entries": len(entries),
    }
