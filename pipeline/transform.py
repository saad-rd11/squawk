import json
import re
from typing import Any, Optional

import pandas as pd

from .config import PipelineConfig
from .normalize import (
    NormalizationCounters,
    normalize_aircraft,
    normalize_conditions,
    normalize_date,
    normalize_factors,
    normalize_flight_plan,
    normalize_operator,
    normalize_phase,
    normalize_problem,
    normalize_state,
    normalize_text,
)
from .validate import (
    ChildPoint,
    ChildPointPayload,
    ParentPoint,
    ParentPointPayload,
    validate_payload_child,
    validate_payload_parent,
)


def chunk_text(text: str, max_chars: int = 2048) -> list[str]:
    if not text:
        return [""]
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 1 <= max_chars:
            current = (current + " " + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > max_chars:
                sentences = re.split(r"(?<=[.;!?])\s+", para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= max_chars:
                        current = (current + " " + sent).strip() if current else sent
                    else:
                        if current:
                            chunks.append(current)
                        current = sent
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks if chunks else [""]


def build_prefix(
    aircraft_models: list[str],
    aircraft_family: str,
    flight_phase: list[str],
    anomaly_plain: list[str],
    component_raw: Optional[str],
    prefix_fields: tuple[str, ...] = ("aircraft", "phase", "anomaly", "component"),
) -> str:
    parts: list[str] = []
    for field in prefix_fields:
        if field == "aircraft":
            ac_str = ", ".join(aircraft_models) if aircraft_models else "Unknown"
            family_str = aircraft_family if aircraft_family != "Unknown" else ""
            ac_part = ac_str
            if family_str:
                ac_part += f" ({family_str})"
            parts.append(f"Aircraft: {ac_part}")
        elif field == "phase" and flight_phase:
            parts.append(f"Phase: {', '.join(flight_phase)}")
        elif field == "anomaly" and anomaly_plain:
            parts.append(f"Anomaly: {', '.join(anomaly_plain)}")
        elif (
            field == "component"
            and component_raw
            and pd.notna(component_raw)
            and component_raw
        ):
            parts.append(f"Component: {component_raw.strip()}")
    return " | ".join(parts)


def merge_aircraft(
    a1_model: str,
    a1_family: str,
    a1_manufacturer: str,
    a2_model: str | None,
    a2_family: str | None,
    a2_manufacturer: str | None,
) -> tuple[list[str], str, str]:
    models: list[str] = []
    families: set[str] = set()
    manufacturers: set[str] = set()
    if a1_model and a1_model != "UNKN":
        models.append(a1_model)
        families.add(a1_family)
        manufacturers.add(a1_manufacturer)
    if a2_model and a2_model not in ("UNKN", "NONE") and a2_model not in models:
        models.append(a2_model)
        families.add(a2_family)
        manufacturers.add(a2_manufacturer)
    if not models:
        models.append("UNKN")
    family = "; ".join(sorted(families)) if families else "Unknown"
    manufacturer = "; ".join(sorted(manufacturers)) if manufacturers else "Unknown"
    return models, family, manufacturer


def transform_row(
    row: dict[str, Any],
    aircraft_map: dict[str, tuple[str, str, str]],
    config: PipelineConfig,
    expand_fn,
    counter: Optional[NormalizationCounters] = None,
) -> tuple[ParentPoint, list[ChildPoint]]:
    report_id = str(row["ACN"])

    a1_raw = row.get("Make Model Name")
    a1_model, a1_family, a1_manufacturer = normalize_aircraft(
        a1_raw, aircraft_map, counter=counter
    )

    a2_raw = row.get("Make Model Name.1")
    a2_model, a2_family, a2_manufacturer = (
        normalize_aircraft(a2_raw, aircraft_map, counter=counter)
        if pd.notna(a2_raw) and a2_raw
        else (None, None, None)
    )

    aircraft_models, aircraft_family, manufacturer = merge_aircraft(
        a1_model,
        a1_family,
        a1_manufacturer,
        a2_model,
        a2_family,
        a2_manufacturer,
    )

    date_iso, year = normalize_date(row.get("Date"), counter=counter)
    anomaly_codes, anomaly_plain = normalize_anomaly(
        row.get("Anomaly"),
        expand_fn=expand_fn,
        strict=True,
    )
    flight_phase = normalize_phase(row.get("Flight Phase"))
    flight_conditions = normalize_conditions(
        row.get("Flight Conditions"), counter=counter
    )
    flight_plan = normalize_flight_plan(row.get("Flight Plan"), counter=counter)
    primary_problem = normalize_problem(row.get("Primary Problem"))
    contributing_factors = normalize_factors(
        row.get("Contributing Factors / Situations")
    )
    state = normalize_state(row.get("State Reference"), counter=counter)
    operator_val = normalize_operator(row.get("Aircraft Operator"))

    synopsis = normalize_text(row.get("Synopsis", ""))
    narrative = normalize_text(row.get("Narrative", ""))
    narrative2_raw = row.get("Narrative.1")
    narrative2 = (
        normalize_text(narrative2_raw)
        if pd.notna(narrative2_raw) and narrative2_raw
        else ""
    )
    component_raw = row.get("Aircraft Component")

    prefix = build_prefix(
        aircraft_models,
        aircraft_family,
        flight_phase,
        anomaly_plain,
        component_raw,
        prefix_fields=config.prefix_fields,
    )

    parent_payload_data: dict[str, Any] = {
        "report_id": report_id,
        "year": year,
        "aircraft_models": aircraft_models,
        "aircraft_family": aircraft_family,
        "manufacturer": manufacturer,
        "flight_phase": flight_phase,
        "flight_conditions": flight_conditions,
        "flight_plan": flight_plan,
        "anomaly": anomaly_codes,
        "primary_problem": primary_problem,
        "contributing_factors": contributing_factors,
        "state": state,
        "operator": operator_val,
        "source": "ASRS",
        "synopsis": synopsis,
    }
    parent_payload = validate_payload_parent(parent_payload_data)

    structured_render = (
        f"{synopsis} | "
        f"{build_prefix(aircraft_models, aircraft_family, flight_phase, anomaly_plain, None, prefix_fields=('aircraft', 'phase', 'anomaly'))}"
    )

    parent_point = ParentPoint(
        id=f"asrs_{report_id}",
        vector=config.dense_model,
        payload=parent_payload,
    )

    chunks = chunk_text(narrative, max_chars=config.chunk_max_chars)
    child_points: list[ChildPoint] = []
    for c_idx, chunk in enumerate(chunks):
        child_id = f"asrs_{report_id}_n{c_idx}"
        text_for_vector = f"[{prefix}]\nNarrative: {chunk}" if prefix else chunk

        child_payload_data: dict[str, Any] = {
            "report_id": report_id,
            "year": year,
            "aircraft_models": aircraft_models,
            "aircraft_family": aircraft_family,
            "manufacturer": manufacturer,
            "flight_phase": flight_phase,
            "anomaly": anomaly_codes,
            "operator": operator_val,
            "state": state,
            "parent_id": parent_point.id,
            "narrative_source": "captain",
            "chunk_index": c_idx,
            "chunk_total": len(chunks),
            "context_prefix": prefix,
        }
        child_payload = validate_payload_child(child_payload_data)

        child_points.append(
            ChildPoint(
                id=child_id,
                vector=config.dense_model,
                sparse=config.sparse_model,
                payload=child_payload,
            )
        )

    return parent_point, child_points


def normalize_anomaly(
    raw: str | None,
    expand_fn,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    if pd.isna(raw) or not raw:
        return ([], [])
    parts = [p.strip() for p in raw.split(";")]
    return expand_fn(parts, strict=strict)
