import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import PipelineConfig


@dataclass
class NormalizationCounters:
    aircraft_unknown: int = 0
    date_invalid: int = 0
    conditions_unknown: int = 0
    flight_plan_unknown: int = 0
    state_unknown: int = 0

    def report(self) -> dict:
        return {
            "aircraft_unknown": self.aircraft_unknown,
            "date_invalid": self.date_invalid,
            "conditions_unknown": self.conditions_unknown,
            "flight_plan_unknown": self.flight_plan_unknown,
            "state_unknown": self.state_unknown,
        }


def load_aircraft_map(path: str) -> dict[str, tuple[str, str, str]]:
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for k, v in raw.items():
        result[k] = (v[0], v[1], v[2])
    return result


def load_generic_descriptors(path: str) -> set[str]:
    with open(path) as f:
        return set(json.load(f))


def normalize_aircraft(
    raw: str | None,
    aircraft_map: dict[str, tuple[str, str, str]],
    counter: Optional[NormalizationCounters] = None,
) -> tuple[str, str, str]:
    if pd.isna(raw) or not raw:
        return ("UNKN", "Unknown", "Unknown")
    raw = raw.strip()
    if raw in aircraft_map:
        return aircraft_map[raw]
    if counter is not None:
        counter.aircraft_unknown += 1
    return ("UNKN", "Unknown", "Unknown")


def normalize_date(
    raw: str | None,
    counter: Optional[NormalizationCounters] = None,
) -> tuple[Optional[str], Optional[int]]:
    if pd.isna(raw) or not raw:
        return (None, None)
    raw = str(raw).strip()
    if len(raw) == 6 and raw.isdigit():
        year = int(raw[:4])
        month = int(raw[4:6])
        iso = f"{raw[:4]}-{raw[4:6]}-01"
        return (iso, year)
    if counter is not None:
        counter.date_invalid += 1
    return (None, None)


PHASE_MAP: dict[str, str] = {
    "takeoff": "Takeoff",
    "launch": "Takeoff",
    "initial climb": "Climb",
    "climb": "Climb",
    "cruise": "En Route",
    "descent": "Descent",
    "initial approach": "Approach",
    "final approach": "Approach",
    "approach": "Approach",
    "landing": "Landing",
    "taxi": "Taxi",
    "parked": "Parked",
    "parking": "Parked",
    "hovering": "Hover",
    "unknown": "Unknown",
    "other": "Unknown",
}


def normalize_phase(raw: str | None) -> list[str]:
    if pd.isna(raw) or not raw:
        return []
    raw_lower = raw.lower()
    candidates = [p.strip() for p in re.split(r"[;,]", raw)]
    results: list[str] = []
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        found = False
        for keyword, phase in PHASE_MAP.items():
            if keyword.lower() in c.lower():
                if phase not in results:
                    results.append(phase)
                found = True
                break
        if not found:
            if "Unknown" not in results:
                results.append("Unknown")
    return results


def normalize_anomaly(
    raw: str | None,
    expand_fn,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    if pd.isna(raw) or not raw:
        return ([], [])
    parts = [p.strip() for p in raw.split(";")]
    return expand_fn(parts, strict=strict)


def normalize_conditions(
    raw: str | None,
    counter: Optional[NormalizationCounters] = None,
) -> Optional[str]:
    if pd.isna(raw) or not raw:
        return None
    raw = raw.strip().upper()
    if raw in ("VMC", "IMC", "MIXED", "MARGINAL"):
        return raw
    if counter is not None:
        counter.conditions_unknown += 1
    return None


def normalize_flight_plan(
    raw: str | None,
    counter: Optional[NormalizationCounters] = None,
) -> Optional[str]:
    if pd.isna(raw) or not raw:
        return None
    raw = raw.strip().upper()
    if raw in ("IFR", "VFR", "SVFR", "DVFR"):
        return raw
    if counter is not None:
        counter.flight_plan_unknown += 1
    return None


def normalize_problem(raw: str | None) -> Optional[str]:
    if pd.isna(raw) or not raw:
        return None
    return raw.strip()


def normalize_factors(raw: str | None) -> Optional[list[str]]:
    if pd.isna(raw) or not raw:
        return None
    return [f.strip() for f in raw.split(";")]


def normalize_operator(raw: str | None) -> Optional[str]:
    if pd.isna(raw) or not raw:
        return None
    return raw.strip()


def normalize_state(
    raw: str | None,
    counter: Optional[NormalizationCounters] = None,
) -> Optional[str]:
    if pd.isna(raw) or not raw:
        return None
    s = raw.strip().upper()
    if len(s) == 2 or s == "US":
        return s
    if counter is not None:
        counter.state_unknown += 1
    return None


def normalize_text(text: str | None) -> str:
    if pd.isna(text) or not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text
