"""
Collision gate for text normalizations.

Any PR that introduces or changes a text normalizer must pass:
    collision_count(normalizer_fn, test_corpus) == 0

A count > 0 means the normalizer destroys protected codes (aircraft models,
ADREP categories, flight phases) or high-frequency narrative jargon. The
gate forces an explicit, measured decision before any destructive transform
can land.

The current normalizer (normalize_text, whitespace-only) passes: 0 collisions.
"""

import re
from typing import Callable

import pandas as pd

from .normalize import PHASE_MAP
from anomaly_map import ASRS_TO_ADREP

_CODE_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{2,5}\b")


def build_artifact_codes(
    aircraft_map_path: str,
) -> set[str]:
    """Canonical codes from pipeline artifacts that normalization must not destroy.

    Includes aircraft model codes from aircraft_map, ADREP codes from
    anomaly_map, and flight phase values from PHASE_MAP. These are the
    tokens the pipeline uses for structured-field lookups — any normalizer
    that changes them produces silent lookup failures.
    """
    with open(aircraft_map_path) as f:
        import json

        raw = json.load(f)

    codes: set[str] = set()

    for key, (model, family, mfr) in raw.items():
        codes.add(key.upper())
        if model and model not in ("UNKN", "NONE", ""):
            codes.add(model.upper())
        if family and family not in ("Unknown", "None", ""):
            codes.add(family.upper())
        if mfr and mfr not in ("Unknown", "None", ""):
            codes.add(mfr.upper())

    for asrs_key, (adrep_code, plain_text) in ASRS_TO_ADREP.items():
        codes.add(adrep_code.upper())

    for phase_val in PHASE_MAP.values():
        for word in phase_val.upper().split():
            if len(word) >= 3:
                codes.add(word)

    return codes


def scan_corpus_tokens(
    df: pd.DataFrame,
    min_frequency: int = 50,
) -> dict[str, int]:
    """Scan narratives and synopses for uppercase code-shaped tokens.

    Returns tokens appearing at least `min_frequency` times, sorted by
    frequency descending. These are the operational-acronym tokens (ATC,
    ILS, TCAS, ZZZ, etc.) that aren't in the artifact set but would be
    corrupted by a broad destructive normalizer.
    """
    counts: dict[str, int] = {}
    for col in ("Narrative", "Synopsis"):
        if col not in df.columns:
            continue
        for text in df[col].dropna():
            for m in _CODE_PATTERN.finditer(str(text)):
                t = m.group()
                counts[t] = counts.get(t, 0) + 1
    return {
        t: c
        for t, c in sorted(counts.items(), key=lambda x: -x[1])
        if c >= min_frequency
    }


def build_test_corpus(
    artifact_codes: set[str],
    corpus_tokens: dict[str, int],
) -> set[str]:
    """Union of artifact codes and high-frequency corpus tokens.

    This is the test corpus the collision gate checks against. The artifact
    codes are non-negotiable (they must survive any normalizer). The
    high-frequency corpus tokens catch broad-impact changes to operational
    jargon even when those terms aren't in the structured pipeline.
    """
    return artifact_codes | set(corpus_tokens.keys())


def collision_count(
    normalizer_fn: Callable[[str], str],
    tokens: set[str],
) -> int:
    """Count how many tokens in the set are changed by the normalizer.

    A count > 0 means the normalizer destructively transforms at least one
    protected or high-frequency token. The gate asserts == 0.
    """
    count = 0
    for t in tokens:
        if normalizer_fn(t) != t:
            count += 1
    return count
