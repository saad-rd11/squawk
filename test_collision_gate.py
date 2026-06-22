"""
Collision gate tests.

The primary test (test_normalize_text_zero_collisions) asserts that the
current whitespace-only normalizer destroys zero protected tokens. This is
the gate: any PR that introduces or changes a text normalizer must
demonstrate that the new normalizer also passes collision_count == 0, or
that any collisions are explicitly understood and signed off.

Run with:
    python -m pytest test_collision_gate.py -v
"""

import os
import tempfile

import pandas as pd

from pipeline.config import DEFAULT_CONFIG
from pipeline.collision_gate import (
    build_artifact_codes,
    build_test_corpus,
    collision_count,
    scan_corpus_tokens,
)
from pipeline.normalize import normalize_text
from pipeline.run import load_data


def _load_test_data():
    """Load a representative subset of narratives for corpus token scan."""
    config = DEFAULT_CONFIG
    df = load_data(config.data_path)
    return df, config


def test_normalize_text_zero_collisions():
    """Gate: current normalizer must not destroy any protected or
    high-frequency tokens. Failure here means a destructive transform
    was introduced without updating the collision gate."""
    df, config = _load_test_data()

    artifact_codes = build_artifact_codes(config.aircraft_map_path)
    corpus_tokens = scan_corpus_tokens(df, min_frequency=50)
    corpus = build_test_corpus(artifact_codes, corpus_tokens)

    count = collision_count(normalize_text, corpus)

    assert count == 0, (
        f"normalize_text produces {count} token collisions — "
        f"it destructively transforms {count} protected or high-frequency tokens. "
        f"Roll back or update the gate."
    )
