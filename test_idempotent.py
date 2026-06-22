"""
Idempotency test: run the pipeline twice, assert byte-identical output.

Run with: python -m pytest test_idempotent.py -v
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pipeline.config import DEFAULT_CONFIG
from pipeline.run import run


def test_phase_map_ordering():
    """PHASE_MAP keys must be ordered so longer strings precede substring matches.
    E.g. 'initial climb' must appear before 'climb' so the longer string matches first.
    Without this ordering, 'climb' would match inside 'initial climb' and the
    more specific phase would never be reached."""
    from pipeline.normalize import PHASE_MAP

    keys = list(PHASE_MAP.keys())
    for short in keys:
        for long in keys:
            if short != long and short.lower() in long.lower():
                assert keys.index(long) < keys.index(short), (
                    f"'{long}' must appear before '{short}' in PHASE_MAP "
                    f"to prevent substring match on '{short}'"
                )


def test_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        config1 = DEFAULT_CONFIG
        config2 = DEFAULT_CONFIG

        out1 = os.path.join(tmp, "run1.jsonl")
        out2 = os.path.join(tmp, "run2.jsonl")

        config1 = config1.__class__(
            data_path=config1.data_path,
            aircraft_map_path=config1.aircraft_map_path,
            generic_descriptors_path=config1.generic_descriptors_path,
            output_path=out1,
            chunk_max_chars=config1.chunk_max_chars,
            dense_model=config1.dense_model,
            sparse_model=config1.sparse_model,
        )
        config2 = config2.__class__(
            data_path=config2.data_path,
            aircraft_map_path=config2.aircraft_map_path,
            generic_descriptors_path=config2.generic_descriptors_path,
            output_path=out2,
            chunk_max_chars=config2.chunk_max_chars,
            dense_model=config2.dense_model,
            sparse_model=config2.sparse_model,
        )

        run(config1)
        run(config2)

        h1 = hashlib.sha256()
        h2 = hashlib.sha256()
        with open(out1, "rb") as f:
            h1.update(f.read())
        with open(out2, "rb") as f:
            h2.update(f.read())

        assert h1.hexdigest() == h2.hexdigest(), (
            f"Outputs differ between runs\n  run1: {h1.hexdigest()}\n  run2: {h2.hexdigest()}"
        )
        print(f"✓ Idempotent: both runs produce same hash ({h1.hexdigest()[:16]}...)")


def test_three_way_decomposition_stable():
    """Re-run coverage and assert the three-way split matches verified values."""
    import tempfile
    from pathlib import Path

    from pipeline.normalize import load_generic_descriptors
    from pipeline.run import load_data, run
    from pipeline.metrics import compute_aircraft_coverage

    with tempfile.TemporaryDirectory() as tmp:
        config = DEFAULT_CONFIG
        config = config.__class__(
            data_path=config.data_path,
            aircraft_map_path=config.aircraft_map_path,
            generic_descriptors_path=config.generic_descriptors_path,
            output_path=os.path.join(tmp, "test.jsonl"),
            chunk_max_chars=config.chunk_max_chars,
        )
        metrics = run(config)
        ac = metrics["aircraft_coverage"]
        assert ac["resolved"] == 7328, f"resolved: {ac['resolved']} != 7328"
        assert ac["partial_map"] == 1153, f"partial_map: {ac['partial_map']} != 1153"
        assert ac["generic_descriptor"] == 6508, (
            f"generic_descriptor: {ac['generic_descriptor']} != 6508"
        )
        assert ac["null"] == 15, f"null: {ac['null']} != 15"
        assert ac["sum_check"] == 15004, f"sum_check: {ac['sum_check']} != 15004"
        print("✓ Three-way decomposition stable (7328 / 1153 / 6508 / 15)")
