from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    data_path: str = "nasa_asrs_2020_2022.csv"
    aircraft_map_path: str = "pipeline/data/aircraft_map.json"
    generic_descriptors_path: str = "pipeline/data/generic_descriptors.json"
    icao_designators_path: str = "pipeline/data/icao_designators.json"
    output_path: str = "points.jsonl"

    chunk_max_chars: int = 1024
    prefix_fields: tuple[str, ...] = ("aircraft", "phase", "anomaly", "component")

    dense_model: str = "BAAI/bge-base-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    record_dense_model: Optional[str] = (
        None  # deferred — wire when parent-similarity search exists
    )
    embed_batch_size: int = 256


DEFAULT_CONFIG = PipelineConfig()
