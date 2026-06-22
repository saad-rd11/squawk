from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    data_path: str = "nasa_asrs_2020_2022.csv"
    aircraft_map_path: str = "pipeline/data/aircraft_map.json"
    generic_descriptors_path: str = "pipeline/data/generic_descriptors.json"
    icao_designators_path: str = "pipeline/data/icao_designators.json"
    output_path: str = "points.jsonl"

    chunk_max_chars: int = 2048
    prefix_fields: tuple[str, ...] = ("aircraft", "phase", "anomaly", "component")

    dense_model: str = "__EMBEDDING_PLACEHOLDER__"
    sparse_model: str = "__SPARSE_PLACEHOLDER__"


DEFAULT_CONFIG = PipelineConfig()
