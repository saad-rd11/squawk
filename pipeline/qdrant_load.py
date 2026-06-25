"""
QdrantLocal collection management for Stage 3 pool.

Usage:
    mgr = Stage3Collection(path="./qdrant_storage")
    mgr.create(recreate=False)
    mgr.upsert_children(points, dense_vecs, sparse_vecs)
    mgr.close()
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    HnswConfigDiff,
    PointStruct,
    SparseVector,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    Distance,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "asrs_pool"


class Stage3Collection:
    def __init__(self, path: str = "./qdrant_storage"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.path))

    def create(self, recreate: bool = False, dense_dim: Optional[int] = None):
        if dense_dim is None:
            raise ValueError(
                "dense_dim is required — pass the embedder's output dimension"
            )

        if self.client.collection_exists(COLLECTION_NAME):
            if recreate:
                self.client.delete_collection(COLLECTION_NAME)
                logger.info("Deleted existing collection %s", COLLECTION_NAME)
            else:
                info = self.client.get_collection(COLLECTION_NAME)
                logger.info(
                    "Collection %s exists (%d points)",
                    COLLECTION_NAME,
                    info.points_count,
                )
                return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=dense_dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    )
                ),
            },
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
        )
        logger.info("Created collection %s", COLLECTION_NAME)

        # Payload indexes
        for field in [
            "aircraft_models",
            "aircraft_family",
            "flight_phase",
            "anomaly",
            "year",
            "state",
            "operator",
        ]:
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=f"payload.{field}",
                field_type="keyword",
            )
        logger.info("Created payload indexes")

    def upsert_children(
        self,
        points_data: list[dict],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict[str, float]],
    ):
        assert len(points_data) == len(dense_vectors) == len(sparse_vectors)

        batch: list[PointStruct] = []
        for pd_data, dense, sparse in zip(points_data, dense_vectors, sparse_vectors):
            point_id = uuid.uuid5(uuid.NAMESPACE_DNS, pd_data["id"])
            payload = pd_data["payload"]

            # Qdrant handles list fields natively — keyword filters match individual elements
            clean_payload = {k: v for k, v in payload.items() if v is not None}

            batch.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense,
                        "sparse": SparseVector(
                            indices=list(sparse.keys()),
                            values=list(sparse.values()),
                        ),
                    },
                    payload=clean_payload,
                )
            )

            if len(batch) >= 256:
                self.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch,
                    wait=True,
                )
                batch = []

        if batch:
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
                wait=True,
            )

        logger.info("Upserted %d child points", len(points_data))

    def close(self):
        self.client.close()

    def point_count(self) -> int:
        info = self.client.get_collection(COLLECTION_NAME)
        return info.points_count
