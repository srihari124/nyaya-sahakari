import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


def get_client(path):
    return QdrantClient(path=path)


def create_collection(client, collection_name, vector_size):

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )


def upload_points(client, collection_name, vectors, chunks):
    """
    Upload embeddings with payload
    """

    points = []
    seen_hashes = set()

    for vector, chunk in zip(vectors, chunks):
        metadata = chunk.get("metadata", {}) or {}
        chunk_text = chunk.get("text", "")
        case_id = metadata.get("id")
        chunk_index = metadata.get("chunk_index")

        # Stable hash from chunk metadata/text when not already provided.
        chunk_hash = metadata.get("chunk_hash")
        if not chunk_hash:
            base = f"{case_id}|{chunk_index}|{chunk_text}".encode("utf-8")
            chunk_hash = hashlib.sha256(base).hexdigest()

        # Dedupe within current upload batch by chunk_hash.
        if chunk_hash in seen_hashes:
            continue
        seen_hashes.add(chunk_hash)

        # Deterministic numeric point id from hash so reruns overwrite.
        point_id = int(chunk_hash[:16], 16)

        payload = {
            "text": chunk_text,
            "case_id": case_id,
            "chunk_index": chunk_index,
            "chunk_hash": chunk_hash,
            **metadata
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )


def create_payload_indexes(client, collection_name):
    """
    Create indexes for filtering
    """

    fields = [
        ("court", "keyword"),
        ("year", "keyword"),
        ("outcome", "keyword"),
        ("crime_type", "keyword"),
        ("case_id", "keyword"),
        ("chunk_hash", "keyword"),
        ("chunk_index", "integer"),
    ]

    for field, schema in fields:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=schema
        )
