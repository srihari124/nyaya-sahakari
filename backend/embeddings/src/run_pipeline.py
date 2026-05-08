import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from embeddings.config import *
from embeddings.src.embedder import Embedder
from embeddings.src.qdrant_store import (
    get_client,
    create_collection,
    upload_points,
    create_payload_indexes
)


def load_chunks(path):
    chunks = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def run():
    print("Loading chunks...")
    chunks_path = os.path.join(WORKSPACE_ROOT, CHUNKS_PATH)
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found at {chunks_path}")

    texts = [c["text"] for c in chunks]

    print("Initializing embedder...")
    embedder = Embedder(MODEL_NAME)

    print("Generating embeddings...")
    vectors = embedder.encode_documents(
        texts,
        batch_size=BATCH_SIZE
    )

    print("Setting up Qdrant...")
    qdrant_path = os.path.join(PROJECT_ROOT, QDRANT_PATH)
    client = get_client(qdrant_path)

    create_collection(
        client,
        COLLECTION_NAME,
        vector_size=len(vectors[0])
    )

    print("Uploading embeddings...")
    upload_points(
        client,
        COLLECTION_NAME,
        vectors,
        chunks
    )

    print("Creating payload indexes...")
    create_payload_indexes(client, COLLECTION_NAME)

    print("\nEmbedding pipeline completed successfully")


if __name__ == "__main__":
    run()
