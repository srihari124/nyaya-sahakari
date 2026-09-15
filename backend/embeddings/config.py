import os

MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "BAAI/bge-base-en-v1.5")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_chunks")
CHUNKS_PATH = os.getenv("EMBEDDINGS_CHUNKS_PATH", "chunking/data/chunks.jsonl")
BATCH_SIZE = int(os.getenv("EMBEDDINGS_BATCH_SIZE", "64"))
QDRANT_PATH = os.getenv("EMBEDDINGS_QDRANT_PATH", "./qdrant_db")
DISTANCE = os.getenv("EMBEDDINGS_DISTANCE", "Cosine")
