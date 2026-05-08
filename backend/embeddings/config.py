MODEL_NAME = "BAAI/bge-base-en-v1.5"
COLLECTION_NAME = "legal_chunks"

CHUNKS_PATH = "chunking/data/chunks.jsonl"

BATCH_SIZE = 64

# Qdrant
QDRANT_PATH = "./qdrant_db"
DISTANCE = "Cosine"