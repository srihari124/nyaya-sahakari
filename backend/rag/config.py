import os

MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))

# Fetch a wider candidate pool before reranking.
RETRIEVAL_TOP_K = int(os.getenv("RAG_RETRIEVAL_TOP_K", "20"))
# Cap per-case chunks before reranking for source diversity.
MAX_CHUNKS_PER_CASE = int(os.getenv("RAG_MAX_CHUNKS_PER_CASE", "3"))
# Limit the final candidate count sent to the reranker.
RERANK_CANDIDATES_K = int(os.getenv("RAG_RERANK_CANDIDATES_K", "12"))

# Cap reranker input size for long legal chunks.
RERANKER_MAX_TEXT_CHARS = int(os.getenv("RERANKER_MAX_TEXT_CHARS", "1600"))
# Smaller batches reduce memory spikes on Apple Silicon.
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "4"))
# Fall back to retrieval order if reranking exceeds the timeout.
RERANKER_TIMEOUT_SEC = int(os.getenv("RERANKER_TIMEOUT_SEC", "20"))
# Override the reranker device if needed.
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "").strip().lower()
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

VALIDATOR_MODEL_NAME = os.getenv("VALIDATOR_MODEL_NAME", "BAAI/bge-base-en-v1.5")
VALIDATOR_GROUNDING_THRESHOLD = float(os.getenv("VALIDATOR_GROUNDING_THRESHOLD", "0.65"))
VALIDATOR_MAX_RISK_SCORE = int(os.getenv("VALIDATOR_MAX_RISK_SCORE", "4"))
