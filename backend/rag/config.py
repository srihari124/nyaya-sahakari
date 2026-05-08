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

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "120"))
OLLAMA_MAX_CONCURRENT = int(os.getenv("OLLAMA_MAX_CONCURRENT", "1"))
OLLAMA_QUEUE_TIMEOUT_SEC = int(os.getenv("OLLAMA_QUEUE_TIMEOUT_SEC", "45"))

# Run a quick health check before generation.
OLLAMA_HEALTHCHECK_ENABLED = os.getenv("OLLAMA_HEALTHCHECK_ENABLED", "1") == "1"
OLLAMA_HEALTHCHECK_TIMEOUT_SEC = int(os.getenv("OLLAMA_HEALTHCHECK_TIMEOUT_SEC", "2"))
