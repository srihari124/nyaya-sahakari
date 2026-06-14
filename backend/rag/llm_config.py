import os

# ──────────────────────────────────────────────────────────────────────────────
# LLM Provider Selection
#
# Change LLM_PROVIDER to switch between providers.
# Options: "ollama" | "anthropic"
# ──────────────────────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# ──────────────────────────────────────────────────────────────────────────────
# Ollama Settings (default provider)
# ──────────────────────────────────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "120"))
OLLAMA_MAX_CONCURRENT = int(os.getenv("OLLAMA_MAX_CONCURRENT", "1"))
OLLAMA_QUEUE_TIMEOUT_SEC = int(os.getenv("OLLAMA_QUEUE_TIMEOUT_SEC", "45"))
OLLAMA_HEALTHCHECK_ENABLED = os.getenv("OLLAMA_HEALTHCHECK_ENABLED", "1") == "1"
OLLAMA_HEALTHCHECK_TIMEOUT_SEC = int(os.getenv("OLLAMA_HEALTHCHECK_TIMEOUT_SEC", "2"))

# ──────────────────────────────────────────────────────────────────────────────
# Anthropic Settings
# ──────────────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_TIMEOUT_SEC = int(os.getenv("ANTHROPIC_TIMEOUT_SEC", "120"))
ANTHROPIC_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024"))
