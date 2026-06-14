import requests
import logging
import threading
import time
from typing import List

from rag.llm_config import (
    OLLAMA_MODEL,
    OLLAMA_URL,
    OLLAMA_TIMEOUT_SEC,
    OLLAMA_MAX_CONCURRENT,
    OLLAMA_QUEUE_TIMEOUT_SEC,
    OLLAMA_HEALTHCHECK_ENABLED,
    OLLAMA_HEALTHCHECK_TIMEOUT_SEC,
)

logger = logging.getLogger(__name__)


class OllamaLLM:
    _guard_lock = threading.Lock()
    _semaphore = None

    def __init__(self, model: str = OLLAMA_MODEL, timeout: int = OLLAMA_TIMEOUT_SEC):
        self.model = model
        self.base_url = OLLAMA_URL.rstrip("/")
        self.url = f"{self.base_url}/api/generate"
        self.timeout = timeout
        self.max_concurrent = OLLAMA_MAX_CONCURRENT
        self.queue_timeout = OLLAMA_QUEUE_TIMEOUT_SEC
        self.healthcheck_enabled = OLLAMA_HEALTHCHECK_ENABLED
        self.healthcheck_timeout = OLLAMA_HEALTHCHECK_TIMEOUT_SEC
        self._ensure_semaphore()
        self.model = self._resolve_model(self.model)
        logger.info("Ollama model resolved: %s", self.model)
        logger.info(
            "Ollama concurrency guard enabled: max_concurrent=%s queue_timeout=%ss",
            self.max_concurrent,
            self.queue_timeout,
        )

    def _ensure_semaphore(self):
        if OllamaLLM._semaphore is not None:
            return
        with OllamaLLM._guard_lock:
            if OllamaLLM._semaphore is None:
                OllamaLLM._semaphore = threading.BoundedSemaphore(self.max_concurrent)

    def health_check(self) -> bool:
        if not self.healthcheck_enabled:
            return True
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.healthcheck_timeout)
            response.raise_for_status()
            return True
        except Exception:
            return False

    def _available_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception:
            return []

    def _resolve_model(self, requested_model: str) -> str:
        available = self._available_models()
        if not available:
            return requested_model
        if requested_model in available:
            return requested_model
        return available[0]

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        queue_start = time.perf_counter()
        acquired = OllamaLLM._semaphore.acquire(timeout=self.queue_timeout)
        queue_wait = time.perf_counter() - queue_start

        if not acquired:
            logger.warning(
                "Ollama queue timeout after %.3fs (max_concurrent=%s)",
                queue_wait,
                self.max_concurrent,
            )
            return (
                "LLM Error: server is busy (generation queue timeout). "
                "Please retry in a few seconds."
            )

        logger.info("Ollama queue wait time: %.3fs", queue_wait)
        if not self.health_check():
            logger.warning("Ollama health check failed before generation; attempting generate anyway")

        full_prompt = f"""<System>
{system_prompt.strip()}
</System>

<User>
{user_prompt.strip()}
</User>
"""

        try:
            generate_start = time.perf_counter()
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            elapsed = time.perf_counter() - generate_start
            logger.info("Ollama generation completed in %.3fs", elapsed)
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"LLM Error: {str(e)}"
        finally:
            OllamaLLM._semaphore.release()
