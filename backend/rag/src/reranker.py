import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import torch
from sentence_transformers import CrossEncoder

from rag.config import (
    RERANKER_MAX_TEXT_CHARS,
    RERANKER_BATCH_SIZE,
    RERANKER_TIMEOUT_SEC,
    RERANKER_DEVICE,
    RERANKER_MODEL_NAME,
)

torch.set_num_threads(2)
logger = logging.getLogger(__name__)


class Reranker:
    """Rerank retrieved chunks with the BGE reranker model."""

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model_name = model_name
        auto_device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = RERANKER_DEVICE if RERANKER_DEVICE in {"cpu", "mps"} else auto_device
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            try:
                self.model = CrossEncoder(
                    self.model_name,
                    local_files_only=True,
                    device=self.device
                )
            except Exception:
                self.model = CrossEncoder(self.model_name, device=self.device)
            logger.info("Reranker model loaded: %s on %s", self.model_name, self.device)

    def preload(self):
        self._ensure_model()

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """Return the top reranked results for a query."""
        if not results:
            return []

        self._ensure_model()

        pairs = [(query, (r["text"] or "")[:RERANKER_MAX_TEXT_CHARS]) for r in results]
        logger.info(
            "Reranker pairs=%d batch_size=%d max_text_chars=%d",
            len(pairs),
            RERANKER_BATCH_SIZE,
            RERANKER_MAX_TEXT_CHARS,
        )

        scores = self.model.predict(
            pairs,
            batch_size=RERANKER_BATCH_SIZE,
            show_progress_bar=False
        )

        for r, score in zip(results, scores):
            r["rerank_score"] = float(score)

        reranked = sorted(
            results,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]

    def rerank_with_timeout(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 5,
        timeout_sec: int = RERANKER_TIMEOUT_SEC,
    ) -> List[Dict]:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.rerank, query, results, top_k)
            try:
                return future.result(timeout=timeout_sec)
            except FuturesTimeoutError:
                logger.warning("Reranker timeout after %ss, fallback to retrieval order", timeout_sec)
                return results[:top_k]


reranker = Reranker()
