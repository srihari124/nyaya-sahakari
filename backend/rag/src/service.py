import logging
import time
import hashlib
from typing import Dict, List, Optional

from retrieval.src.search import Retriever
from rag.src.llm import QwenLLM
from rag.src.prompt import get_system_prompt, build_user_prompt
from rag.src.rewrite import rewrite_query
from rag.src.reranker import reranker
from rag.src.validator import validator
from rag.src.confidence import confidence_scorer
from rag.src.policy import policy_engine

from rag.config import (
    MAX_CONTEXT_CHARS,
    RETRIEVAL_TOP_K,
    MAX_CHUNKS_PER_CASE,
    RERANK_CANDIDATES_K,
)

logger = logging.getLogger(__name__)


class RAGService:

    def __init__(self):
        self.retriever = Retriever()
        self.reranker = reranker
        self.llm = QwenLLM()
        logger.info("RAGService initialized")

    def preload(self):
        # Load heavy models during server startup.
        self.reranker.preload()
        logger.info("RAGService preload complete")

    def _stable_chunk_hash(self, item: Dict) -> str:
        meta = item.get("metadata", {}) or {}
        chunk_hash = meta.get("chunk_hash")
        if chunk_hash:
            return str(chunk_hash)
        text = (item.get("text") or "").strip().lower()
        normalized = " ".join(text.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _prepare_rerank_candidates(self, results: List[Dict]) -> List[Dict]:
        """Deduplicate and diversify candidates before reranking."""
        deduped: List[Dict] = []
        seen_hashes = set()

        for item in results:
            h = self._stable_chunk_hash(item)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            deduped.append(item)

        case_counts = {}
        diversified: List[Dict] = []
        for item in deduped:
            meta = item.get("metadata", {}) or {}
            case_id = meta.get("case_id") or meta.get("id") or "unknown_case"
            current = case_counts.get(case_id, 0)
            if current >= MAX_CHUNKS_PER_CASE:
                continue
            case_counts[case_id] = current + 1
            diversified.append(item)
            if len(diversified) >= RERANK_CANDIDATES_K:
                break

        logger.info(
            "Candidate prep: raw=%d deduped=%d diversified=%d per_case_cap=%d",
            len(results),
            len(deduped),
            len(diversified),
            MAX_CHUNKS_PER_CASE,
        )
        return diversified

    def _lowercase_first(self, text: str) -> str:
        if not text:
            return text
        return text[0].lower() + text[1:]

    def _apply_answer_tone(self, answer: str, response_mode: str) -> str:
        if not answer:
            return answer

        normalized = answer.strip().lower()
        if response_mode == "normal":
            return answer

        if response_mode == "cautious":
            prefix = "Based on the retrieved cases, "
            if normalized.startswith(prefix.strip().lower()):
                return answer
            return prefix + self._lowercase_first(answer)

        if response_mode == "limited":
            prefix = "The retrieved cases provide limited support, but they suggest that "
            if normalized.startswith("the retrieved cases provide limited support"):
                return answer
            return prefix + self._lowercase_first(answer)

        return answer

    def generate_answer(self, query: str) -> Dict:
        total_start = time.perf_counter()

        rewritten_query, filters = rewrite_query(query)

        retrieval_start = time.perf_counter()
        results = self.retriever.search(
            rewritten_query,
            top_k=RETRIEVAL_TOP_K,
            filters=filters if filters else None
        )
        retrieval_time = time.perf_counter() - retrieval_start
        logger.info("Retrieval time: %.3fs", retrieval_time)

        if not results:
            return {
                "answer": "No relevant cases found.",
                "sources": []
            }

        candidates = self._prepare_rerank_candidates(results)
        if not candidates:
            return {
                "answer": "No relevant cases found.",
                "sources": []
            }

        rerank_start = time.perf_counter()
        top_results = self.reranker.rerank_with_timeout(query, candidates, top_k=5)
        rerank_time = time.perf_counter() - rerank_start
        logger.info("Rerank time: %.3fs", rerank_time)

        contexts: List[str] = []
        total_context_chars = 0

        for i, r in enumerate(top_results):
            meta = r["metadata"]

            context = f"""
CASE {i+1}:
Court: {meta.get("court", "Unknown")}
Year: {meta.get("year", "Unknown")}
Outcome: {meta.get("outcome", "Unknown")}

Text:
{r["text"]}
"""
            context_len = len(context)
            if total_context_chars + context_len > MAX_CONTEXT_CHARS:
                break
            contexts.append(context)
            total_context_chars += context_len

        system_prompt = get_system_prompt()
        user_prompt = build_user_prompt(query, contexts)

        llm_start = time.perf_counter()
        answer = self.llm.generate(
            system_prompt,
            user_prompt
        )

        llm_time = time.perf_counter() - llm_start
        logger.info("LLM generation time: %.3fs", llm_time)

        validation_start = time.perf_counter()
        validation = validator.validate(
            query=query,
            contexts=contexts,
            answer=answer
        )
        validation_time = time.perf_counter() - validation_start

        logger.info(
            "Validation time: %.3fs | grounded=%s | score=%.3f",
            validation_time,
            validation["supported"],
            validation["grounding_score"]
        )

        rerank_scores = [
            r.get("rerank_score", 0.0)
            for r in top_results
        ]

        confidence = confidence_scorer.compute(
            validation=validation,
            rerank_scores=rerank_scores
        )

        logger.info(
            "Confidence score: %.3f (%s)",
            confidence["score"],
            confidence["level"]
        )

        decision = policy_engine.evaluate(
            validation=validation,
            confidence=confidence
        )

        logger.info(
            "Policy decision: accepted=%s reason=%s",
            decision["accepted"],
            decision["reason"]
        )

        if not decision["accepted"]:
            logger.warning(
                "Answer rejected by policy engine"
            )

            answer = (
                "Unable to generate a sufficiently grounded legal answer from the retrieved cases."
            )
        else:
            answer = self._apply_answer_tone(
                answer=answer,
                response_mode=decision.get("response_mode", "normal")
            )
        total_time = time.perf_counter() - total_start

        logger.info(
            "Total request time: %.3fs",
            total_time
        )
        return {
            "answer": answer,
            "sources": top_results,
            "validation": validation,
            "confidence": confidence,
            "decision": decision,
            "meta": {
                "rewritten_query": rewritten_query,
                "filters": filters,
                "retrieval_time": round(retrieval_time, 3),
                "rerank_time": round(rerank_time, 3),
                "llm_time": round(llm_time, 3),
                "total_time": round(total_time, 3),
            }
        }


_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def preload_rag_service() -> RAGService:
    service = get_rag_service()
    service.preload()
    return service


def generate_answer(query: str) -> Dict:
    return get_rag_service().generate_answer(query)
