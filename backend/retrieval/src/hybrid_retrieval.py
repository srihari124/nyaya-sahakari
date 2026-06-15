import os
import logging
import atexit
from typing import Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi

from embeddings.src.embedder import Embedder
from embeddings.config import COLLECTION_NAME, MODEL_NAME
from rag.src.reranker import reranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Dense + BM25 hybrid search with RRF fusion and BGE reranking."""

    def __init__(self):
        workspace_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        qdrant_path = os.path.join(workspace_root, "embeddings", "qdrant_db")
        self.client = QdrantClient(path=qdrant_path)
        atexit.register(self.close)
        self.embedder = Embedder(MODEL_NAME)

        self._corpus_ids, self._corpus_texts, self._corpus_payloads = self._load_corpus()
        self._bm25 = self._build_bm25(self._corpus_texts)

        logger.info(
            "HybridRetriever initialized: corpus_size=%d",
            len(self._corpus_ids),
        )

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass

    def _load_corpus(self) -> Tuple[List, List[str], List[Dict]]:
        """Scroll all Qdrant points to build the BM25 corpus at startup."""
        ids: List = []
        texts: List[str] = []
        payloads: List[Dict] = []
        offset = None

        while True:
            records, offset = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                ids.append(record.id)
                texts.append(record.payload.get("text", "") or "")
                payloads.append(record.payload)

            if offset is None:
                break

        logger.info("Loaded %d chunks from Qdrant for BM25 index", len(ids))
        return ids, texts, payloads

    def _build_bm25(self, texts: List[str]) -> BM25Okapi:
        """Tokenize corpus and build BM25Okapi index."""
        tokenized = [text.lower().split() for text in texts]
        return BM25Okapi(tokenized)

    def dense_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Semantic search via Qdrant dense vectors."""
        query_vector = self.embedder.encode_query(query)

        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
            )
        else:
            response = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
            )
            results = getattr(response, "points", None) or getattr(response, "result", []) or []

        return [
            {"score": r.score, "text": r.payload["text"], "metadata": r.payload}
            for r in results
        ]

    def keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """BM25 keyword search over the corpus loaded at startup."""
        scores = self._bm25.get_scores(query.lower().split())
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [
            {
                "score": float(scores[idx]),
                "text": self._corpus_texts[idx],
                "metadata": self._corpus_payloads[idx],
                "_corpus_idx": idx,
            }
            for idx in top_indices
        ]

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        keyword_results: List[Dict],
        k: int = 60,
        dense_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ) -> List[Dict]:
        """Combine dense and keyword result lists using Reciprocal Rank Fusion."""
        fused_scores: Dict[str, float] = {}
        result_map: Dict[str, Dict] = {}

        def _chunk_key(result: Dict) -> str:
            meta = result.get("metadata", {}) or {}
            chunk_hash = meta.get("chunk_hash")
            if chunk_hash:
                return str(chunk_hash)
            return (result.get("text", "") or "")[:128]

        for rank, result in enumerate(dense_results):
            key = _chunk_key(result)
            fused_scores[key] = fused_scores.get(key, 0.0) + dense_weight / (k + rank + 1)
            result_map[key] = result

        for rank, result in enumerate(keyword_results):
            key = _chunk_key(result)
            fused_scores[key] = fused_scores.get(key, 0.0) + keyword_weight / (k + rank + 1)
            if key not in result_map:
                result_map[key] = result

        fused = []
        for key in sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True):
            item = result_map[key].copy()
            item["rrf_score"] = round(fused_scores[key], 6)
            item.pop("_corpus_idx", None)
            fused.append(item)

        return fused

    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank fused results using the BGE cross-encoder."""
        return reranker.rerank_with_timeout(query, results, top_k=top_k)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Full pipeline: dense + BM25 → RRF fusion → BGE reranker."""
        dense_results = self.dense_search(query, top_k=top_k, filters=filters)
        keyword_results = self.keyword_search(query, top_k=top_k)
        fused = self.reciprocal_rank_fusion(dense_results, keyword_results)
        reranked = self.rerank(query, fused, top_k=top_k)

        logger.info(
            "HybridSearch: dense=%d bm25=%d fused=%d reranked=%d",
            len(dense_results),
            len(keyword_results),
            len(fused),
            len(reranked),
        )
        return reranked


_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever
