import os
import atexit
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from embeddings.src.embedder import Embedder
from embeddings.config import *

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self):
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        qdrant_path = os.path.join(workspace_root, "embeddings", "qdrant_db")
        self.client = QdrantClient(path=qdrant_path)
        atexit.register(self.close)
        self.embedder = Embedder(MODEL_NAME)
        logger.info("Retriever initialized with Qdrant path: %s", qdrant_path)

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict = None
    ):
        query_vector = self.embedder.encode_query(query)

        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )

            qdrant_filter = Filter(must=conditions)

        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter
            )
        else:
            response = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k,
                query_filter=qdrant_filter
            )
            results = getattr(response, "points", None) or getattr(response, "result", []) or []

        output = []
        for r in results:
            output.append({
                "score": r.score,
                "text": r.payload["text"],
                "metadata": r.payload
            })

        return output
