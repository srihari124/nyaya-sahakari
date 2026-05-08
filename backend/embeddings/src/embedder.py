import logging

import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(2)
logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        # Prefer local cache first so pipelines work offline when models
        # have already been downloaded in previous runs.
        try:
            self.model = SentenceTransformer(
                model_name,
                local_files_only=True,
                device=self.device
            )
        except Exception:
            self.model = SentenceTransformer(model_name, device=self.device)
        logger.info("Embedder model loaded: %s on %s", model_name, self.device)

    def encode_documents(self, texts, batch_size=64):
        """
        For document embeddings
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )

    def encode_query(self, query: str):
        """
        For query embeddings (important for BGE models)
        """
        query = f"query: {query}"

        return self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]
