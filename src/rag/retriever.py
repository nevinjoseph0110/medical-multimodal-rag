from typing import List

from .embeddings import MedicalEmbedder
from .vector_store import FAISSVectorStore


class MedicalRetriever:
    """Retrieve relevant medical knowledge using embeddings and FAISS."""

    def __init__(
        self,
        embedder: MedicalEmbedder,
        vector_store: FAISSVectorStore,
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[str]:
        """Retrieve the most relevant medical chunks for a query."""

        if not query or not query.strip():
            return []

        query_embedding = self.embedder.embed_text(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        return results
