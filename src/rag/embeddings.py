from typing import List

from sentence_transformers import SentenceTransformer


class MedicalEmbedder:
    """Create embeddings for medical knowledge-base text."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        """Convert a single text into an embedding vector."""
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Convert multiple text chunks into embedding vectors."""
        if not documents:
            return []

        embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings.tolist()
