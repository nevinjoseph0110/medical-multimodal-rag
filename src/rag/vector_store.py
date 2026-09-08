from pathlib import Path
from typing import List

import faiss
import numpy as np


class FAISSVectorStore:
    """FAISS vector store for medical knowledge retrieval."""

    def __init__(self, dimension: int, index_path: str = "data/vector_store"):
        self.dimension = dimension
        self.index_path = Path(index_path)

        self.index_path.mkdir(parents=True, exist_ok=True)

        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[str] = []

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]]
    ) -> None:
        """Add documents and their embeddings to the FAISS index."""

        if not documents or not embeddings:
            return

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings."
            )

        vectors = np.asarray(embeddings, dtype="float32")

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"but received {vectors.shape[1]}."
            )

        self.index.add(vectors)
        self.documents.extend(documents)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[str]:
        """Return the most relevant documents for a query."""

        if self.index.ntotal == 0:
            return []

        query_vector = np.asarray(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_vector,
            min(top_k, self.index.ntotal)
        )

        results = []

        for index in indices[0]:
            if index != -1:
                results.append(self.documents[index])

        return results

    def save(self) -> None:
        """Save the FAISS index to disk."""

        faiss.write_index(
            self.index,
            str(self.index_path / "medical.index")
        )

        documents_file = self.index_path / "documents.npy"

        np.save(
            documents_file,
            np.array(self.documents, dtype=object),
            allow_pickle=True
        )

    def load(self) -> None:
        """Load an existing FAISS index from disk."""

        index_file = self.index_path / "medical.index"
        documents_file = self.index_path / "documents.npy"

        if not index_file.exists() or not documents_file.exists():
            raise FileNotFoundError(
                "FAISS vector store files were not found."
            )

        self.index = faiss.read_index(str(index_file))

        self.documents = np.load(
            documents_file,
            allow_pickle=True
        ).tolist()
