from pathlib import Path
from typing import List, Tuple

from .document_loader import load_document
from .chunker import chunk_text
from .embeddings import MedicalEmbedder
from .vector_store import FAISSVectorStore


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def collect_documents(data_dir: str) -> List[Path]:
    """Find supported documents in the knowledge-base directory."""
    directory = Path(data_dir)

    if not directory.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}"
        )

    return [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def prepare_chunks(
    data_dir: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> Tuple[List[str], List[str]]:
    """Load documents and split them into chunks with source metadata."""

    documents = collect_documents(data_dir)

    all_chunks = []
    sources = []

    for document in documents:
        text = load_document(str(document))

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in chunks:
            all_chunks.append(chunk)
            sources.append(str(document))

    return all_chunks, sources


def build_vector_store(
    data_dir: str = "data/raw",
    index_path: str = "data/vector_store"
) -> None:
    """Build and save the FAISS medical knowledge store."""

    chunks, sources = prepare_chunks(data_dir)

    if not chunks:
        raise ValueError(
            "No supported documents were found in the data directory."
        )

    embedder = MedicalEmbedder()

    embeddings = embedder.embed_documents(chunks)

    dimension = len(embeddings[0])

    vector_store = FAISSVectorStore(
        dimension=dimension,
        index_path=index_path
    )

    vector_store.add_documents(
        documents=chunks,
        embeddings=embeddings
    )

    vector_store.save()

    print(f"Processed {len(sources)} source documents/chunks.")
    print(f"Created {len(chunks)} text chunks.")
    print(f"Saved FAISS vector store to: {index_path}")


if __name__ == "__main__":
    build_vector_store()
