from pathlib import Path
from pypdf import PdfReader


def load_text_file(file_path: str) -> str:
    """Load text from a .txt file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return path.read_text(encoding="utf-8")


def load_pdf_file(file_path: str) -> str:
    """Extract text from a PDF file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(str(path))

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def load_document(file_path: str) -> str:
    """Load a supported medical document."""
    path = Path(file_path)

    if path.suffix.lower() == ".txt":
        return load_text_file(file_path)

    if path.suffix.lower() == ".pdf":
        return load_pdf_file(file_path)

    raise ValueError(
        f"Unsupported file type: {path.suffix}. "
        "Supported types are .txt and .pdf."
    )
