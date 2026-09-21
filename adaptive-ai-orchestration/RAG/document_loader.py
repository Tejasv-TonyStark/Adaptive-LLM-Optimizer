"""Page-preserving PDF ingestion. Run RAG.vector_store to embed and build."""
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
DOCUMENTS_DIR = Path(__file__).resolve().parents[1] / "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
def load_pdf(file_path):
    return "\n".join(page.extract_text() or "" for page in PdfReader(file_path).pages)
def load_all_documents(directory=None):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for path in sorted(Path(directory or DOCUMENTS_DIR).glob("*.pdf")):
        for page_number, page in enumerate(PdfReader(path).pages, 1):
            for n, text in enumerate(splitter.split_text(page.extract_text() or "")):
                chunks.append(dict(text=text, source=path.name, page=page_number,
                                   chunk_id=f"{path.stem}:p{page_number}:c{n}"))
    if not chunks:
        raise ValueError("No extractable PDF text found; scanned PDFs require OCR.")
    return chunks
if __name__ == "__main__":
    print(f"Loaded {len(load_all_documents())} chunks. Build with python -m RAG.vector_store")
