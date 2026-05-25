# rag/document_loader.py

import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ──────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────

DOCUMENTS_DIR = "documents"
CHUNK_SIZE    = 500    # characters per chunk
CHUNK_OVERLAP = 50     # overlap between chunks


def load_pdf(file_path: str) -> str:
    """
    Reads a single PDF file and returns full text.

    Args:
        file_path: path to PDF file

    Returns:
        str — full extracted text
    """
    reader   = PdfReader(file_path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    return full_text


def load_all_documents() -> list[dict]:
    """
    Loads all PDFs from the documents folder.
    Splits each into chunks for embedding.

    Returns:
        list of dicts with text, source, chunk_id
    """
    if not os.path.exists(DOCUMENTS_DIR):
        print(f"❌ Documents folder not found: {DOCUMENTS_DIR}")
        return []

    pdf_files = [
        f for f in os.listdir(DOCUMENTS_DIR)
        if f.endswith(".pdf")
    ]

    if not pdf_files:
        print(f"❌ No PDF files found in {DOCUMENTS_DIR}")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )

    all_chunks = []

    for pdf_file in pdf_files:
        file_path = os.path.join(DOCUMENTS_DIR, pdf_file)
        print(f"Loading: {pdf_file}")

        try:
            full_text = load_pdf(file_path)
            chunks    = splitter.split_text(full_text)

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text":     chunk,
                    "source":   pdf_file,
                    "chunk_id": f"{pdf_file}_{i}"
                })

            print(f"✅ {pdf_file} → {len(chunks)} chunks")

        except Exception as e:
            print(f"❌ Failed to load {pdf_file}: {e}")

    print(f"\nTotal chunks loaded: {len(all_chunks)}")
    return all_chunks


# ──────────────────────────────────────────
# TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Document Loader Test ──\n")
    chunks = load_all_documents()

    if chunks:
        print(f"\nSample chunk 1:")
        print(f"Source:   {chunks[0]['source']}")
        print(f"Chunk ID: {chunks[0]['chunk_id']}")
        print(f"Text:     {chunks[0]['text'][:200]}...")
        print("\n✅ Document loader working correctly!")
    else:
        print("❌ No chunks loaded")