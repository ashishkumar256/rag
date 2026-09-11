"""
ingest.py - Module for loading, cleaning, and chunking PDF documents.
"""

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_clean_pdf(
    pdf_path: str, header_margin: float = 50.0, footer_margin: float = 50.0
) -> list[Document]:
    """
    Loads a PDF and crops out top/bottom margins to remove headers and footers.
    """
    doc = pymupdf.open(pdf_path)
    documents = []

    for page_num, page in enumerate(doc):
        rect = page.rect
        crop_box = pymupdf.Rect(
            rect.x0, rect.y0 + header_margin, rect.x1, rect.y1 - footer_margin
        )

        text = page.get_text("text", clip=crop_box).strip()

        if text:  # Avoid appending empty pages
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf_path, "page": page_num + 1},
                )
            )

    return documents


def chunk_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    """
    Splits loaded documents into smaller overlapping chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    return chunks
