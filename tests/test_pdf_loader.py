from pathlib import Path

import pytest
from langchain_community.document_loaders import PyPDFLoader


PDF_PATH = Path("../documents/dummy_file.pdf")
EXPECTED_PAGES = 4
EXPECTED_TEXT = (
    "Lá dentro, a única luz vinha do brilho esverdeado e fosfórico de um monitor"
)


def load_pdf(path: Path):
    return PyPDFLoader(str(path)).load()


def test_pdf_file_exists():
    assert PDF_PATH.exists() == True


def test_pdf_number_of_pages():
    docs = PyPDFLoader(str(PDF_PATH)).load()

    assert len(docs) == EXPECTED_PAGES


def test_pdf_contains_expected_text():
    docs = load_pdf(PDF_PATH)

    full_text = "\n".join(doc.page_content for doc in docs)

    assert EXPECTED_TEXT in full_text
