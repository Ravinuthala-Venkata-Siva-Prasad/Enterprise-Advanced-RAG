import pytest

from pathlib import Path

from src.ingestion.txt_loader import TXTLoader
from src.ingestion.html_loader import HTMLLoader
from src.ingestion.csv_loader import CSVLoader
from src.ingestion.docx_loader import DOCXLoader
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.document_loader import DocumentLoader

from docx import Document as DocxDocument
from reportlab.pdfgen import canvas


def test_txt_loader(tmp_path):

    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "Enterprise RAG testing",
        encoding="utf-8"
    )

    documents = TXTLoader().load(file_path)

    assert len(documents) == 1
    assert "Enterprise RAG" in documents[0].page_content
    assert documents[0].metadata["file_type"] == "txt"


def test_html_loader(tmp_path):

    file_path = tmp_path / "sample.html"

    file_path.write_text(
        "<html><title>Test</title><body>"
        "<h1>RAG Platform</h1>"
        "<script>ignore this</script>"
        "</body></html>",
        encoding="utf-8"
    )

    documents = HTMLLoader().load(file_path)

    assert len(documents) == 1
    assert "RAG Platform" in documents[0].page_content
    assert "ignore this" not in documents[0].page_content


def test_csv_loader(tmp_path):

    file_path = tmp_path / "sample.csv"

    file_path.write_text(
        "name,department\n"
        "Alice,AI\n"
        "Bob,Engineering\n",
        encoding="utf-8"
    )

    documents = CSVLoader().load(file_path)

    assert len(documents) == 2
    assert "Alice" in documents[0].page_content
    assert documents[0].metadata["row_number"] == 2


def test_docx_loader(tmp_path):

    file_path = tmp_path / "sample.docx"

    document = DocxDocument()

    document.add_paragraph(
        "Enterprise RAG DOCX test"
    )

    document.save(file_path)

    documents = DOCXLoader().load(file_path)

    assert len(documents) == 1
    assert "DOCX test" in documents[0].page_content


def test_pdf_loader(tmp_path):

    file_path = tmp_path / "sample.pdf"

    pdf = canvas.Canvas(str(file_path))

    pdf.drawString(
        100,
        750,
        "Enterprise RAG PDF test"
    )

    pdf.save()

    documents = PDFLoader().load(file_path)

    assert len(documents) == 1
    assert "PDF test" in documents[0].page_content


def test_unsupported_file(tmp_path):

    file_path = tmp_path / "sample.xyz"

    file_path.write_text("Unsupported format")

    loader = DocumentLoader()

    with pytest.raises(ValueError):

        loader.load_file(file_path)


def test_missing_file():

    with pytest.raises(FileNotFoundError):

        TXTLoader().load("missing.txt")