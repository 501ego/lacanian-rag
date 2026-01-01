import json
from pathlib import Path

import pytest

from app.infrastructure import text_extractor


def test_strip_accents_and_normalize_month():
    assert text_extractor._strip_accents("février") == "fevrier"
    assert text_extractor._normalize_month("Fevrier") == "février"


def test_fix_ocr_digits():
    assert text_extractor._fix_ocr_digits("l23") == "123"
    assert text_extractor._fix_ocr_digits("0l") == "01"


def test_infer_year():
    assert text_extractor._infer_year("1964", 1960) == 1964
    assert text_extractor._infer_year("64", 1963) == 1964
    assert text_extractor._infer_year("64", None) is None
    assert text_extractor._infer_year("xx", 1960) is None


def test_extract_seminar_start_year():
    assert text_extractor._extract_seminar_start_year("1953-54") == 1953
    assert text_extractor._extract_seminar_start_year("no year") is None


def test_build_date_to_label():
    text = "Leçon 1 5 janvier 1964\nLeçon 2 6 février 1964"
    mapping, default_label = text_extractor.build_date_to_label(text)
    assert default_label == "Leçon"
    assert "05_janvier_1964" in mapping


def test_split_by_body_dates():
    text = "lundi 3 janvier 1964 Foo. mardi 4 janvier 1964 Bar."
    segments = text_extractor.split_by_body_dates(text)
    assert len(segments) == 2
    assert segments[0][0] is not None


def test_clean_text():
    cleaned = text_extractor.PDFChunker.clean_text("Page 1\nfoo  --  bar [12]")
    assert "Page" not in cleaned
    assert "[" not in cleaned


def test_chunk_text_with_pages():
    chunker = text_extractor.PDFChunker(Path("in"), Path("out"), chunk_size=3, overlap=1)
    text = "[PAGE_1] one two three four"
    chunks = chunker.chunk_text_with_pages(text)
    assert chunks
    assert chunks[0]["pages"]


def test_process_pdf_to_json(tmp_path, monkeypatch):
    class DummyPage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class DummyReader:
        def __init__(self, _path):
            self.pages = [DummyPage("Leçon 1 5 janvier 1964")]

    monkeypatch.setattr(text_extractor, "PdfReader", DummyReader)
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    pdf_path = input_dir / "Seminar.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    chunker = text_extractor.PDFChunker(input_dir, output_dir, chunk_size=3, overlap=1)
    chunker.process_pdf_to_json(pdf_path)

    output_file = output_dir / "Seminar.json"
    assert output_file.exists()
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert isinstance(payload, list)


def test_text_chunker_overlapping():
    chunker = text_extractor.TextChunker(chunk_size=3, overlap=1)
    chunks = chunker.chunk_text("one two three four five")
    assert len(chunks) >= 2


def test_text_chunker_invalid_overlap():
    chunker = text_extractor.TextChunker(chunk_size=2, overlap=2)
    with pytest.raises(ValueError):
        chunker.chunk_text("one two three")
