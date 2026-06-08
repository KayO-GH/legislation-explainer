from __future__ import annotations

from pathlib import Path

import pytest

from services import ingest


ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


@pytest.mark.skipif(not (ASSETS_DIR / "NITA-2008-act-2025-1.pdf").exists(), reason="Sample PDF missing")
def test_extract_text_from_pdf_returns_content():
    pdf_path = ASSETS_DIR / "NITA-2008-act-2025-1.pdf"
    data = pdf_path.read_bytes()
    text = ingest.extract_text_from_pdf(data)
    assert "National Information Technology" in text


def test_fetch_url_content_monkeypatched(monkeypatch):
    html_body = """<html><head><title>Bill</title></head><body><p>Main insight.</p></body></html>"""

    class DummyResponse:
        def __init__(self, text: str) -> None:
            self.text = text
            self.content = text.encode("utf-8")
            self.headers = {"content-type": "text/html"}

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, timeout: int = 0):  # noqa: ARG001
        return DummyResponse(html_body)

    monkeypatch.setattr("requests.get", fake_get)

    text = ingest.fetch_url_content("https://example.com")
    assert "Main insight." in text
    assert "Bill" in text


def test_fetch_url_content_handles_pdf_url(monkeypatch):
    pdf_path = ASSETS_DIR / "NITA-2008-act-2025-1.pdf"
    if not pdf_path.exists():
        pytest.skip("Sample PDF missing")

    class DummyResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content
            self.text = content.decode("utf-8", errors="ignore")
            self.headers = {"content-type": "application/pdf"}

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, timeout: int = 0):  # noqa: ARG001
        return DummyResponse(pdf_path.read_bytes())

    monkeypatch.setattr("requests.get", fake_get)

    text = ingest.fetch_url_content("https://example.com/sample.pdf")
    assert "National Information Technology" in text


def test_ingest_combines_sources():
    combined = ingest.combine_sources(["First part", "Second part"])
    assert "First part" in combined and "Second part" in combined
