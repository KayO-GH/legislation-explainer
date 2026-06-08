"""Shared example bill manifest helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_BILL_ASSETS_ROOT = PROJECT_ROOT / "assets" / "example_bills"
EXAMPLE_BILL_MANIFEST_PATH = EXAMPLE_BILL_ASSETS_ROOT / "manifest.json"


@dataclass(frozen=True)
class ExampleBill:
    id: str
    title: str
    source_url: str
    document_hash: str
    asset_dir: str

    @property
    def root_dir(self) -> Path:
        return EXAMPLE_BILL_ASSETS_ROOT / self.asset_dir

    @property
    def analysis_path(self) -> Path:
        return self.root_dir / "analysis.json"

    @property
    def document_path(self) -> Path:
        return self.root_dir / "document.txt"

    @property
    def chunks_path(self) -> Path:
        return self.root_dir / "chunks.json"

    @property
    def metadata_path(self) -> Path:
        return self.root_dir / "metadata.json"

    @property
    def vector_store_dir(self) -> Path:
        return self.root_dir / "vector_store"


def normalize_example_url(url: str) -> str:
    stripped = url.strip()
    parsed = urlsplit(stripped)
    normalized_path = parsed.path.rstrip("/") or parsed.path
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), normalized_path, parsed.query, ""))


@lru_cache(maxsize=1)
def load_example_bills() -> tuple[ExampleBill, ...]:
    payload = json.loads(EXAMPLE_BILL_MANIFEST_PATH.read_text(encoding="utf-8"))
    return tuple(ExampleBill(**item) for item in payload)


def reload_example_bills() -> tuple[ExampleBill, ...]:
    load_example_bills.cache_clear()
    return load_example_bills()


def example_bill_titles() -> list[str]:
    return [bill.title for bill in load_example_bills()]


def example_bill_urls_by_title() -> dict[str, str]:
    return {bill.title: bill.source_url for bill in load_example_bills()}


def example_bill_by_title(title: str) -> ExampleBill | None:
    for bill in load_example_bills():
        if bill.title == title:
            return bill
    return None


def example_bill_by_url(url: str | None) -> ExampleBill | None:
    if not url:
        return None
    normalized = normalize_example_url(url)
    for bill in load_example_bills():
        if normalize_example_url(bill.source_url) == normalized:
            return bill
    return None
