"""Helpers for bundled example-bill analysis assets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.example_bills import ExampleBill, example_bill_by_url


@dataclass(frozen=True)
class PrecomputedExampleAsset:
    bill: ExampleBill
    analysis_payload: dict[str, Any]
    document_text: str
    chunks: list[str]
    metadata: dict[str, Any]


def load_precomputed_asset_for_url(url: str | None) -> PrecomputedExampleAsset | None:
    bill = example_bill_by_url(url)
    if bill is None:
        return None
    return load_precomputed_asset(bill)


def load_precomputed_asset(bill: ExampleBill) -> PrecomputedExampleAsset | None:
    if not bill.analysis_path.exists():
        return None
    if not bill.document_path.exists():
        return None
    if not bill.chunks_path.exists():
        return None
    if not bill.vector_store_dir.exists():
        return None

    analysis_payload = json.loads(bill.analysis_path.read_text(encoding="utf-8"))
    document_text = bill.document_path.read_text(encoding="utf-8")
    chunks = json.loads(bill.chunks_path.read_text(encoding="utf-8"))
    metadata = {}
    if bill.metadata_path.exists():
        metadata = json.loads(bill.metadata_path.read_text(encoding="utf-8"))

    return PrecomputedExampleAsset(
        bill=bill,
        analysis_payload=analysis_payload,
        document_text=document_text,
        chunks=chunks,
        metadata=metadata,
    )


def write_precomputed_asset(
    bill: ExampleBill,
    *,
    analysis_payload: dict[str, Any],
    document_text: str,
    chunks: list[str],
    metadata: dict[str, Any],
) -> None:
    bill.root_dir.mkdir(parents=True, exist_ok=True)
    bill.analysis_path.write_text(json.dumps(analysis_payload, indent=2), encoding="utf-8")
    bill.document_path.write_text(document_text, encoding="utf-8")
    bill.chunks_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    bill.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
