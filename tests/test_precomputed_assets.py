from __future__ import annotations

import json

from services.example_bills import ExampleBill
from services.precomputed_assets import load_precomputed_asset
from services.rag_pipeline import hydrate_precomputed_example


def test_load_precomputed_asset_does_not_require_vector_store_dir(monkeypatch, tmp_path) -> None:
    from services import example_bills as example_bills_module

    monkeypatch.setattr(example_bills_module, "EXAMPLE_BILL_ASSETS_ROOT", tmp_path)
    bill = ExampleBill(
        id="example-bill",
        title="Example Bill",
        source_url="https://example.com/bill.pdf",
        document_hash="hash",
        asset_dir="example-bill",
    )
    root_dir = tmp_path / bill.asset_dir
    root_dir.mkdir(parents=True)
    (root_dir / "analysis.json").write_text(json.dumps({"executive_summary": "Summary"}), encoding="utf-8")
    (root_dir / "document.txt").write_text("Document text", encoding="utf-8")
    (root_dir / "chunks.json").write_text(json.dumps(["Chunk one", "Chunk two"]), encoding="utf-8")

    asset = load_precomputed_asset(bill)

    assert asset is not None
    assert asset.document_text == "Document text"
    assert asset.chunks == ["Chunk one", "Chunk two"]


def test_hydrate_precomputed_example_rebuilds_vector_store_from_chunks(monkeypatch, tmp_path) -> None:
    from services import example_bills as example_bills_module

    monkeypatch.setattr(example_bills_module, "EXAMPLE_BILL_ASSETS_ROOT", tmp_path)
    bill = ExampleBill(
        id="example-bill",
        title="Example Bill",
        source_url="https://example.com/bill.pdf",
        document_hash="hash",
        asset_dir="example-bill",
    )
    root_dir = tmp_path / bill.asset_dir
    root_dir.mkdir(parents=True)
    (root_dir / "analysis.json").write_text(json.dumps({"executive_summary": "Summary"}), encoding="utf-8")
    (root_dir / "document.txt").write_text("Document text", encoding="utf-8")
    (root_dir / "chunks.json").write_text(json.dumps(["Chunk one", "Chunk two"]), encoding="utf-8")

    asset = load_precomputed_asset(bill)
    assert asset is not None

    built_chunks: list[list[str]] = []

    monkeypatch.setattr("services.rag_pipeline.build_vector_store", lambda chunks: built_chunks.append(list(chunks)) or object())

    artifacts = hydrate_precomputed_example(asset)

    assert built_chunks == [["Chunk one", "Chunk two"]]
    assert artifacts.vector_store is not None
    assert artifacts.document_text == "Document text"
