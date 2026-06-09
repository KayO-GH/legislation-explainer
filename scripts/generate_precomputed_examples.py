"""Generate bundled precomputed assets for example bills."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, UTC
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_QWEN_MODEL,
)
from services.example_bills import EXAMPLE_BILL_MANIFEST_PATH, ExampleBill, load_example_bills
from services.ingest import fetch_url_content
from services.precomputed_assets import write_precomputed_asset
from services.providers import get_default_api_key, instantiate_client
from services.rag_pipeline import document_hash_for_text, generate_analysis_once, split_into_chunks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bill-id",
        action="append",
        dest="bill_ids",
        help="Limit generation to one or more example bill ids.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even when a bill already has bundled assets.",
    )
    parser.add_argument(
        "--fetch-timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds when downloading source bills.",
    )
    return parser.parse_args()


def _load_manifest() -> list[dict[str, object]]:
    return json.loads(EXAMPLE_BILL_MANIFEST_PATH.read_text(encoding="utf-8"))


def _write_manifest(items: list[dict[str, object]]) -> None:
    EXAMPLE_BILL_MANIFEST_PATH.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")


def _has_complete_assets(bill: ExampleBill) -> bool:
    return (
        bill.analysis_path.exists()
        and bill.document_path.exists()
        and bill.chunks_path.exists()
        and bill.metadata_path.exists()
    )


def _metadata(document_hash: str, source_url: str) -> dict[str, object]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "provider": DEFAULT_PROVIDER,
        "model": DEFAULT_QWEN_MODEL,
        "source_url": source_url,
        "document_hash": document_hash,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
    }


def main() -> int:
    args = _parse_args()
    manifest_items = _load_manifest()
    manifest_by_id = {str(item["id"]): item for item in manifest_items}
    all_bills = load_example_bills()
    selected = [bill for bill in all_bills if not args.bill_ids or bill.id in set(args.bill_ids)]

    if args.bill_ids:
        missing = sorted(set(args.bill_ids) - {bill.id for bill in selected})
        if missing:
            raise SystemExit(f"Unknown bill ids: {', '.join(missing)}")

    api_key = get_default_api_key(DEFAULT_PROVIDER)
    if not api_key:
        raise SystemExit("HF_TOKEN is required to generate precomputed example analysis.")
    provider_client = instantiate_client(DEFAULT_PROVIDER, api_key)

    failures: list[tuple[str, str]] = []
    for bill in selected:
        if _has_complete_assets(bill) and not args.force:
            print(f"Skipping {bill.id}: bundled assets already exist.")
            continue

        try:
            print(f"Generating assets for {bill.id}...")
            document_text = fetch_url_content(bill.source_url, timeout_seconds=args.fetch_timeout)
            chunks = split_into_chunks(document_text)
            analysis = generate_analysis_once(provider_client, document_text)

            document_hash = document_hash_for_text(document_text)
            write_precomputed_asset(
                bill,
                analysis_payload=analysis.model_dump(mode="json"),
                document_text=document_text,
                chunks=chunks,
                metadata=_metadata(document_hash, bill.source_url),
            )

            manifest_entry = manifest_by_id[bill.id]
            manifest_entry["document_hash"] = document_hash
            print(f"Finished {bill.id}.")
        except Exception as exc:  # noqa: BLE001
            failures.append((bill.id, str(exc)))
            print(f"Failed {bill.id}: {exc}")

    _write_manifest(manifest_items)
    if failures:
        print("\nFailures:")
        for bill_id, message in failures:
            print(f"- {bill_id}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
