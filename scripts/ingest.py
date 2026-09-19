"""Parse a Python source tree, then explicitly replace its structural Neo4j snapshot."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError

from backend.config import Settings
from backend.graph.store import GraphWriteError, write_snapshot
from backend.ingestion.documents import read_documents
from backend.ingestion.repository import IngestionError, parse_repository


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("--repository-id", required=True, help="Stable logical ID; reusing it replaces that graph")
    parser.add_argument("--source-root", default=".", help="One Python import root relative to repository (e.g. src)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report counts without opening Neo4j")
    parser.add_argument("--timeout-seconds", type=float, default=60, help="Graph transaction timeout (max 600)")
    args = parser.parse_args(argv)
    try:
        snapshot = parse_repository(args.repository, args.repository_id, args.source_root)
        documents = read_documents(args.repository, args.repository_id)
        if args.dry_run:
            result = {
                "mode": "parse_only", "repository_id": snapshot.repository_id,
                "snapshot_hash": snapshot.digest, "entities": len(snapshot.entities),
                "relationships": len(snapshot.relationships),
                "unresolved_references": sum(ref.target_id is None for ref in snapshot.references),
            }
        else:
            result = asdict(write_snapshot(Settings(), snapshot, args.timeout_seconds, documents=documents))
            result["mode"] = "persisted"
        result["skipped_paths"] = snapshot.skipped_paths
        result["documents"] = len(documents.items)
        result["documentation_hash"] = documents.digest
        print(json.dumps(result, sort_keys=True))
        return 0
    except (IngestionError, GraphWriteError, ValidationError, ValueError) as exc:
        message = "Invalid typed application configuration" if isinstance(exc, ValidationError) else str(exc)
        print(f"FAIL ingestion: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
