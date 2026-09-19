"""Add source/document evidence and embeddings to an existing structural repository."""

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError

from backend.config import Settings
from backend.graph.evidence_store import EvidenceStoreError
from backend.llm import ProviderError
from backend.retrieval.indexing import index_repository


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True, help="Stable checkout including documentation")
    parser.add_argument("--repository-id", required=True)
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(index_repository(Settings(), args.repository, args.repository_id))
        print(json.dumps(asdict(result), sort_keys=True))
        return 0
    except (EvidenceStoreError, ProviderError, ValueError) as exc:
        message = "Invalid typed configuration" if isinstance(exc, ValidationError) else str(exc)
        print(f"FAIL indexing: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
