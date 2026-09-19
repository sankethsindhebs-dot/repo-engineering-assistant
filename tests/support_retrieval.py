"""Deterministic test vectors; these do not establish real model relevance."""

import hashlib
from dataclasses import asdict
import json
from pathlib import Path

from backend.llm import EmbeddingIdentity
from backend.ingestion.chunks import document_chunks, python_chunks
from backend.ingestion.documents import read_documents
from backend.ingestion.repository import parse_repository


FIXTURE = Path(__file__).resolve().parent / "fixtures/retrieval_repo"


def unit_vector(axis: int) -> list[float]:
    vector = [0.0] * 768
    vector[axis] = 1.0
    return vector


class DeterministicProvider:
    def __init__(self, digest: str = "a" * 64):
        self.digest = digest
        self.calls: list[list[str]] = []
        self.hook = None

    async def embedding_identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity("ollama", "nomic-embed-text:latest", self.digest)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.hook:
            self.hook()
        vectors = []
        for text in texts:
            if text.startswith("search_query: "):
                axis = 2 if "documentation" in text else 0
            elif "README.md" in text:
                axis = 2
            elif "Function: entry.handle_request\n" in text:
                axis = 0
            elif "Function: worker.transform\n" in text:
                axis = 1
            else:
                axis = 3 + int(hashlib.sha256(text.encode()).hexdigest()[:6], 16) % 700
            vectors.append(unit_vector(axis))
        return vectors

    async def generate(self, *args, **kwargs):
        raise AssertionError("Generation must not be used by retrieval/indexing")

    async def aclose(self):
        pass


class OfflineGraph:
    """A fixed parsed graph for unit tests; live tests exercise the real database."""
    def __init__(self):
        self.snapshot_value = parse_repository(FIXTURE, "fixture")
        documents = read_documents(FIXTURE, "fixture")
        self.rows = [asdict(c) for c in python_chunks(self.snapshot_value) + document_chunks(documents)]
        refs = self.snapshot_value.references
        self.owner_rows = [asdict(entity) | {"owner_kind": "entity", "references_json": json.dumps([
            asdict(ref) for ref in refs if ref.owner_id == entity.id])} for entity in self.snapshot_value.entities]
        self.owner_rows += [asdict(doc) | {"owner_kind": "document"} for doc in documents.items]
        self.edges = []
        ref_map = {ref.id: ref for ref in refs}
        for edge in self.snapshot_value.relationships:
            row = asdict(edge)
            if edge.reference_id:
                ref = asdict(ref_map[edge.reference_id])
                row.update({key: ref[key] for key in ["expression", "path", "start_line", "end_line", "start_column", "end_column"]})
            self.edges.append(row)
        self.state_value = {"retrieval_schema_version": 1, "retrieval_state": "UNINDEXED",
                            "snapshot_hash": self.snapshot_value.digest, "source_revision": 1, "data_revision": 1,
                            "documentation_hash": documents.digest, "documentation_status": "current"}

    def state(self, repository_id):
        return dict(self.state_value)

    def owners(self, repository_id):
        return self.owner_rows

    def chunks(self, repository_id):
        return self.rows

    def adjacent(self, repository_id, entity_id, step, limit, timeout):
        rows = []
        source, target = ("source_id", "target_id") if step.direction == "outgoing" else ("target_id", "source_id")
        for edge in self.edges:
            if edge["kind"] == step.relationship and edge[source] == entity_id:
                rows.append({"target": edge[target], "edge": edge})
        return sorted(rows, key=lambda row: (row["target"], row["edge"]["id"]))[:limit]

    def entity(self, name):
        return next(owner for owner in self.owner_rows if owner.get("qualified_name") == name)
