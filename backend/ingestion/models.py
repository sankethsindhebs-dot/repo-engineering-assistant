"""Deterministic structural records; these describe syntax, not runtime behaviour."""

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal


Kind = Literal["Module", "Class", "Function", "Method"]
Relation = Literal["CONTAINS", "IMPORTS", "CALLS", "INHERITS"]


def stable_id(*parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Entity:
    id: str
    repository_id: str
    path: str
    kind: Kind
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    source_hash: str
    source: str
    module_name: str


@dataclass(frozen=True)
class Reference:
    """One explicit AST call, imported name, or class base (resolved or unresolved)."""

    id: str
    owner_id: str
    kind: Literal["CALLS", "IMPORTS", "INHERITS"]
    path: str
    expression: str
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    target_id: str | None = None
    reason: str = "unresolved"


@dataclass(frozen=True)
class Relationship:
    id: str
    source_id: str
    target_id: str
    kind: Relation
    reference_id: str | None = None


@dataclass(frozen=True)
class Snapshot:
    repository_id: str
    source_root: str
    entities: tuple[Entity, ...]
    relationships: tuple[Relationship, ...]
    references: tuple[Reference, ...]
    skipped_paths: tuple[str, ...] = ()

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def validate(self) -> None:
        """Reject inconsistent snapshots before any database operation."""
        ids = {entity.id for entity in self.entities}
        if len(ids) != len(self.entities) or not ids:
            raise ValueError("Snapshot must contain unique entities and at least one Python file")
        if any(entity.repository_id != self.repository_id for entity in self.entities):
            raise ValueError("Snapshot contains another repository")
        refs = {reference.id: reference for reference in self.references}
        if len(refs) != len(self.references):
            raise ValueError("Duplicate reference IDs")
        for reference in self.references:
            if reference.owner_id not in ids or (
                reference.target_id is not None and reference.target_id not in ids
            ):
                raise ValueError("Reference endpoint is missing")
        if len({edge.id for edge in self.relationships}) != len(self.relationships):
            raise ValueError("Duplicate relationship IDs")
        resolved = set()
        for edge in self.relationships:
            if edge.source_id not in ids or edge.target_id not in ids:
                raise ValueError("Relationship endpoint is missing")
            if edge.kind == "CONTAINS":
                if edge.reference_id is not None:
                    raise ValueError("Containment must not have a reference")
                continue
            reference = refs.get(edge.reference_id)
            if reference is None or (reference.owner_id, reference.target_id, reference.kind) != (
                edge.source_id, edge.target_id, edge.kind
            ):
                raise ValueError("Relationship does not match its source evidence")
            resolved.add(reference.id)
        if resolved != {ref.id for ref in self.references if ref.target_id is not None}:
            raise ValueError("Resolved reference is missing its relationship")
