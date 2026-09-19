"""Explicit, repository-scoped structural writes. Readiness remains read-only."""

from collections import defaultdict
from dataclasses import asdict, dataclass
import json

from neo4j import GraphDatabase, ManagedTransaction, Query, unit_of_work
from neo4j.exceptions import DriverError, Neo4jError

from backend.config import NEO4J_VERSION, Settings
from backend.ingestion.documents import Documents
from backend.ingestion.models import Snapshot


SCHEMA = (
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT ingestion_repository IF NOT EXISTS FOR (n:IngestionState) REQUIRE n.repository_id IS UNIQUE",
    "CREATE INDEX entity_repository IF NOT EXISTS FOR (n:Entity) ON (n.repository_id)",
)
LABELS = {"Module": "File:Module", "Class": "Class", "Function": "Function", "Method": "Function:Method"}
RELATIONS = ("CONTAINS", "IMPORTS", "CALLS", "INHERITS")


class GraphWriteError(RuntimeError):
    """Credential-free write failure; the previous snapshot remains on rollback."""


@dataclass(frozen=True)
class WriteResult:
    repository_id: str
    snapshot_hash: str
    entities: int
    relationships: int
    unresolved_references: int


def _replace_snapshot(tx: ManagedTransaction, snapshot: Snapshot, documents: Documents | None = None,
                      chunk_bytes: int = 1536) -> None:
    from backend.graph.evidence_store import reconcile_evidence
    parameters = {"repository_id": snapshot.repository_id}
    # SET obtains a write lock on the one metadata node before reading/deleting
    # entities. The uniqueness constraint also serializes first-time ingestion.
    previous = tx.run(
        "MERGE (s:IngestionState {repository_id: $repository_id}) "
        "SET s.lock_version = coalesce(s.lock_version, 0) + 1 RETURN properties(s) AS state", **parameters,
    ).single(strict=True)["state"]
    tx.run("MATCH (n:Entity {repository_id: $repository_id}) DETACH DELETE n", **parameters).consume()
    references = defaultdict(list)
    for reference in snapshot.references:
        references[reference.owner_id].append(asdict(reference))
    for kind, labels in LABELS.items():
        rows = []
        for entity in snapshot.entities:
            if entity.kind == kind:
                values = asdict(entity)
                values["references_json"] = json.dumps(references[entity.id], sort_keys=True, separators=(",", ":"))
                rows.append(values)
        if rows:
            tx.run(f"UNWIND $rows AS row CREATE (n:Entity:{labels}) SET n = row", rows=rows).consume()
    reference_map = {reference.id: reference for reference in snapshot.references}
    for kind in RELATIONS:
        rows = []
        for edge in snapshot.relationships:
            if edge.kind == kind:
                row = asdict(edge)
                row["repository_id"] = snapshot.repository_id
                if edge.reference_id:
                    reference = reference_map[edge.reference_id]
                    row.update(path=reference.path, start_line=reference.start_line,
                               end_line=reference.end_line, start_column=reference.start_column,
                               end_column=reference.end_column, expression=reference.expression,
                               resolution=reference.reason)
                rows.append(row)
        if rows:
            record = tx.run(
                "UNWIND $rows AS row "
                "MATCH (a:Entity {id: row.source_id, repository_id: row.repository_id}) "
                "MATCH (b:Entity {id: row.target_id, repository_id: row.repository_id}) "
                f"CREATE (a)-[r:{kind}]->(b) SET r = row RETURN count(r) AS created",
                rows=rows,
            ).single(strict=True)
            if record["created"] != len(rows):
                raise GraphWriteError("Structural relationship count mismatch")
    tx.run(
        "MATCH (s:IngestionState {repository_id: $repository_id}) "
        "SET s.snapshot_hash = $digest, s.schema_version = 1, s.source_root = $source_root, "
        "s.skipped_paths = $skipped_paths REMOVE s.lock_version",
        **parameters, digest=snapshot.digest, source_root=snapshot.source_root,
        skipped_paths=list(snapshot.skipped_paths),
    ).consume()
    reconcile_evidence(tx, snapshot, documents, chunk_bytes, previous)


def write_snapshot(settings: Settings, snapshot: Snapshot, timeout_seconds: float = 60,
                   *, documents: Documents | None = None) -> WriteResult:
    from backend.graph.evidence_store import EvidenceStoreError, RETRIEVAL_SCHEMA
    snapshot.validate()
    if not 0 < timeout_seconds <= 600:
        raise ValueError("Ingestion timeout must be greater than zero and at most 600 seconds")
    if settings.neo4j_password is None:
        raise GraphWriteError("NEO4J_PASSWORD is not configured")
    try:
        with GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
            connection_timeout=settings.neo4j_timeout_seconds,
            connection_acquisition_timeout=settings.neo4j_timeout_seconds,
            max_transaction_retry_time=0,
        ) as driver:
            with driver.session(database=settings.neo4j_database) as session:
                record = session.run(Query(
                    "CALL dbms.components() YIELD versions RETURN versions[0] AS version",
                    timeout=settings.neo4j_timeout_seconds,
                )).single(strict=True)
                if record["version"] != NEO4J_VERSION:
                    raise GraphWriteError(f"Ingestion requires Neo4j {NEO4J_VERSION}")
                # Neo4j schema commands run separately from the atomic data transaction.
                for statement in (*SCHEMA, *RETRIEVAL_SCHEMA):
                    session.run(Query(statement, timeout=timeout_seconds)).consume()
                @unit_of_work(timeout=timeout_seconds)
                def work(tx: ManagedTransaction) -> None:
                    _replace_snapshot(tx, snapshot, documents, settings.retrieval_chunk_bytes)

                session.execute_write(work)
    except (DriverError, Neo4jError, OSError, EvidenceStoreError, ValueError) as exc:
        raise GraphWriteError(f"Neo4j ingestion failed ({type(exc).__name__}); inspect or retry the snapshot") from None
    return WriteResult(snapshot.repository_id, snapshot.digest, len(snapshot.entities),
                       len(snapshot.relationships), sum(ref.target_id is None for ref in snapshot.references))
