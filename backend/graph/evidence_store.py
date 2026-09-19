"""Repository-scoped evidence writes and reads for Neo4j 5.26.30."""

from dataclasses import asdict, fields
import json
import math
import time
from typing import Callable, TypeVar

from neo4j import GraphDatabase, ManagedTransaction, Query, unit_of_work
from neo4j.exceptions import DriverError, Neo4jError

from backend.config import NEO4J_VERSION, Settings
from backend.graph.store import SCHEMA
from backend.ingestion.chunks import Chunk, document_chunks, policy_id, python_chunks
from backend.ingestion.documents import Documents
from backend.ingestion.models import Entity, Reference, Relationship, Snapshot, stable_id
from backend.retrieval.models import EmbeddingProfile, Readiness, TraversalStep


RETRIEVAL_SCHEMA = (
    "CREATE CONSTRAINT rea_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE INDEX rea_chunk_repository IF NOT EXISTS FOR (c:Chunk) ON (c.repository_id)",
    "CREATE CONSTRAINT rea_document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE INDEX rea_document_repository IF NOT EXISTS FOR (d:Document) ON (d.repository_id)",
)
T = TypeVar("T")


class EvidenceStoreError(RuntimeError):
    """A safe database or publication diagnostic."""


class ConcurrentUpdate(EvidenceStoreError):
    """A source revision or indexing attempt was superseded."""


def revision_token(state: dict) -> tuple:
    return (state.get("data_revision", 0), state.get("snapshot_hash"),
            state.get("documentation_hash"), state.get("documentation_status"))


def effective_readiness(state: dict, profile_id: str | None = None) -> Readiness:
    if not state.get("retrieval_schema_version"):
        return Readiness(state="UNINDEXED", reasons=("retrieval_not_initialized",))
    reasons = []
    for current, published in (
        ("source_revision", "published_source_revision"),
        ("snapshot_hash", "published_snapshot_hash"),
        ("documentation_hash", "published_documentation_hash"),
    ):
        if state.get(current) is None or state.get(current) != state.get(published):
            reasons.append(current + "_not_published")
    if state.get("documentation_status") != "current":
        reasons.append("documentation_needs_refresh")
    if profile_id is not None and state.get("published_profile_id") != profile_id:
        reasons.append("embedding_profile_mismatch")
    if not state.get("published_profile_id"):
        reasons.append("embedding_profile_missing")
    stored = state.get("retrieval_state", "UNINDEXED")
    status = stored if stored in {"UNINDEXED", "DIRTY", "BUILDING", "READY", "FAILED"} else "DIRTY"
    if reasons and status == "READY":
        status = "DIRTY"
    return Readiness(state=status, reasons=tuple(reasons))


def validate_vectors(vectors: list[list[float]], expected_count: int) -> None:
    if len(vectors) != expected_count:
        raise ValueError("Embedding response count does not match input count")
    for vector in vectors:
        if len(vector) != 768 or any(isinstance(value, bool) or not isinstance(value, (int, float))
                                     or not math.isfinite(value) for value in vector):
            raise ValueError("Retrieval embeddings must contain exactly 768 finite numeric values")
        norm = math.hypot(*vector)
        if norm == 0 or not math.isfinite(norm):
            raise ValueError("Retrieval embedding norm must be finite and nonzero")


def lock_state(tx: ManagedTransaction, repository_id: str) -> dict:
    record = tx.run(
        "MATCH (s:IngestionState {repository_id:$repository_id}) "
        "SET s.lock_version = coalesce(s.lock_version,0)+1 RETURN properties(s) AS state",
        repository_id=repository_id,
    ).single()
    if record is None:
        raise EvidenceStoreError("Repository has no structural snapshot")
    return record["state"]


def reconcile_evidence(tx: ManagedTransaction, snapshot: Snapshot, documents: Documents | None,
                       max_bytes: int, previous: dict) -> None:
    """Runs inside the structural write or additive upgrade transaction, after locking."""
    repository_id = snapshot.repository_id
    if documents is not None and documents.repository_id != repository_id:
        raise ValueError("Documentation belongs to another repository")
    chunks = python_chunks(snapshot, max_bytes)
    if documents is not None:
        chunks += document_chunks(documents, max_bytes)
    docs_hash = documents.digest if documents is not None else previous.get("documentation_hash")
    docs_status = "current" if documents is not None else "needs_refresh"
    changed = (previous.get("snapshot_hash") != snapshot.digest
               or previous.get("documentation_hash") != docs_hash
               or previous.get("documentation_status") != docs_status
               or previous.get("chunk_policy_id") != policy_id(max_bytes))
    revision = previous.get("source_revision", 0) + int(changed or not previous.get("retrieval_schema_version"))
    state = "DIRTY" if changed else previous.get("retrieval_state", "UNINDEXED")
    if not previous.get("published_profile_id"):
        state = "UNINDEXED"
    if documents is not None:
        rows = [asdict(item) | {"source_revision": revision} for item in documents.items]
        tx.run("MATCH (d:Document {repository_id:$repository_id}) WHERE NOT d.id IN $ids DETACH DELETE d",
               repository_id=repository_id, ids=[row["id"] for row in rows]).consume()
        tx.run("UNWIND $rows AS row MERGE (d:Document {id:row.id}) SET d += row", rows=rows).consume()
    tx.run(
        "MATCH (c:Chunk {repository_id:$repository_id}) "
        "WHERE ($complete OR c.owner_kind = 'entity') AND NOT c.id IN $ids DETACH DELETE c",
        repository_id=repository_id, complete=documents is not None, ids=[chunk.id for chunk in chunks],
    ).consume()
    for kind, label in (("entity", "Entity"), ("document", "Document")):
        rows = [asdict(chunk) | {"source_revision": revision,
                "evidence_id": stable_id("chunk-evidence-v1", chunk.id, chunk.owner_id)}
                for chunk in chunks if chunk.owner_kind == kind]
        record = tx.run(
            f"UNWIND $rows AS row MATCH (o:{label} {{id:row.owner_id, repository_id:row.repository_id}}) "
            "MERGE (c:Chunk {id:row.id}) SET c += row "
            "MERGE (c)-[r:EVIDENCE_FOR]->(o) "
            "SET r.id=row.evidence_id, r.repository_id=row.repository_id RETURN count(c) AS count",
            rows=rows,
        ).single(strict=True)
        if record["count"] != len(rows):
            raise EvidenceStoreError("Evidence owner/link count mismatch")
    tx.run(
        "MATCH (s:IngestionState {repository_id:$repository_id}) "
        "SET s.retrieval_schema_version=1, s.source_revision=$revision, "
        "s.data_revision=coalesce(s.data_revision,0)+1, s.chunk_policy_id=$policy, "
        "s.documentation_hash=$docs_hash, s.documentation_status=$docs_status, s.retrieval_state=$state "
        "REMOVE s.lock_version, s.indexing_attempt, s.pending_profile_id",
        repository_id=repository_id, revision=revision, policy=policy_id(max_bytes),
        docs_hash=docs_hash, docs_status=docs_status, state=state,
    ).consume()


class EvidenceStore:
    """One short-lived driver/session boundary; model calls are always outside it."""

    def __init__(self, settings: Settings, *, deadline: float | None = None):
        if settings.neo4j_password is None:
            raise EvidenceStoreError("NEO4J_PASSWORD is not configured")
        self.settings = settings
        self.deadline = deadline
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
            connection_timeout=settings.neo4j_timeout_seconds,
            connection_acquisition_timeout=settings.neo4j_timeout_seconds, max_transaction_retry_time=0,
        )
        try:
            version = self.read("CALL dbms.components() YIELD versions RETURN versions[0] AS version")[0]["version"]
            if version != NEO4J_VERSION:
                raise EvidenceStoreError(f"Retrieval requires Neo4j {NEO4J_VERSION}")
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "EvidenceStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self.driver.close()

    def read(self, cypher: str, *, timeout: float | None = None, **parameters: object) -> list[dict]:
        budget = min(timeout or self.settings.neo4j_timeout_seconds, self.settings.neo4j_timeout_seconds)
        if self.deadline is not None:
            budget = min(budget, self.deadline - time.monotonic())
        if budget <= 0:
            raise TimeoutError("Retrieval deadline exceeded")
        try:
            with self.driver.session(database=self.settings.neo4j_database) as session:
                return session.run(Query(cypher, timeout=budget),
                                   **parameters).data()
        except (Neo4jError, DriverError, OSError) as exc:
            raise EvidenceStoreError(f"Evidence query failed ({type(exc).__name__})") from None

    def write(self, callback: Callable[[ManagedTransaction], T]) -> T:
        try:
            with self.driver.session(database=self.settings.neo4j_database) as session:
                return session.execute_write(unit_of_work(timeout=60)(callback))
        except (Neo4jError, DriverError, OSError) as exc:
            raise EvidenceStoreError(f"Evidence transaction failed ({type(exc).__name__})") from None

    def ensure_schema(self) -> None:
        for statement in (*SCHEMA, *RETRIEVAL_SCHEMA):
            self.read(statement)

    def state(self, repository_id: str) -> dict:
        rows = self.read("MATCH (s:IngestionState {repository_id:$repository_id}) RETURN properties(s) AS state",
                         repository_id=repository_id)
        if not rows:
            raise EvidenceStoreError("Repository has no structural snapshot")
        return rows[0]["state"]

    def snapshot(self, repository_id: str) -> tuple[Snapshot, dict]:
        before = self.state(repository_id)
        nodes = self.read("MATCH (n:Entity {repository_id:$repository_id}) RETURN properties(n) AS value ORDER BY n.id",
                          repository_id=repository_id)
        edges = self.read(
            "MATCH (a:Entity {repository_id:$repository_id})-[r]->(b:Entity {repository_id:$repository_id}) "
            "WHERE type(r) IN ['CONTAINS','CALLS','IMPORTS','INHERITS'] AND r.repository_id=$repository_id "
            "RETURN properties(r) AS value ORDER BY r.id", repository_id=repository_id,
        )
        if revision_token(before) != revision_token(self.state(repository_id)):
            raise ConcurrentUpdate("Repository changed while reading structural source")
        names = [field.name for field in fields(Entity)]
        entities = tuple(Entity(**{name: row["value"][name] for name in names}) for row in nodes)
        refs = [Reference(**ref) for row in nodes for ref in json.loads(row["value"]["references_json"])]
        names = [field.name for field in fields(Relationship)]
        relationships = tuple(Relationship(**{name: row["value"].get(name) for name in names}) for row in edges)
        snapshot = Snapshot(repository_id, before["source_root"], entities, relationships,
                            tuple(sorted(refs, key=lambda item: item.id)), tuple(before.get("skipped_paths", [])))
        snapshot.validate()
        if snapshot.digest != before["snapshot_hash"]:
            raise EvidenceStoreError("Stored structural snapshot does not match its manifest")
        return snapshot, before

    def refresh(self, snapshot: Snapshot, documents: Documents, expected: dict, max_bytes: int) -> dict:
        self.ensure_schema()

        def work(tx: ManagedTransaction) -> dict:
            state = lock_state(tx, snapshot.repository_id)
            if revision_token(state) != revision_token(expected) or state["snapshot_hash"] != snapshot.digest:
                raise ConcurrentUpdate("Repository changed before evidence refresh")
            reconcile_evidence(tx, snapshot, documents, max_bytes, state)
            return tx.run("MATCH (s:IngestionState {repository_id:$id}) RETURN properties(s) AS state",
                          id=snapshot.repository_id).single(strict=True)["state"]

        return self.write(work)

    @staticmethod
    def vector_names(repository_id: str) -> tuple[str, str]:
        suffix = stable_id("repository-vector-v1", repository_id)
        return "rea_evidence_vec_v1_" + suffix, "REA_Evidence_v1_" + suffix

    def index_ready(self, repository_id: str) -> bool:
        name, label = self.vector_names(repository_id)
        rows = self.read("SHOW VECTOR INDEXES YIELD name, labelsOrTypes, properties, options, state "
                         "WHERE name=$name RETURN *", name=name)
        if not rows:
            return False
        row = rows[0]
        config = row["options"].get("indexConfig", {})
        if (row["labelsOrTypes"] != [label] or row["properties"] != ["embedding"]
                or config.get("vector.dimensions") != 768
                or str(config.get("vector.similarity_function", "")).lower() != "cosine"):
            raise EvidenceStoreError("Repository vector index has incompatible configuration")
        if row["state"] == "FAILED":
            raise EvidenceStoreError("Repository vector index failed to populate")
        return row["state"] == "ONLINE"

    def ensure_vector_index(self, repository_id: str) -> None:
        if self.index_ready(repository_id):
            return
        name, label = self.vector_names(repository_id)
        self.read(f"CREATE VECTOR INDEX `{name}` IF NOT EXISTS FOR (c:`{label}`) ON c.embedding "
                  "OPTIONS {indexConfig: {`vector.dimensions`:768, `vector.similarity_function`:'cosine'}}")
        deadline = time.monotonic() + self.settings.neo4j_timeout_seconds
        while not self.index_ready(repository_id):
            if time.monotonic() >= deadline:
                raise EvidenceStoreError("Repository vector index is not ONLINE")
            time.sleep(0.05)

    def chunks(self, repository_id: str, include_vectors: bool = False) -> list[dict]:
        projection = "properties(c)" if include_vectors else "c{.*, embedding:null}"
        return [row["chunk"] for row in self.read(
            "MATCH (s:IngestionState {repository_id:$repository_id}), "
            "(c:Chunk {repository_id:$repository_id})-[:EVIDENCE_FOR]->(o) "
            "WHERE o.repository_id=$repository_id AND c.owner_id=o.id "
            "AND c.source_revision=s.source_revision AND c.file_source_hash=o.source_hash "
            "AND (c.owner_kind='entity' OR s.documentation_status='current') "
            f"RETURN {projection} AS chunk ORDER BY c.path,c.start_line,c.id", repository_id=repository_id,
        )]

    def begin(self, repository_id: str, expected: dict, profile: EmbeddingProfile, attempt: str) -> dict:
        def work(tx: ManagedTransaction) -> dict:
            state = lock_state(tx, repository_id)
            if revision_token(state) != revision_token(expected):
                raise ConcurrentUpdate("Source changed before indexing began")
            return tx.run("MATCH (s:IngestionState {repository_id:$repository_id}) "
                   "SET s.retrieval_state='BUILDING', s.indexing_attempt=$attempt, "
                   "s.pending_profile_id=$profile, s.data_revision=coalesce(s.data_revision,0)+1 "
                   "REMOVE s.lock_version RETURN properties(s) AS state", repository_id=repository_id,
                   attempt=attempt, profile=profile.id).single(strict=True)["state"]
        return self.write(work)

    def fail(self, repository_id: str, attempt: str) -> None:
        def work(tx: ManagedTransaction) -> None:
            state = lock_state(tx, repository_id)
            if state.get("indexing_attempt") == attempt:
                tx.run("MATCH (s:IngestionState {repository_id:$repository_id}) "
                       "SET s.retrieval_state='FAILED', s.data_revision=coalesce(s.data_revision,0)+1 "
                       "REMOVE s.indexing_attempt, s.pending_profile_id", repository_id=repository_id).consume()
            tx.run("MATCH (s:IngestionState {repository_id:$repository_id}) REMOVE s.lock_version",
                   repository_id=repository_id).consume()
        self.write(work)

    def publish(self, repository_id: str, captured: dict, attempt: str, profile: EmbeddingProfile,
                chunks: list[dict], vectors: list[list[float]]) -> None:
        validate_vectors(vectors, len(chunks))
        _, label = self.vector_names(repository_id)

        def work(tx: ManagedTransaction) -> None:
            state = lock_state(tx, repository_id)
            if (revision_token(state) != revision_token(captured) or state.get("indexing_attempt") != attempt
                    or state.get("pending_profile_id") != profile.id or state.get("documentation_status") != "current"):
                raise ConcurrentUpdate("Indexing attempt or source revision was superseded")
            rows = [{"id": chunk["id"], "hash": chunk["embedding_input_hash"], "vector": vector}
                    for chunk, vector in zip(chunks, vectors, strict=True)]
            count = tx.run("MATCH (c:Chunk {repository_id:$repository_id}) "
                           "WHERE c.source_revision=$revision RETURN count(c) AS count",
                           repository_id=repository_id, revision=state["source_revision"]).single(strict=True)["count"]
            if count != len(rows):
                raise ConcurrentUpdate("Chunk manifest changed before publication")
            record = tx.run(
                "UNWIND $rows AS row MATCH (c:Chunk {id:row.id, repository_id:$repository_id}) "
                "WHERE c.embedding_input_hash=row.hash AND c.source_revision=$revision "
                f"SET c:`{label}`, c.embedding=row.vector, c.vector_input_hash=row.hash, "
                "c.embedding_profile_id=$profile_id, c.embedding_profile_json=$profile_json, "
                "c.embedding_dimension=768 RETURN count(c) AS count",
                rows=rows, repository_id=repository_id, revision=state["source_revision"],
                profile_id=profile.id, profile_json=profile.model_dump_json(),
            ).single(strict=True)
            if record["count"] != len(rows):
                raise ConcurrentUpdate("Chunk input changed before publication")
            tx.run("MATCH (s:IngestionState {repository_id:$repository_id}) "
                   "SET s.published_source_revision=s.source_revision, s.published_snapshot_hash=s.snapshot_hash, "
                   "s.published_documentation_hash=s.documentation_hash, s.published_profile_id=$profile_id, "
                   "s.published_profile_json=$profile_json, s.retrieval_state='READY', "
                   "s.data_revision=coalesce(s.data_revision,0)+1 "
                   "REMOVE s.indexing_attempt, s.pending_profile_id, s.lock_version",
                   repository_id=repository_id, profile_id=profile.id, profile_json=profile.model_dump_json()).consume()
        self.write(work)

    def vector_candidates(self, repository_id: str, profile: EmbeddingProfile,
                          vector: list[float], limit: int, timeout: float | None = None) -> list[dict]:
        validate_vectors([vector], 1)
        state = self.state(repository_id)
        if effective_readiness(state, profile.id).state != "READY" or not self.index_ready(repository_id):
            raise EvidenceStoreError("Semantic retrieval is not ready for the current source/profile")
        name, _ = self.vector_names(repository_id)
        # Native ANN can miss the nearest vector for tiny k. Bound oversampling,
        # then apply the requested result limit after stable ordering and guards.
        candidates = min(400, max(20, 4 * limit))
        rows = self.read(
            "CALL db.index.vector.queryNodes($index_name,$candidates,$vector) YIELD node AS c, score "
            "MATCH (s:IngestionState {repository_id:$repository_id}), (c)-[:EVIDENCE_FOR]->(o) "
            "WHERE c.repository_id=$repository_id AND o.repository_id=$repository_id AND c.owner_id=o.id "
            "AND c.source_revision=s.source_revision AND c.file_source_hash=o.source_hash "
            "AND c.embedding_profile_id=$profile_id AND c.vector_input_hash=c.embedding_input_hash "
            "AND s.retrieval_state='READY' AND s.published_profile_id=$profile_id "
            "AND s.published_source_revision=s.source_revision AND s.published_snapshot_hash=s.snapshot_hash "
            "AND s.published_documentation_hash=s.documentation_hash AND s.documentation_status='current' "
            "RETURN c.id AS chunk_id, score ORDER BY score DESC,chunk_id LIMIT $limit",
            index_name=name, candidates=candidates, limit=limit, vector=vector, repository_id=repository_id, profile_id=profile.id,
            timeout=timeout,
        )
        if revision_token(state) != revision_token(self.state(repository_id)):
            raise ConcurrentUpdate("Repository changed during vector retrieval")
        return rows

    def owners(self, repository_id: str) -> list[dict]:
        entities = self.read("MATCH (n:Entity {repository_id:$repository_id}) "
                             "RETURN properties(n) AS owner, 'entity' AS owner_kind ORDER BY n.id",
                             repository_id=repository_id)
        documents = self.read("MATCH (s:IngestionState {repository_id:$repository_id}), "
                              "(n:Document {repository_id:$repository_id}) "
                              "WHERE s.documentation_status='current' AND n.source_revision=s.source_revision "
                              "RETURN properties(n) AS owner, 'document' AS owner_kind ORDER BY n.id",
                              repository_id=repository_id)
        return [row["owner"] | {"owner_kind": row["owner_kind"]} for row in entities + documents]

    def adjacent(self, repository_id: str, entity_id: str, step: TraversalStep,
                 limit: int, timeout: float) -> list[dict]:
        # Both interpolated tokens come exclusively from validated Literal fields.
        pattern = (f"(a)-[r:{step.relationship}]->(b)" if step.direction == "outgoing"
                   else f"(a)<-[r:{step.relationship}]-(b)")
        return self.read(
            "MATCH (a:Entity {id:$entity_id, repository_id:$repository_id}), " + pattern + " "
            "WHERE b:Entity AND b.repository_id=$repository_id AND r.repository_id=$repository_id "
            "RETURN b.id AS target, properties(r) AS edge ORDER BY b.id,r.id LIMIT $limit",
            repository_id=repository_id, entity_id=entity_id, limit=limit, timeout=timeout,
        )
