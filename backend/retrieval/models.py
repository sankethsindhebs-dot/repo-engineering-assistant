"""Internal retrieval contracts. These are neither agent tools nor HTTP routes."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.ingestion.chunks import INPUT_FORMAT, policy_id
from backend.ingestion.models import stable_id
from backend.llm import EmbeddingIdentity


Relation = Literal["CALLS", "IMPORTS", "INHERITS", "CONTAINS"]
Direction = Literal["outgoing", "incoming"]
Channel = Literal["exact", "lexical", "vector"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class EmbeddingProfile(Contract):
    provider: str
    model: str
    digest: str = Field(min_length=1)
    dimension: Literal[768] = 768
    prefix_policy: Literal["nomic-search-v1"] = "nomic-search-v1"
    input_format: str = INPUT_FORMAT
    chunk_policy_id: str

    @property
    def id(self) -> str:
        return stable_id("embedding-profile-v1", self.model_dump())

    @classmethod
    def from_identity(cls, identity: EmbeddingIdentity, max_bytes: int) -> "EmbeddingProfile":
        if identity.provider != "ollama" or identity.model.split(":")[0] != "nomic-embed-text":
            raise ValueError("Retrieval v1 requires the configured local nomic-embed-text profile")
        return cls(provider=identity.provider, model=identity.model, digest=identity.digest,
                   chunk_policy_id=policy_id(max_bytes))


class TraversalStep(Contract):
    relationship: Relation
    direction: Direction


class TraversalProfile(Contract):
    steps: tuple[TraversalStep, ...] = (
        TraversalStep(relationship="CALLS", direction="outgoing"),
        TraversalStep(relationship="IMPORTS", direction="outgoing"),
        TraversalStep(relationship="INHERITS", direction="outgoing"),
        TraversalStep(relationship="CONTAINS", direction="outgoing"),
        TraversalStep(relationship="CONTAINS", direction="incoming"),
    )
    max_seeds: int = Field(default=5, ge=1, le=20)
    max_depth: int = Field(default=2, ge=0, le=6)
    max_nodes: int = Field(default=40, ge=1, le=200)
    max_edges: int = Field(default=80, ge=0, le=1000)
    branch_size: int = Field(default=12, ge=1, le=100)
    parent_context: Literal["terminal", "ascend"] = "terminal"

    @model_validator(mode="after")
    def unique_steps(self) -> "TraversalProfile":
        if len(set((s.relationship, s.direction) for s in self.steps)) != len(self.steps):
            raise ValueError("Traversal steps must be unique")
        return self


class RetrievalRequest(Contract):
    repository_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    query: str = Field(min_length=1, max_length=1000)
    channels: tuple[Channel, ...] = ("exact", "lexical", "vector")
    graph: bool = True
    traversal: TraversalProfile = Field(default_factory=TraversalProfile)
    channel_limit: int = Field(default=20, ge=1, le=100)
    direct_evidence_limit: int = Field(default=10, ge=1, le=50)
    graph_evidence_limit: int = Field(default=10, ge=1, le=50)
    chunks_per_owner: int = Field(default=2, ge=1, le=10)
    source_budget_bytes: int = Field(default=32768, ge=1, le=262144)
    deadline_seconds: float = Field(default=15, gt=0, le=120)

    @model_validator(mode="after")
    def valid_query(self) -> "RetrievalRequest":
        if not self.query.strip() or not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError("Provide a nonblank query and unique retrieval channels")
        return self


class Readiness(Contract):
    state: Literal["UNINDEXED", "DIRTY", "BUILDING", "READY", "FAILED"]
    reasons: tuple[str, ...] = ()


class DirectOrigin(Contract):
    channel: Channel
    rank: int
    score: float | None = None
    matched_chunk_id: str | None = None


class PathEdge(Contract):
    relationship_id: str
    relationship: Relation
    stored_source_id: str
    stored_target_id: str
    direction: Direction
    reference_id: str | None = None
    expression: str | None = None
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    start_column: int | None = None
    end_column: int | None = None


class GraphOrigin(Contract):
    seed_entity_id: str
    hops: int
    entity_path: tuple[str, ...]
    edges: tuple[PathEdge, ...]


class Seed(Contract):
    owner_kind: Literal["entity", "document"]
    owner_id: str
    chunk_ids: tuple[str, ...]
    direct: tuple[DirectOrigin, ...]
    rrf_score: float


class Evidence(Contract):
    owner_kind: Literal["entity", "document"]
    owner_id: str
    entity_id: str | None = None
    document_id: str | None = None
    chunk_id: str
    entity_kind: str | None = None
    qualified_name: str | None = None
    heading: str | None = None
    path: str
    start_line: int
    end_line: int
    text: str
    text_hash: str
    file_source_hash: str
    direct: tuple[DirectOrigin, ...] = ()
    graph: tuple[GraphOrigin, ...] = ()


class UnresolvedReference(Contract):
    id: str
    owner_id: str
    kind: str
    path: str
    expression: str
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    reason: str
    expression_truncated: bool = False


class RetrievalResult(Contract):
    repository_id: str
    query: str
    snapshot_hash: str
    source_revision: int
    data_revision: int
    documentation_hash: str | None
    active_embedding_profile: EmbeddingProfile | None
    enabled_channels: tuple[Channel, ...]
    traversal: TraversalProfile
    graph_enabled: bool
    readiness: Readiness
    seeds: tuple[Seed, ...]
    evidence: tuple[Evidence, ...]
    unresolved: tuple[UnresolvedReference, ...]
    admitted_entities: int
    examined_relationships: int
    bounds_reached: tuple[str, ...]
    diagnostics: tuple[str, ...]
