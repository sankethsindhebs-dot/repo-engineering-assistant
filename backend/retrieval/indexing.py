"""Explicit embedding publication; neither ingestion nor retrieval generates answers."""

from dataclasses import dataclass
from pathlib import Path
import uuid

from backend.config import Settings
from backend.graph.evidence_store import ConcurrentUpdate, EvidenceStore, effective_readiness, revision_token, validate_vectors
from backend.ingestion.documents import Documents, read_documents
from backend.ingestion.repository import parse_repository
from backend.llm import ModelProvider, create_provider
from backend.retrieval.models import EmbeddingProfile


@dataclass(frozen=True)
class IndexResult:
    repository_id: str
    snapshot_hash: str
    source_revision: int
    profile_id: str
    chunks: int
    embedded: int
    reused: int
    state: str = "READY"


def verify_checkout(repository: Path | str, repository_id: str, source_root: str,
                    snapshot_hash: str, documentation_hash: str | None = None) -> Documents:
    snapshot = parse_repository(repository, repository_id, source_root)
    if snapshot.digest != snapshot_hash:
        raise ConcurrentUpdate("Checkout differs from stored Python snapshot; run structural ingestion first")
    documents = read_documents(repository, repository_id)
    if documentation_hash is not None and documents.digest != documentation_hash:
        raise ConcurrentUpdate("Documentation changed during indexing; refresh and retry")
    return documents


async def index_repository(settings: Settings, repository: Path | str, repository_id: str,
                           *, provider: ModelProvider | None = None) -> IndexResult:
    """Upgrade/index an existing graph. A supplied provider remains owned by its caller."""
    with EvidenceStore(settings) as store:
        snapshot, before = store.snapshot(repository_id)
        documents = verify_checkout(repository, repository_id, snapshot.source_root, snapshot.digest)
        current = store.refresh(snapshot, documents, before, settings.retrieval_chunk_bytes)
        if current["snapshot_hash"] != snapshot.digest or current["documentation_hash"] != documents.digest:
            raise ConcurrentUpdate("Source changed after evidence refresh")
        chunks = store.chunks(repository_id, include_vectors=True)
        owned_provider = provider is None
        provider = provider or create_provider(settings)
        attempt = None
        try:
            identity = await provider.embedding_identity()
            profile = EmbeddingProfile.from_identity(identity, settings.retrieval_chunk_bytes)
            store.ensure_vector_index(repository_id)
            vectors, missing = [], []
            for position, chunk in enumerate(chunks):
                reusable = (chunk.get("embedding_profile_id") == profile.id
                            and chunk.get("vector_input_hash") == chunk["embedding_input_hash"]
                            and chunk.get("embedding") is not None)
                vector = chunk["embedding"] if reusable else None
                if reusable:
                    validate_vectors([vector], 1)
                else:
                    missing.append(position)
                vectors.append(vector)
            if not missing and effective_readiness(current, profile.id).state == "READY":
                verify_checkout(repository, repository_id, snapshot.source_root, snapshot.digest, documents.digest)
                if identity != await provider.embedding_identity():
                    raise ConcurrentUpdate("Embedding model identity changed during indexing")
                if revision_token(current) != revision_token(store.state(repository_id)):
                    raise ConcurrentUpdate("Source changed during unchanged-index verification")
                return IndexResult(repository_id, snapshot.digest, current["source_revision"],
                                   profile.id, len(chunks), 0, len(chunks))
            attempt = uuid.uuid4().hex
            captured = store.begin(repository_id, current, profile, attempt)
            for offset in range(0, len(missing), settings.embedding_batch_size):
                positions = missing[offset:offset + settings.embedding_batch_size]
                inputs = [chunks[position]["embedding_input"] for position in positions]
                batch = await provider.embed(inputs)
                validate_vectors(batch, len(positions))
                for position, vector in zip(positions, batch, strict=True):
                    vectors[position] = vector
            if identity != await provider.embedding_identity():
                raise ConcurrentUpdate("Embedding model identity changed during indexing")
            verify_checkout(repository, repository_id, snapshot.source_root, snapshot.digest, documents.digest)
            store.publish(repository_id, captured, attempt, profile, chunks, vectors)
            return IndexResult(repository_id, snapshot.digest, captured["source_revision"], profile.id,
                               len(chunks), len(missing), len(chunks) - len(missing))
        except BaseException:
            if attempt is not None:
                store.fail(repository_id, attempt)
            raise
        finally:
            if owned_provider:
                await provider.aclose()
