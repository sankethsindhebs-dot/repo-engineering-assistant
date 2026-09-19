"""Internal direct retrieval and bounded traversal over actual stored relationships."""

import asyncio
from collections import defaultdict, deque
from dataclasses import asdict
import json
from pathlib import PurePosixPath
import time

from pydantic import ValidationError

from backend.config import Settings
from backend.graph.evidence_store import ConcurrentUpdate, EvidenceStore, EvidenceStoreError, effective_readiness, revision_token
from backend.ingestion.chunks import QUERY_PREFIX, lexical_terms, python_chunks
from backend.llm import ModelProvider, ProviderError, create_provider
from backend.retrieval.models import (
    DirectOrigin, EmbeddingProfile, Evidence, GraphOrigin, PathEdge, Readiness,
    RetrievalRequest, RetrievalResult, Seed, TraversalProfile, UnresolvedReference,
)


STATIC_LIMIT = "Static edges are incomplete; a missing CALLS edge is not proof of no runtime call."
RELATION_ORDER = {"CALLS": 0, "INHERITS": 1, "IMPORTS": 2, "CONTAINS": 3}


def remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("Retrieval deadline exceeded")
    return value


def path_query(query: str) -> str:
    value = query.strip().replace("\\", "/")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ":" in value:
        return ""
    return path.as_posix()


def direct_seeds(request: RetrievalRequest, owners: list[dict], chunks: list[dict],
                 vectors: list[dict]) -> list[Seed]:
    """Fuse owner ranks once per channel, preserving the actual matching chunk."""
    by_id = {chunk["id"]: chunk for chunk in chunks}
    by_owner = defaultdict(list)
    owner_map = {owner["id"]: owner for owner in owners}
    for chunk in chunks:
        by_owner[chunk["owner_id"]].append(chunk)
    channels = {}
    query, normalized_path = request.query.strip(), path_query(request.query)
    if "exact" in request.channels:
        hits = []
        for owner in owners:
            strength = (3 if query == owner.get("qualified_name") else
                        2 if normalized_path == owner["path"] else
                        1 if query == owner.get("name") else 0)
            if strength:
                items = by_owner[owner["id"]]
                hits.append((owner["id"], float(strength), items[0]["id"] if items else None))
        channels["exact"] = hits
    if "lexical" in request.channels:
        terms = set(lexical_terms(query))
        channels["lexical"] = [(chunk["owner_id"], float(len(terms.intersection(chunk["terms"]))), chunk["id"])
                               for chunk in chunks if terms.intersection(chunk["terms"])]
    if "vector" in request.channels:
        channels["vector"] = [(by_id[row["chunk_id"]]["owner_id"], row["score"], row["chunk_id"])
                              for row in vectors if row["chunk_id"] in by_id]
    origins, matched, totals = defaultdict(list), defaultdict(set), defaultdict(float)
    for channel, hits in channels.items():
        ordered = sorted(hits, key=lambda item: (-item[1], owner_map[item[0]]["path"],
                                                owner_map[item[0]].get("start_line", 1), item[0], item[2] or ""))
        seen = set()
        for owner_id, score, chunk_id in ordered:
            if owner_id in seen:
                continue
            if len(seen) >= request.channel_limit:
                break
            seen.add(owner_id)
            rank = len(seen)
            origins[owner_id].append(DirectOrigin(channel=channel, rank=rank, score=score, matched_chunk_id=chunk_id))
            totals[owner_id] += 1 / (60 + rank)
            if chunk_id:
                matched[owner_id].add(chunk_id)
    ordered_ids = sorted(origins, key=lambda identity: (
        not any(origin.channel == "exact" for origin in origins[identity]), -totals[identity], identity))
    return [Seed(owner_kind=owner_map[identity]["owner_kind"], owner_id=identity,
                 chunk_ids=tuple(sorted(matched[identity])), direct=tuple(origins[identity]), rrf_score=totals[identity])
            for identity in ordered_ids[:request.traversal.max_seeds]]


def expand_graph(store: EvidenceStore, repository_id: str, seeds: list[Seed], profile: TraversalProfile,
                 deadline: float) -> tuple[dict[str, list[GraphOrigin]], set[str], int, set[str]]:
    roots = [seed.owner_id for seed in seeds if seed.owner_kind == "entity"]
    bounds = set()
    if len(roots) > profile.max_nodes:
        bounds.add("node_limit")
    roots = roots[:profile.max_nodes]
    admitted, scheduled, expanded = set(roots), set(roots), set()
    queue = deque((root, root, (root,), ()) for root in roots)
    origins = defaultdict(list)
    examined = 0
    steps = sorted(profile.steps, key=lambda step: (RELATION_ORDER[step.relationship], step.direction == "incoming"))
    while queue:
        identity, seed_id, nodes, path = queue.popleft()
        if identity in expanded:
            continue
        expanded.add(identity)
        if len(path) >= profile.max_depth:
            bounds.add("depth_limit")
            continue
        branch = 0
        for step in steps:
            limit = min(profile.branch_size - branch, profile.max_edges - examined)
            if limit <= 0:
                bounds.add("edge_limit" if examined >= profile.max_edges else "branch_limit")
                break
            rows = store.adjacent(repository_id, identity, step, limit, remaining(deadline))
            if len(rows) == limit:
                bounds.add("edge_limit" if examined + len(rows) >= profile.max_edges else "branch_limit")
            for row in rows:
                examined += 1
                branch += 1
                target, edge = row["target"], row["edge"]
                if target in nodes:
                    continue
                if target not in admitted and len(admitted) >= profile.max_nodes:
                    bounds.add("node_limit")
                    continue
                admitted.add(target)
                evidence = PathEdge(
                    relationship_id=edge["id"], relationship=step.relationship,
                    stored_source_id=edge["source_id"], stored_target_id=edge["target_id"], direction=step.direction,
                    **{name: edge.get(name) for name in ("reference_id", "expression", "path", "start_line",
                                                       "end_line", "start_column", "end_column")},
                )
                origin = GraphOrigin(seed_entity_id=seed_id, hops=len(path) + 1,
                                     entity_path=(*nodes, target), edges=(*path, evidence))
                # Expand each entity once, but retain every distinct discovered
                # origin, including separate call sites between the same owners.
                if origin not in origins[target]:
                    origins[target].append(origin)
                terminal = step.relationship == "CONTAINS" and step.direction == "incoming" and profile.parent_context == "terminal"
                if not terminal and target not in scheduled:
                    scheduled.add(target)
                    queue.append((target, seed_id, origin.entity_path, origin.edges))
            if examined >= profile.max_edges:
                break
        if examined >= profile.max_edges and queue:
            bounds.add("edge_limit")
            break
    return dict(origins), admitted, examined, bounds


def assemble_evidence(request: RetrievalRequest, seeds: list[Seed], chunks: list[dict],
                      graph: dict[str, list[GraphOrigin]]) -> tuple[list[Evidence], set[str]]:
    by_owner = defaultdict(list)
    terms = set(lexical_terms(request.query))
    for chunk in chunks:
        by_owner[chunk["owner_id"]].append(chunk)
    seed_map = {seed.owner_id: seed for seed in seeds}
    seed_rank = {seed.owner_id: rank for rank, seed in enumerate(seeds)}
    graph_ids = sorted(graph, key=lambda identity: (
        min((seed_rank[item.seed_entity_id], item.hops) for item in graph[identity]), identity))
    output, bounds, used_bytes = {}, set(), 0
    for identities, capacity in ((list(seed_map), request.direct_evidence_limit), (graph_ids, request.graph_evidence_limit)):
        emitted = 0
        for identity in identities:
            candidates = sorted(by_owner[identity], key=lambda item: (
                item["id"] not in (seed_map[identity].chunk_ids if identity in seed_map else ()),
                -len(terms.intersection(item["terms"])), item["start_line"], item["id"]))
            if len(candidates) > request.chunks_per_owner:
                bounds.add("owner_chunk_limit")
            for chunk in candidates[:request.chunks_per_owner]:
                if chunk["id"] in output:
                    continue
                if emitted >= capacity:
                    bounds.add("evidence_limit")
                    break
                size = len(chunk["text"].encode("utf-8"))
                if used_bytes + size > request.source_budget_bytes:
                    bounds.add("source_budget")
                    continue
                keys = ("owner_kind", "owner_id", "entity_id", "document_id", "entity_kind", "qualified_name",
                        "heading", "path", "start_line", "end_line", "text", "text_hash", "file_source_hash")
                output[chunk["id"]] = Evidence(
                    chunk_id=chunk["id"], **{key: chunk.get(key) for key in keys},
                    direct=seed_map[identity].direct if identity in seed_map else (), graph=tuple(graph.get(identity, [])),
                )
                used_bytes += size
                emitted += 1
    return list(output.values()), bounds


def unresolved_diagnostics(owners: list[dict], selected: set[str]) -> tuple[list[UnresolvedReference], bool]:
    diagnostics = []
    for owner in sorted(owners, key=lambda item: item["id"]):
        if owner["owner_kind"] != "entity" or owner["id"] not in selected:
            continue
        for ref in sorted(json.loads(owner.get("references_json", "[]")), key=lambda item: item["id"]):
            if ref.get("target_id") is None:
                values = {key: value for key, value in ref.items() if key != "target_id"}
                values["expression_truncated"] = len(values["expression"]) > 512
                values["expression"] = values["expression"][:512]
                diagnostics.append(UnresolvedReference(**values))
                if len(diagnostics) > 100:
                    return diagnostics[:100], True
    return diagnostics, False


async def _retrieve_once(settings: Settings, store: EvidenceStore, request: RetrievalRequest,
                         provider: ModelProvider | None, deadline: float) -> RetrievalResult:
    before = store.state(request.repository_id)
    readiness = effective_readiness(before)
    owners = store.owners(request.repository_id)
    chunks = store.chunks(request.repository_id)
    diagnostics = [STATIC_LIMIT]
    if not before.get("retrieval_schema_version"):
        snapshot, _ = store.snapshot(request.repository_id)
        chunks = [asdict(chunk) for chunk in python_chunks(snapshot, settings.retrieval_chunk_bytes)]
        diagnostics.append("Python evidence derived from the stored legacy snapshot; retrieval is unindexed.")
    active = None
    try:
        if before.get("published_profile_json"):
            active = EmbeddingProfile.model_validate_json(before["published_profile_json"])
    except ValidationError:
        readiness = Readiness(state="DIRTY", reasons=("invalid_published_profile",))
    vectors = []
    own_provider = False
    if readiness.state == "READY":
        try:
            if not store.index_ready(request.repository_id):
                readiness = Readiness(state="DIRTY", reasons=("vector_index_not_online",))
        except EvidenceStoreError:
            readiness = Readiness(state="DIRTY", reasons=("vector_index_incompatible",))
    if "vector" in request.channels and readiness.state == "READY":
        try:
            own_provider = provider is None
            provider = provider or create_provider(settings)
            identity = await asyncio.wait_for(provider.embedding_identity(), remaining(deadline))
            profile = EmbeddingProfile.from_identity(identity, settings.retrieval_chunk_bytes)
            readiness = effective_readiness(before, profile.id)
            if readiness.state == "READY":
                payload = QUERY_PREFIX + request.query
                if len(payload.encode("utf-8")) > settings.retrieval_chunk_bytes:
                    raise ValueError("Query exceeds embedding input budget")
                batch = await asyncio.wait_for(provider.embed([payload]), remaining(deadline))
                if len(batch) != 1:
                    raise ValueError("Query embedding response count mismatch")
                if identity != await asyncio.wait_for(provider.embedding_identity(), remaining(deadline)):
                    raise ProviderError("Embedding model changed during query")
                vectors = store.vector_candidates(request.repository_id, profile, batch[0], request.channel_limit,
                                                   remaining(deadline))
        except ConcurrentUpdate:
            raise
        except (ProviderError, ValueError, EvidenceStoreError) as exc:
            readiness = Readiness(state="FAILED", reasons=("vector_request_unavailable",))
            diagnostics.append(f"Vector retrieval unavailable ({type(exc).__name__}); no model fallback used.")
        finally:
            if own_provider and provider is not None:
                await provider.aclose()
    if "vector" in request.channels and readiness.state != "READY":
        diagnostics.append("Semantic retrieval is not ready for this source/profile; only available direct/graph channels ran.")
    remaining(deadline)
    seeds = direct_seeds(request, owners, chunks, vectors)
    graph, admitted, examined, bounds = {}, set(), 0, set()
    if request.graph:
        graph, admitted, examined, bounds = expand_graph(store, request.repository_id, seeds, request.traversal, deadline)
    evidence, evidence_bounds = assemble_evidence(request, seeds, chunks, graph)
    bounds.update(evidence_bounds)
    unresolved, capped = unresolved_diagnostics(owners, admitted | {seed.owner_id for seed in seeds})
    if capped:
        bounds.add("reference_limit")
    remaining(deadline)
    if revision_token(before) != revision_token(store.state(request.repository_id)):
        raise ConcurrentUpdate("Repository changed during retrieval")
    return RetrievalResult(
        repository_id=request.repository_id, query=request.query, snapshot_hash=before["snapshot_hash"],
        source_revision=before.get("source_revision", 0), data_revision=before.get("data_revision", 0),
        documentation_hash=before.get("documentation_hash"), active_embedding_profile=active,
        enabled_channels=request.channels, traversal=request.traversal, graph_enabled=request.graph,
        readiness=readiness, seeds=tuple(seeds), evidence=tuple(evidence), unresolved=tuple(unresolved),
        admitted_entities=len(admitted), examined_relationships=examined, bounds_reached=tuple(sorted(bounds)),
        diagnostics=tuple(diagnostics),
    )


async def retrieve(settings: Settings, request: RetrievalRequest, *, provider: ModelProvider | None = None) -> RetrievalResult:
    """Return evidence only; retry a concurrent read once within the original deadline."""
    deadline = time.monotonic() + request.deadline_seconds
    with EvidenceStore(settings, deadline=deadline) as store:
        for attempt in range(2):
            try:
                return await _retrieve_once(settings, store, request, provider, deadline)
            except ConcurrentUpdate:
                if attempt:
                    raise
                remaining(deadline)
    raise ConcurrentUpdate("Repository changed repeatedly during retrieval")
