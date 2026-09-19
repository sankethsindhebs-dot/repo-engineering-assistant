"""Opt-in causal retrieval/lifecycle checks on the pinned real Neo4j server."""

import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import secrets
import shutil
import tempfile
import unittest
from unittest.mock import patch

from backend.config import Settings
from backend.graph.evidence_store import ConcurrentUpdate, EvidenceStore, EvidenceStoreError, effective_readiness
from backend.graph.store import write_snapshot
from backend.ingestion.documents import read_documents
from backend.ingestion.repository import parse_repository
from backend.llm import create_provider
from backend.retrieval.indexing import index_repository
from backend.retrieval.models import EmbeddingProfile, RetrievalRequest, TraversalProfile, TraversalStep
from backend.retrieval.retrieve import retrieve
from tests.support_retrieval import DeterministicProvider, FIXTURE, unit_vector


@unittest.skipUnless(os.environ.get("PHASE1B_NEO4J_TESTS") == "1", "Set PHASE1B_NEO4J_TESTS=1 for live writes")
class RetrievalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(_env_file=None)
        self.store = EvidenceStore(self.settings)
        self.addCleanup(self.store.close)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "repository"
        shutil.copytree(FIXTURE, self.root, ignore=shutil.ignore_patterns("__pycache__"))
        self.repository_id = "phase1b-test-" + secrets.token_hex(8)
        self.ids = [self.repository_id, self.repository_id + "-other"]
        self.addCleanup(self.cleanup)
        self.provider = DeterministicProvider()

    def cleanup(self):
        for label in ["Chunk", "Document", "Entity", "IngestionState"]:
            self.store.read(f"MATCH (n:{label}) WHERE n.repository_id IN $ids DETACH DELETE n", ids=self.ids)
        # Only deterministic index names allocated to this test's random repository IDs.
        for identity in self.ids:
            name, _ = self.store.vector_names(identity)
            self.store.read(f"DROP INDEX `{name}` IF EXISTS")

    def ingest(self, *, documents=True, repository_id=None):
        identity = repository_id or self.repository_id
        snapshot = parse_repository(self.root, identity)
        write_snapshot(self.settings, snapshot, documents=read_documents(self.root, identity) if documents else None)
        return snapshot

    def index(self, provider=None, settings=None, repository_id=None):
        return asyncio.run(index_repository(settings or self.settings, self.root, repository_id or self.repository_id,
                                            provider=provider or self.provider))

    def result(self, query="handle_request", *, channels=("exact",), relationship="CALLS", direction="outgoing",
               graph=True, repository_id=None, **limits):
        request = RetrievalRequest(repository_id=repository_id or self.repository_id, query=query, channels=channels,
                                   graph=graph, channel_limit=1 if channels == ("vector",) else 20,
                                   traversal=TraversalProfile(steps=(TraversalStep(relationship=relationship, direction=direction),),
                                                              max_seeds=1, **limits), deadline_seconds=30)
        return asyncio.run(retrieve(self.settings, request, provider=self.provider))

    def current_profile(self):
        return EmbeddingProfile.model_validate_json(self.store.state(self.repository_id)["published_profile_json"])

    def vectors(self):
        return self.store.read("MATCH (c:Chunk {repository_id:$id}) RETURN c.id AS id,c.embedding AS vector, "
                               "c.embedding_profile_id AS profile,c.vector_input_hash AS input_hash ORDER BY c.id",
                               id=self.repository_id)

    def structural_records(self):
        return (
            self.store.read("MATCH (n:Entity {repository_id:$id}) RETURN elementId(n) AS internal, "
                            "properties(n) AS properties ORDER BY n.id", id=self.repository_id),
            self.store.read("MATCH (:Entity {repository_id:$id})-[r]->(:Entity) RETURN elementId(r) AS internal, "
                            "properties(r) AS properties ORDER BY r.id", id=self.repository_id),
        )

    def test_additive_upgrade_from_phase1a_only_preserves_existing_nodes_edges_and_refs(self):
        snapshot = parse_repository(self.root, self.repository_id)
        # The structural writer's unchanged persistence path creates the exact Phase 1A shape.
        with patch("backend.graph.evidence_store.reconcile_evidence"), patch("backend.graph.evidence_store.RETRIEVAL_SCHEMA", ()):
            write_snapshot(self.settings, snapshot)
        state = self.store.state(self.repository_id)
        self.assertEqual(effective_readiness(state).state, "UNINDEXED")
        self.assertNotIn("retrieval_schema_version", state)
        self.assertEqual(self.store.read("MATCH (c:Chunk {repository_id:$id}) RETURN count(c) AS n", id=self.repository_id)[0]["n"], 0)
        before = self.structural_records()
        first = self.index()
        self.assertEqual(first.state, "READY")
        self.assertEqual(before, self.structural_records())
        calls = len(self.provider.calls)
        second = self.index()
        self.assertEqual(second.embedded, 0)
        self.assertEqual(len(self.provider.calls), calls)
        self.assertEqual(before, self.structural_records())

    def test_real_vector_graph_off_on_edge_intervention_causal_proof(self):
        self.ingest()
        self.index()
        off = self.result("entry flow", channels=("vector",), graph=False)
        on = self.result("entry flow", channels=("vector",))
        self.assertEqual(off.readiness.state, "READY")
        self.assertEqual(off.seeds, on.seeds)
        self.assertFalse(any("casefold" in item.text for item in off.evidence))
        target = next(item for item in on.evidence if item.qualified_name == "worker.transform")
        self.assertIn("casefold", target.text)
        edge = target.graph[0].edges[0]
        self.assertEqual((edge.relationship, edge.direction, edge.path, edge.start_line), ("CALLS", "outgoing", "entry.py", 5))
        self.store.read("MATCH ()-[r:CALLS {repository_id:$id, id:$edge}]->() DELETE r", id=self.repository_id, edge=edge.relationship_id)
        changed = self.result("entry flow", channels=("vector",))
        self.assertEqual(on.seeds, changed.seeds)
        self.assertFalse(any(item.qualified_name == "worker.transform" for item in changed.evidence))

    def test_lexical_graph_ablation_is_independent_of_model(self):
        self.ingest()
        off = self.result(channels=("lexical",), graph=False)
        on = self.result(channels=("lexical",))
        self.assertEqual(off.seeds, on.seeds)
        self.assertFalse(any("casefold" in item.text for item in off.evidence))
        self.assertTrue(any("casefold" in item.text and item.graph for item in on.evidence))
        self.assertEqual(self.provider.calls, [])

    def test_reverse_calls_imports_and_inheritance_use_actual_directions(self):
        self.ingest()
        cases = [("worker.transform", "CALLS", "entry.handle_request"),
                 ("worker", "IMPORTS", "entry"), ("base.Base", "INHERITS", "worker.Worker")]
        for query, relationship, expected in cases:
            with self.subTest(relationship=relationship):
                result = self.result(query, relationship=relationship, direction="incoming")
                item = next(item for item in result.evidence if item.qualified_name == expected)
                edge = item.graph[0].edges[0]
                self.assertEqual(edge.direction, "incoming")
                self.assertEqual(edge.stored_source_id, item.owner_id)
                forward = self.result(query, relationship=relationship)
                self.assertFalse(any(item.qualified_name == expected and item.graph for item in forward.evidence))

    def test_import_and_inheritance_edge_interventions(self):
        self.ingest()
        for query, relationship, target in [("entry", "IMPORTS", "worker"), ("worker.Worker", "INHERITS", "base.Base")]:
            before = self.result(query, relationship=relationship)
            item = next(item for item in before.evidence if item.qualified_name == target)
            edge = item.graph[0].edges[0]
            self.store.read(f"MATCH ()-[r:{relationship} {{repository_id:$id, id:$edge}}]->() DELETE r",
                            id=self.repository_id, edge=edge.relationship_id)
            after = self.result(query, relationship=relationship)
            self.assertEqual(before.seeds, after.seeds)
            self.assertFalse(any(item.qualified_name == target for item in after.evidence))

    def test_unresolved_callback_does_not_acquire_a_target_from_references(self):
        self.ingest()
        result = self.result("entry.via_runtime")
        self.assertTrue(any("callback" in ref.expression for ref in result.unresolved))
        self.assertFalse(any(item.graph for item in result.evidence))

    def test_live_depth_node_edge_branch_and_containment_bounds(self):
        self.ingest()
        for limits in [{"max_depth": 0}, {"max_nodes": 1}, {"max_edges": 0}]:
            result = self.result(**limits)
            self.assertFalse(any(item.qualified_name == "worker.transform" for item in result.evidence))
        result = self.result("worker.transform", direction="incoming", branch_size=1)
        self.assertEqual(result.examined_relationships, 1)
        self.assertLessEqual(result.admitted_entities, 2)
        terminal = self.result("worker.Worker.run", relationship="CONTAINS", direction="incoming")
        ascent = self.result("worker.Worker.run", relationship="CONTAINS", direction="incoming", parent_context="ascend")
        self.assertFalse(any(item.qualified_name == "worker" for item in terminal.evidence))
        self.assertTrue(any(item.qualified_name == "worker" for item in ascent.evidence))

    def test_document_exact_lexical_and_native_vector_evidence(self):
        self.ingest()
        self.index()
        for query, channels in [("README.md", ("exact",)), ("amber", ("lexical",)), ("documentation", ("vector",))]:
            result = self.result(query, channels=channels)
            matches = [item for item in result.evidence if item.owner_kind == "document" and item.path == "README.md"]
            self.assertTrue(matches, (query, result.model_dump()))
            item = matches[0]
            self.assertIsNone(item.entity_id)
            self.assertIsNone(item.qualified_name)
            self.assertEqual(item.graph, ())
            from backend.ingestion.chunks import source_lines
            source = (self.root / item.path).read_text()
            self.assertEqual(item.text, "".join(source_lines(source)[item.start_line - 1:item.end_line]))
        nodes = self.store.read("MATCH (d:Document {repository_id:$id}) RETURN labels(d) AS labels", id=self.repository_id)
        self.assertTrue(all(row["labels"] == ["Document"] for row in nodes))
        count = self.store.read("MATCH (d:Document {repository_id:$id})-[r]-() "
                                "WHERE type(r) IN ['CALLS','IMPORTS','INHERITS','CONTAINS'] RETURN count(r) AS n", id=self.repository_id)
        self.assertEqual(count[0]["n"], 0)

    def test_source_change_invalidates_semantics_but_current_exact_evidence_survives(self):
        self.ingest()
        self.index()
        old = self.store.state(self.repository_id)
        old_profile = self.current_profile()
        worker = self.root / "worker.py"
        worker.write_text(worker.read_text().replace("raw.strip()", "raw.lstrip()"))
        current = self.ingest(documents=False)
        state = self.store.state(self.repository_id)
        self.assertGreater(state["source_revision"], old["source_revision"])
        self.assertNotEqual(effective_readiness(state, old_profile.id).state, "READY")
        with self.assertRaises(EvidenceStoreError):
            self.store.vector_candidates(self.repository_id, old_profile, unit_vector(0), 5)
        exact = self.result("worker.transform")
        self.assertEqual(exact.snapshot_hash, current.digest)
        self.assertTrue(any("lstrip" in item.text for item in exact.evidence))
        self.assertFalse(any(item.owner_kind == "document" for item in self.result("README.md").evidence))
        published = self.index()
        self.assertEqual(published.embedded, 1)
        self.assertEqual(effective_readiness(self.store.state(self.repository_id), old_profile.id).state, "READY")

    def test_stored_ready_string_cannot_override_manifest_mismatch(self):
        self.ingest()
        self.index()
        profile = self.current_profile()
        self.store.read("MATCH (s:IngestionState {repository_id:$id}) SET s.snapshot_hash='changed'", id=self.repository_id)
        self.assertEqual(effective_readiness(self.store.state(self.repository_id), profile.id).state, "DIRTY")
        with self.assertRaises(EvidenceStoreError):
            self.store.vector_candidates(self.repository_id, profile, unit_vector(0), 1)

    def test_document_change_delete_rename_removes_old_chunks_and_vectors(self):
        self.ingest()
        self.index()
        before = self.store.state(self.repository_id)
        readme = self.root / "README.md"
        readme.write_text(readme.read_text().replace("amber", "violet"))
        self.ingest()
        self.assertEqual(effective_readiness(self.store.state(self.repository_id)).state, "DIRTY")
        self.assertEqual(self.result("amber", channels=("lexical",), graph=False).evidence, ())
        self.index()
        usage = self.root / "docs/usage.md"
        usage.rename(self.root / "docs/retention.md")
        readme.unlink()
        self.index()
        rows = self.store.chunks(self.repository_id, include_vectors=True)
        self.assertFalse(any(row["path"] in {"README.md", "docs/usage.md"} for row in rows))
        self.assertTrue(any(row["path"] == "docs/retention.md" for row in rows))
        self.assertNotEqual(before["documentation_hash"], self.store.state(self.repository_id)["documentation_hash"])

    def test_batched_order_association_and_unchanged_reindex_uses_no_embedding_calls(self):
        self.ingest()
        settings = self.settings.model_copy(update={"embedding_batch_size": 3})
        first = self.index(settings=settings)
        self.assertGreater(len(self.provider.calls), 1)
        self.assertTrue(all(1 <= len(batch) <= 3 for batch in self.provider.calls))
        rows = self.store.chunks(self.repository_id, include_vectors=True)
        expected = asyncio.run(DeterministicProvider().embed([row["embedding_input"] for row in rows]))
        self.assertEqual([row["embedding"] for row in rows], expected)
        vectors = self.vectors()
        calls = len(self.provider.calls)
        second = self.index(settings=settings)
        self.assertEqual((second.embedded, second.reused), (0, first.chunks))
        self.assertEqual(len(self.provider.calls), calls)
        self.assertEqual(vectors, self.vectors())

    def test_unchanged_structural_reingestion_preserves_ready_vectors_and_unique_links(self):
        self.ingest()
        first = self.index()
        vectors = self.vectors()
        state = self.store.state(self.repository_id)
        self.ingest()
        self.ingest()
        current = self.store.state(self.repository_id)
        self.assertEqual(current["source_revision"], state["source_revision"])
        self.assertEqual(effective_readiness(current).state, "READY")
        self.assertEqual(vectors, self.vectors())
        counts = self.store.read("MATCH (c:Chunk {repository_id:$id})-[r:EVIDENCE_FOR]->(o) "
                                 "RETURN count(c) AS chunks,count(DISTINCT c.id) AS unique_chunks, "
                                 "count(r) AS links,count(DISTINCT r.id) AS unique_links", id=self.repository_id)[0]
        self.assertEqual(set(counts.values()), {first.chunks})
        calls = len(self.provider.calls)
        self.assertEqual(self.index().embedded, 0)
        self.assertEqual(len(self.provider.calls), calls)

    def test_requested_model_digest_mismatch_blocks_vector_query_before_embedding(self):
        self.ingest()
        self.index()
        self.provider.digest = "b" * 64
        calls = len(self.provider.calls)
        result = self.result("entry flow", channels=("vector",))
        self.assertEqual(result.readiness.state, "DIRTY")
        self.assertIn("embedding_profile_mismatch", result.readiness.reasons)
        self.assertEqual(result.evidence, ())
        self.assertEqual(len(self.provider.calls), calls)
        self.index()
        self.assertEqual(self.result("entry flow", channels=("vector",)).readiness.state, "READY")

    def test_live_cycles_and_parallel_call_sites_preserve_distinct_provenance(self):
        (self.root / "cycles.py").write_text("def first():\n    second()\n    second()\n\ndef second():\n    first()\n")
        self.ingest()
        result = self.result("cycles.first", max_depth=6)
        second = next(item for item in result.evidence if item.qualified_name == "cycles.second")
        self.assertEqual(len(second.graph), 2)
        self.assertEqual({origin.edges[0].start_line for origin in second.graph}, {2, 3})
        self.assertEqual(result.examined_relationships, 3)
        self.assertEqual(result.admitted_entities, 2)
        self.assertEqual(result.model_dump_json(), self.result("cycles.first", max_depth=6).model_dump_json())

    def test_profile_changes_reuse_one_index_and_preserve_unrelated_index(self):
        self.ingest()
        self.index()
        name, _ = self.store.vector_names(self.repository_id)
        unrelated = "external_test_" + secrets.token_hex(8)
        self.store.read(f"CREATE INDEX `{unrelated}` FOR (n:`{unrelated}`) ON (n.value)")
        self.addCleanup(self.store.read, f"DROP INDEX `{unrelated}` IF EXISTS")
        for digest in ["b" * 64, "c" * 64]:
            self.provider.digest = digest
            result = self.index()
            self.assertEqual(result.embedded, result.chunks)
            self.assertTrue(all(row["profile"] == result.profile_id for row in self.vectors()))
        rows = self.store.read("SHOW VECTOR INDEXES YIELD name WHERE name=$name RETURN name", name=name)
        self.assertEqual(rows, [{"name": name}])
        self.assertTrue(self.store.read("SHOW INDEXES YIELD name WHERE name=$name RETURN name", name=unrelated))

    def test_incompatible_existing_index_is_rejected_without_dropping_it(self):
        self.ingest()
        name, label = self.store.vector_names(self.repository_id)
        self.store.read(f"CREATE VECTOR INDEX `{name}` FOR (c:`{label}`) ON c.embedding "
                        "OPTIONS {indexConfig:{`vector.dimensions`:1024, `vector.similarity_function`:'cosine'}}")
        with self.assertRaisesRegex(EvidenceStoreError, "incompatible"):
            self.index()
        row = self.store.read("SHOW VECTOR INDEXES YIELD name,options WHERE name=$name RETURN options", name=name)[0]
        self.assertEqual(row["options"]["indexConfig"]["vector.dimensions"], 1024)

    def test_dimension_mismatch_marks_attempt_failed_without_partial_vectors(self):
        self.ingest()
        self.index()
        before = self.vectors()
        bad = DeterministicProvider("b" * 64)
        async def embed(texts):
            return [[1.0] * 767 for _ in texts]
        bad.embed = embed
        with self.assertRaises(ValueError):
            self.index(provider=bad)
        self.assertEqual(before, self.vectors())
        self.assertEqual(effective_readiness(self.store.state(self.repository_id)).state, "FAILED")

    def test_failed_real_publication_rolls_back_vectors_and_blocks_current_semantics(self):
        self.ingest()
        self.index()
        profile = self.current_profile().model_copy(update={"digest": "b" * 64})
        before = self.vectors()
        current = self.store.state(self.repository_id)
        captured = self.store.begin(self.repository_id, current, profile, "failed-publication")
        chunks = self.store.chunks(self.repository_id, include_vectors=True)
        chunks[-1]["embedding_input_hash"] = "does-not-match"
        with self.assertRaises(ConcurrentUpdate):
            self.store.publish(self.repository_id, captured, "failed-publication", profile,
                               chunks, [unit_vector(767) for _ in chunks])
        self.store.fail(self.repository_id, "failed-publication")
        self.assertEqual(before, self.vectors())
        self.assertEqual(effective_readiness(self.store.state(self.repository_id)).state, "FAILED")
        with self.assertRaises(EvidenceStoreError):
            self.store.vector_candidates(self.repository_id, self.current_profile(), unit_vector(0), 5)

    def test_concurrent_source_change_during_embedding_rejects_publication(self):
        self.ingest()
        self.index()
        worker = self.root / "worker.py"
        worker.write_text(worker.read_text().replace("strip()", "lstrip()"))
        self.ingest()
        def change_source():
            self.provider.hook = None
            worker.write_text(worker.read_text().replace("lstrip()", "rstrip()"))
            self.ingest()
        self.provider.hook = change_source
        with self.assertRaises(ConcurrentUpdate):
            self.index()
        self.assertNotEqual(effective_readiness(self.store.state(self.repository_id)).state, "READY")
        self.assertTrue(any("rstrip" in item.text for item in self.result("worker.transform").evidence))

    def test_superseded_attempt_cannot_publish_or_fail_newer_attempt(self):
        self.ingest()
        self.index()
        profile = self.current_profile()
        first = self.store.begin(self.repository_id, self.store.state(self.repository_id), profile, "first")
        second = self.store.begin(self.repository_id, first, profile, "second")
        chunks = self.store.chunks(self.repository_id, include_vectors=True)
        vectors = [row["embedding"] for row in chunks]
        with self.assertRaises(ConcurrentUpdate):
            self.store.publish(self.repository_id, first, "first", profile, chunks, vectors)
        self.store.fail(self.repository_id, "first")
        self.assertEqual(self.store.state(self.repository_id)["indexing_attempt"], "second")
        self.store.publish(self.repository_id, second, "second", profile, chunks, vectors)
        self.assertEqual(effective_readiness(self.store.state(self.repository_id)).state, "READY")

    def test_model_changes_during_batch_are_rejected(self):
        self.ingest()
        self.index()
        before = self.vectors()
        self.provider.digest = "b" * 64
        self.provider.hook = lambda: setattr(self.provider, "digest", "c" * 64)
        with self.assertRaises(ConcurrentUpdate):
            self.index()
        self.assertEqual(before, self.vectors())
        self.assertEqual(effective_readiness(self.store.state(self.repository_id)).state, "FAILED")

    def test_repository_vector_and_evidence_isolation(self):
        self.ingest()
        self.index()
        other = self.ids[1]
        self.ingest(repository_id=other)
        self.index(repository_id=other)
        result = self.result("entry flow", channels=("vector",))
        identities = {row["id"] for row in self.store.owners(self.repository_id)}
        self.assertTrue(all(item.owner_id in identities for item in result.evidence))
        self.assertTrue(all(seed.owner_id in identities for seed in result.seeds))
        self.assertNotEqual(self.store.vector_names(other), self.store.vector_names(self.repository_id))

    @unittest.skipUnless(os.environ.get("PHASE1B_OLLAMA_TESTS") == "1", "Set PHASE1B_OLLAMA_TESTS=1 for real local embeddings")
    def test_real_ollama_embedding_only_index_and_query(self):
        self.ingest()
        settings = self.settings.model_copy(update={"model_provider": "ollama", "model_embedding_model": "nomic-embed-text",
                                                   "model_generation_model": None})
        async def run():
            provider = create_provider(settings)
            try:
                indexed = await index_repository(settings, self.root, self.repository_id, provider=provider)
                request = RetrievalRequest(repository_id=self.repository_id, query="deployment tracing documentation",
                                           channels=("vector",), graph=False, deadline_seconds=120)
                result = await retrieve(settings, request, provider=provider)
                self.assertEqual(indexed.state, "READY")
                self.assertEqual(result.readiness.state, "READY")
                self.assertTrue(result.evidence)
                self.assertTrue(all(origin.score is not None for item in result.evidence for origin in item.direct))
            finally:
                await provider.aclose()
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
