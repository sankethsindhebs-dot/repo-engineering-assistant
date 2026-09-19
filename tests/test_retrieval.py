import time
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from backend.config import Settings
from backend.graph.evidence_store import ConcurrentUpdate
from backend.retrieval.models import RetrievalRequest, TraversalProfile, TraversalStep
from backend.retrieval.retrieve import _retrieve_once, direct_seeds, expand_graph, retrieve
from tests.support_retrieval import OfflineGraph


def profile(kind="CALLS", direction="outgoing", **kwargs):
    return TraversalProfile(steps=(TraversalStep(relationship=kind, direction=direction),), **kwargs)


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = OfflineGraph()
        self.settings = Settings(_env_file=None)

    async def result(self, query="handle_request", **kwargs):
        request = RetrievalRequest(repository_id="fixture", query=query, channels=("lexical",),
                                   traversal=profile(), **kwargs)
        return await _retrieve_once(self.settings, self.store, request, None, time.monotonic() + 10)

    async def test_graph_off_on_and_edge_intervention(self):
        off = await self.result(graph=False)
        on = await self.result()
        self.assertEqual(off.seeds, on.seeds)
        self.assertFalse(any("casefold" in item.text for item in off.evidence))
        implementation = next(item for item in on.evidence if item.qualified_name == "worker.transform")
        self.assertIn("casefold", implementation.text)
        self.assertEqual(implementation.graph[0].edges[0].relationship, "CALLS")
        entry = self.store.entity("entry.handle_request")["id"]
        self.store.edges = [e for e in self.store.edges if not (e["source_id"] == entry and e["kind"] == "CALLS")]
        changed = await self.result()
        self.assertEqual(on.seeds, changed.seeds)
        self.assertFalse(any(item.qualified_name == "worker.transform" for item in changed.evidence))

    async def test_unresolved_callback_is_diagnostic_not_a_path(self):
        result = await self.result("via_runtime")
        self.assertTrue(any("callback" in ref.expression for ref in result.unresolved))
        self.assertFalse(any(item.graph for item in result.evidence))

    async def test_document_lexical_evidence_has_no_fake_entity_or_graph_path(self):
        result = await self.result("amber")
        item = next(item for item in result.evidence if item.owner_kind == "document")
        self.assertEqual(item.path, "README.md")
        self.assertIsNone(item.entity_id)
        self.assertIsNone(item.qualified_name)
        self.assertEqual(item.document_id, item.owner_id)
        self.assertEqual(item.graph, ())

    async def test_repeat_result_is_deterministic(self):
        self.assertEqual((await self.result()).model_dump_json(), (await self.result()).model_dump_json())

    async def test_source_budget_never_silently_truncates_a_chunk(self):
        result = await self.result(source_budget_bytes=1)
        self.assertEqual(result.evidence, ())
        self.assertIn("source_budget", result.bounds_reached)

    async def test_exact_path_and_qualified_name(self):
        for query, expected in [("docs\\usage.md", "document"), ("entry.handle_request", "entity")]:
            request = RetrievalRequest(repository_id="fixture", query=query, channels=("exact",), graph=False)
            result = await _retrieve_once(self.settings, self.store, request, None, time.monotonic() + 10)
            self.assertTrue(result.seeds)
            self.assertEqual(result.seeds[0].owner_kind, expected)

    async def test_direct_channels_merge_without_duplicate_evidence(self):
        request = RetrievalRequest(repository_id="fixture", query="handle_request", channels=("exact", "lexical"), graph=False)
        result = await _retrieve_once(self.settings, self.store, request, None, time.monotonic() + 10)
        self.assertEqual(len({item.chunk_id for item in result.evidence}), len(result.evidence))
        item = next(item for item in result.evidence if item.qualified_name == "entry.handle_request")
        self.assertEqual({origin.channel for origin in item.direct}, {"exact", "lexical"})

    async def test_expired_deadline_is_explicit(self):
        request = RetrievalRequest(repository_id="fixture", query="handle_request", channels=("exact",))
        with self.assertRaises(TimeoutError):
            await _retrieve_once(self.settings, self.store, request, None, time.monotonic() - 1)

    async def test_concurrent_read_retries_once_and_never_returns_mixed_revision(self):
        request = RetrievalRequest(repository_id="fixture", query="handle_request", channels=("exact",))
        wanted = await _retrieve_once(self.settings, self.store, request, None, time.monotonic() + 10)
        with patch("backend.retrieval.retrieve.EvidenceStore") as boundary:
            boundary.return_value.__enter__.return_value = self.store
            with patch("backend.retrieval.retrieve._retrieve_once", side_effect=[ConcurrentUpdate("changed"), wanted]) as read:
                self.assertEqual(await retrieve(self.settings, request), wanted)
                self.assertEqual(read.call_count, 2)
            with patch("backend.retrieval.retrieve._retrieve_once", side_effect=ConcurrentUpdate("changed")) as read:
                with self.assertRaises(ConcurrentUpdate):
                    await retrieve(self.settings, request)
                self.assertEqual(read.call_count, 2)

    async def test_short_name_ambiguity_retains_distinct_exact_seeds(self):
        source = self.store.entity("entry.handle_request")
        self.store.owner_rows.append(source | {"id": "ambiguous-owner", "qualified_name": "other.handle_request", "path": "other.py"})
        request = RetrievalRequest(repository_id="fixture", query="handle_request", channels=("exact",), graph=False)
        result = await _retrieve_once(self.settings, self.store, request, None, time.monotonic() + 10)
        self.assertEqual({seed.owner_id for seed in result.seeds}, {source["id"], "ambiguous-owner"})


class TraversalTests(unittest.TestCase):
    def setUp(self):
        self.store = OfflineGraph()

    def traverse(self, query, traversal):
        request = RetrievalRequest(repository_id="fixture", query=query, channels=("exact",), traversal=traversal)
        seeds = direct_seeds(request, self.store.owner_rows, self.store.rows, [])
        return expand_graph(self.store, "fixture", seeds, traversal, time.monotonic() + 10)

    def names(self, ids):
        return {row.get("qualified_name") for row in self.store.owner_rows if row["id"] in ids}

    def test_incoming_calls_returns_callers_and_preserves_edge_direction(self):
        origins, admitted, _, _ = self.traverse("worker.transform", profile(direction="incoming"))
        self.assertIn("entry.handle_request", self.names(admitted))
        caller = self.store.entity("entry.handle_request")["id"]
        edge = origins[caller][0].edges[0]
        self.assertEqual(edge.direction, "incoming")
        self.assertEqual(edge.stored_source_id, caller)
        self.assertEqual(edge.stored_target_id, self.store.entity("worker.transform")["id"])

    def test_incoming_imports_and_inherits(self):
        for query, kind, expected in [("worker", "IMPORTS", "entry"), ("base.Base", "INHERITS", "worker.Worker")]:
            with self.subTest(kind=kind):
                _, admitted, _, _ = self.traverse(query, profile(kind, "incoming"))
                self.assertIn(expected, self.names(admitted))

    def test_outgoing_profile_does_not_implicitly_find_callers(self):
        _, admitted, _, _ = self.traverse("worker.transform", profile())
        self.assertNotIn("entry.handle_request", self.names(admitted))

    def test_parent_context_terminal_or_continued_ascent(self):
        _, terminal, _, _ = self.traverse("worker.Worker.run", profile("CONTAINS", "incoming", parent_context="terminal"))
        _, ascent, _, _ = self.traverse("worker.Worker.run", profile("CONTAINS", "incoming", parent_context="ascend"))
        self.assertIn("worker.Worker", self.names(terminal))
        self.assertNotIn("worker", self.names(terminal))
        self.assertIn("worker", self.names(ascent))

    def test_outgoing_imports_inherits_and_contains(self):
        for query, kind, expected in [("entry", "IMPORTS", "worker"), ("worker.Worker", "INHERITS", "base.Base"),
                                       ("worker.Worker", "CONTAINS", "worker.Worker.run")]:
            _, admitted, _, _ = self.traverse(query, profile(kind))
            self.assertIn(expected, self.names(admitted))

    def test_depth_node_edge_and_branch_budgets_limit_actual_expansion(self):
        for options, expected in [({"max_depth": 0}, 1), ({"max_nodes": 1}, 1), ({"max_edges": 0}, 1)]:
            _, admitted, examined, _ = self.traverse("entry.handle_request", profile(**options))
            self.assertEqual(len(admitted), expected)
            if "max_edges" in options:
                self.assertEqual(examined, 0)
        _, admitted, examined, bounds = self.traverse("worker.transform", profile(direction="incoming", branch_size=1))
        self.assertEqual(examined, 1)
        self.assertEqual(len(admitted), 2)
        self.assertIn("branch_limit", bounds)

    def test_cycles_and_duplicate_targets_terminate(self):
        source = self.store.entity("entry.handle_request")["id"]
        target = self.store.entity("worker.transform")["id"]
        self.store.edges += [{"id": "cycle", "source_id": target, "target_id": source, "kind": "CALLS"},
                             {"id": "parallel", "source_id": source, "target_id": target, "kind": "CALLS"}]
        origins, admitted, examined, _ = self.traverse("entry.handle_request", profile(max_depth=6))
        self.assertEqual(len(admitted), 2)
        self.assertEqual(len(origins[target]), 2)
        self.assertEqual(len({origin.edges[0].relationship_id for origin in origins[target]}), 2)
        self.assertEqual(examined, 3)

    def test_limits_and_step_combinations_are_validated(self):
        for options in [{"max_depth": -1}, {"max_nodes": 0}, {"branch_size": 0}, {"max_edges": 1001}]:
            with self.assertRaises(ValidationError):
                TraversalProfile(**options)
        step = TraversalStep(relationship="CALLS", direction="incoming")
        with self.assertRaises(ValidationError):
            TraversalProfile(steps=(step, step))
        with self.assertRaises(ValidationError):
            TraversalStep(relationship="EVIDENCE_FOR", direction="outgoing")


if __name__ == "__main__":
    unittest.main()
