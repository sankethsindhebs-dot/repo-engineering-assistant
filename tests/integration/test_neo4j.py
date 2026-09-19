"""Opt-in tests against real Neo4j. Each test owns and cleans unique repository IDs."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import secrets
import tempfile
import unittest
from unittest.mock import patch

from neo4j import GraphDatabase

from backend.config import NEO4J_VERSION, Settings
from backend.graph.store import SCHEMA, write_snapshot
from backend.ingestion.repository import parse_repository
from scripts.ingest import main


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/python_repo"


@unittest.skipUnless(os.environ.get("PHASE1A_NEO4J_TESTS") == "1", "Set PHASE1A_NEO4J_TESTS=1 for live writes")
class Neo4jIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        if self.settings.neo4j_password is None:
            self.fail("Live integration tests require configured Neo4j credentials")
        self.prefix = "phase1a-test-" + secrets.token_hex(8)
        self.repository_ids = [self.prefix, self.prefix + "-other"]
        self.driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_username, self.settings.neo4j_password.get_secret_value()),
        )
        self.addCleanup(self.driver.close)
        self.addCleanup(self.cleanup_graph)
        version = self.query("CALL dbms.components() YIELD versions RETURN versions[0] AS version")[0]["version"]
        self.assertEqual(version, NEO4J_VERSION)
        self.snapshot = parse_repository(FIXTURE, self.prefix)

    def query(self, cypher, **parameters):
        with self.driver.session(database=self.settings.neo4j_database) as session:
            return session.run(cypher, **parameters).data()

    def cleanup_graph(self):
        # No blanket DELETE: cleanup is restricted to IDs allocated by this test.
        self.query("MATCH (n:Entity) WHERE n.repository_id IN $ids DETACH DELETE n", ids=self.repository_ids)
        self.query("MATCH (n:IngestionState) WHERE n.repository_id IN $ids DELETE n", ids=self.repository_ids)

    def records(self, repository_id=None):
        identity = repository_id or self.prefix
        nodes = self.query(
            "MATCH (n:Entity {repository_id:$id}) RETURN properties(n) AS properties, labels(n) AS labels ORDER BY n.id",
            id=identity,
        )
        edges = self.query(
            "MATCH (:Entity {repository_id:$id})-[r]->(:Entity) "
            "RETURN properties(r) AS properties, type(r) AS kind ORDER BY r.id", id=identity,
        )
        for node in nodes:
            node["labels"].sort()
        return nodes, edges

    def test_roundtrip_labels_evidence_and_minimal_schema(self):
        result = write_snapshot(self.settings, self.snapshot)
        nodes, edges = self.records()
        self.assertEqual((len(nodes), len(edges)), (result.entities, result.relationships))
        persisted = {row["properties"]["id"]: row for row in nodes}
        for entity in self.snapshot.entities:
            node = persisted[entity.id]
            for name in ("path", "qualified_name", "start_line", "end_line", "source", "source_hash", "kind"):
                self.assertEqual(node["properties"][name], getattr(entity, name))
            if entity.kind == "Module":
                self.assertEqual(node["labels"], ["Entity", "File", "Module"])
            if entity.kind == "Method":
                self.assertEqual(node["labels"], ["Entity", "Function", "Method"])
        saved_refs = [ref for node in nodes for ref in json.loads(node["properties"]["references_json"])]
        self.assertEqual(len(saved_refs), len(self.snapshot.references))
        self.assertEqual(sum(ref["target_id"] is None for ref in saved_refs), result.unresolved_references)
        self.assertEqual({row["kind"] for row in edges}, {"CONTAINS", "IMPORTS", "CALLS", "INHERITS"})
        constraints = {row["name"] for row in self.query("SHOW CONSTRAINTS YIELD name RETURN name")}
        self.assertTrue({"entity_id", "ingestion_repository"} <= constraints)
        index = self.query("SHOW INDEXES YIELD name, type WHERE name = 'entity_repository' RETURN type")
        self.assertEqual(index, [{"type": "RANGE"}])

    def test_reingestion_has_identical_ids_properties_counts_and_no_duplicates(self):
        first = write_snapshot(self.settings, self.snapshot)
        before = self.records()
        second = write_snapshot(self.settings, self.snapshot)
        self.assertEqual(first, second)
        self.assertEqual(before, self.records())
        counts = self.query(
            "MATCH (n:Entity {repository_id:$id}) RETURN count(n) AS total, count(DISTINCT n.id) AS unique",
            id=self.prefix,
        )[0]
        self.assertEqual(counts["total"], counts["unique"])
        counts = self.query(
            "MATCH (:Entity {repository_id:$id})-[r]->() RETURN count(r) AS total, count(DISTINCT r.id) AS unique",
            id=self.prefix,
        )[0]
        self.assertEqual(counts["total"], counts["unique"])

    def test_persisted_local_call_chain_connects_source_evidence(self):
        write_snapshot(self.settings, self.snapshot)
        rows = self.query(
            "MATCH (a:Function {repository_id:$id, qualified_name:'sample.service.pipeline.<locals>.finish'})"
            "-[:CALLS]->(b:Function)-[:CALLS]->(c:Function) "
            "RETURN b.qualified_name AS middle, c.qualified_name AS target, c.path AS path, c.start_line AS line",
            id=self.prefix,
        )
        self.assertEqual(rows, [{"middle": "sample.helpers.decorate", "target": "sample.helpers.normalize",
                                 "path": "sample/helpers.py", "line": 1}])

    def test_changed_snapshot_removes_stale_entities_and_relationships(self):
        write_snapshot(self.settings, self.snapshot)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "new.py").write_text("def replacement(): return 1\n")
            changed = parse_repository(directory, self.prefix)
        write_snapshot(self.settings, changed)
        nodes, edges = self.records()
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(edges), 1)
        self.assertEqual({node["properties"]["path"] for node in nodes}, {"new.py"})

    def test_other_repository_is_unchanged(self):
        other = parse_repository(FIXTURE, self.repository_ids[1])
        write_snapshot(self.settings, other)
        before = self.records(self.repository_ids[1])
        write_snapshot(self.settings, self.snapshot)
        write_snapshot(self.settings, self.snapshot)
        self.assertEqual(before, self.records(self.repository_ids[1]))

    def test_malformed_input_leaves_existing_graph_unchanged(self):
        write_snapshot(self.settings, self.snapshot)
        before = self.records()
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "valid.py").write_text("def valid(): pass\n")
            Path(directory, "broken.py").write_text("def broken(\n")
            with redirect_stderr(io.StringIO()):
                code = main([directory, "--repository-id", self.prefix])
        self.assertEqual(code, 1)
        self.assertEqual(before, self.records())

    def test_failed_data_transaction_rolls_back_deletion(self):
        write_snapshot(self.settings, self.snapshot)
        before = self.records()
        # Valid structure but an unsupported Neo4j property causes a real database
        # failure after DELETE has executed in the same transaction.
        entity = replace(self.snapshot.entities[0], source={"invalid": "property-map"})
        invalid = replace(self.snapshot, entities=(entity, *self.snapshot.entities[1:]))
        from backend.graph.store import GraphWriteError
        with self.assertRaises(GraphWriteError):
            write_snapshot(self.settings, invalid)
        self.assertEqual(before, self.records())

    def test_concurrent_same_repository_ingestion_is_serialized(self):
        # Include simultaneous first ingestion, when the metadata node is absent.
        with self.driver.session(database=self.settings.neo4j_database) as session:
            for statement in SCHEMA:
                session.run(statement).consume()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: write_snapshot(self.settings, self.snapshot), range(2)))
        self.assertEqual(results[0], results[1])
        nodes, edges = self.records()
        self.assertEqual((len(nodes), len(edges)), (len(self.snapshot.entities), len(self.snapshot.relationships)))

    def test_cli_persists_without_model_provider(self):
        with patch("scripts.ingest.Settings", return_value=self.settings), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main([str(FIXTURE), "--repository-id", self.prefix]), 0)
        self.assertEqual(json.loads(output.getvalue())["mode"], "persisted")
        self.assertEqual(len(self.records()[0]), len(self.snapshot.entities))


if __name__ == "__main__":
    unittest.main()
