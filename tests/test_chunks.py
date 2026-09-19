from dataclasses import asdict
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

from backend.ingestion.chunks import document_chunks, lexical_terms, python_chunks, source_lines
from backend.ingestion.documents import read_documents
from backend.ingestion.repository import parse_repository
from tests.support_retrieval import FIXTURE


class ChunkTests(unittest.TestCase):
    def test_all_python_owner_kinds_and_no_nested_body_duplication(self):
        snapshot = parse_repository(FIXTURE, "fixture")
        chunks = python_chunks(snapshot)
        self.assertEqual({c.entity_kind for c in chunks}, {"Module", "Function", "Class", "Method"})
        worker_class = next(c for c in chunks if c.qualified_name == "worker.Worker")
        self.assertNotIn("def run", worker_class.text)
        module = "".join(c.text for c in chunks if c.qualified_name == "worker" and c.entity_kind == "Module")
        self.assertNotIn("def transform", module)
        self.assertNotIn("class Worker", module)

    def test_exact_source_spans_hashes_and_embedding_metadata(self):
        snapshot = parse_repository(FIXTURE, "fixture")
        files = {e.path: e for e in snapshot.entities if e.kind == "Module"}
        for chunk in python_chunks(snapshot):
            expected = "".join(source_lines(files[chunk.path].source)[chunk.start_line - 1:chunk.end_line])
            self.assertEqual(chunk.text, expected)
            self.assertEqual(chunk.file_source_hash, files[chunk.path].source_hash)
            self.assertEqual(chunk.text_hash, hashlib.sha256(expected.encode()).hexdigest())
            self.assertEqual(chunk.embedding_input_hash, hashlib.sha256(chunk.embedding_input.encode()).hexdigest())
            self.assertTrue(chunk.embedding_input.startswith("search_document: "))
            self.assertLessEqual(len(chunk.embedding_input.encode()), 1536)

    def test_copy_to_other_machine_path_and_repeat_are_deterministic(self):
        first = python_chunks(parse_repository(FIXTURE, "fixture"))
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "copy"
            shutil.copytree(FIXTURE, copy)
            self.assertEqual(first, python_chunks(parse_repository(copy, "fixture")))
        self.assertEqual(first, python_chunks(parse_repository(FIXTURE, "fixture")))

    def test_blank_lines_move_citation_without_reembedding_unchanged_function(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.py"
            path.write_text("def work():\n    return 1\n")
            first = next(c for c in python_chunks(parse_repository(directory, "repo")) if c.entity_kind == "Function")
            path.write_text("\n\ndef work():\n    return 1\n")
            second = next(c for c in python_chunks(parse_repository(directory, "repo")) if c.entity_kind == "Function")
            self.assertEqual((first.id, first.embedding_input_hash), (second.id, second.embedding_input_hash))
            self.assertNotEqual(first.start_line, second.start_line)
            self.assertNotEqual(first.file_source_hash, second.file_source_hash)

    def test_nested_function_is_separate_contiguous_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "a.py").write_text("def outer():\n    def inner():\n        return 3\n    return inner()\n")
            chunks = python_chunks(parse_repository(directory, "repo"))
        outer = [c for c in chunks if c.qualified_name == "a.outer"]
        self.assertEqual([(c.start_line, c.end_line) for c in outer], [(1, 1), (4, 4)])
        self.assertFalse(any("return 3" in c.text for c in outer))

    def test_long_region_splits_at_lines_without_losing_text(self):
        with tempfile.TemporaryDirectory() as directory:
            source = "def work():\n" + "    value = 123456789\n" * 40
            Path(directory, "a.py").write_text(source)
            chunks = [c for c in python_chunks(parse_repository(directory, "repo"), 256) if c.entity_kind == "Function"]
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(c.text for c in chunks), source)
        self.assertTrue(all(len(c.embedding_input.encode()) <= 256 for c in chunks))

    def test_indivisible_long_line_fails_without_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "a.py").write_text("value = '" + "x" * 2000 + "'\n")
            with self.assertRaisesRegex(ValueError, "exceeds chunk"):
                python_chunks(parse_repository(directory, "repo"))

    def test_decorators_and_unicode_physical_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "a.py").write_text("@decorate\ndef work():\n    return 'a\u2028b'\n", encoding="utf-8")
            chunk = next(c for c in python_chunks(parse_repository(directory, "repo")) if c.entity_kind == "Function")
        self.assertEqual((chunk.start_line, chunk.end_line), (1, 3))
        self.assertTrue(chunk.text.startswith("@decorate\n"))

    def test_lexical_identifiers_preserve_whole_snake_and_camel_components(self):
        terms = lexical_terms("loadHTTPValue save_result README.md")
        self.assertTrue({"loadhttpvalue", "load", "http", "value", "save_result", "save", "result"} <= set(terms))

    def test_document_text_is_not_structural_python(self):
        snapshot = parse_repository(FIXTURE, "fixture")
        docs = document_chunks(read_documents(FIXTURE, "fixture"))
        self.assertFalse(any(e.name == "documented_example" for e in snapshot.entities))
        self.assertTrue(any("documented_example" in c.text for c in docs))
        self.assertTrue(all(asdict(c)["entity_id"] is None for c in docs))


if __name__ == "__main__":
    unittest.main()
