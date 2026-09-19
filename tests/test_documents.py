from pathlib import Path
import tempfile
import unittest

from backend.ingestion.chunks import document_chunks, source_lines
from backend.ingestion.documents import read_documents
from backend.ingestion.repository import IngestionError, parse_repository


class DocumentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="")

    def test_discovery_is_case_insensitive_and_independent_of_python_source_root(self):
        for name in ["README", "ReadMe.TxT", "docs/Guide.MD"]:
            self.write(name, "Documentation\n")
        self.write("src/app.py", "pass\n")
        self.write("ignored.rst", "not selected")
        parse_repository(self.root, "repo", "src")
        documents = read_documents(self.root, "repo")
        self.assertEqual([doc.path for doc in documents.items], ["README", "ReadMe.TxT", "docs/Guide.MD"])

    def test_exclusions_and_symlinks(self):
        self.write("README.md", "Visible")
        self.write(".git/private.md", "Excluded")
        try:
            (self.root / "linked.md").symlink_to(self.root / "README.md")
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation unavailable")
        docs = read_documents(self.root, "repo")
        self.assertEqual([doc.path for doc in docs.items], ["README.md"])
        self.assertIn("linked.md", docs.skipped_paths)

    def test_utf8_bom_newline_normalization_hash_and_lines(self):
        raw = b"\xef\xbb\xbf# Title\r\nUnicode: \xe2\x80\xa8 inside line\r\n"
        (self.root / "README.md").write_bytes(raw)
        doc = read_documents(self.root, "repo").items[0]
        import hashlib
        self.assertEqual(doc.source_hash, hashlib.sha256(raw).hexdigest())
        self.assertEqual(doc.end_line, 2)
        self.assertEqual(len(source_lines(doc.source)), 2)
        self.assertNotIn("\r", doc.source)

    def test_heading_inside_fence_is_not_section_boundary(self):
        self.write("README.md", "# Top\n```python\n# Not a section\n```\n## Next\nBody\n")
        chunks = document_chunks(read_documents(self.root, "repo"))
        self.assertEqual([(c.start_line, c.end_line) for c in chunks], [(1, 4), (5, 6)])
        self.assertEqual([c.heading for c in chunks], ["Top", "Next"])
        self.assertTrue(all(c.entity_id is None and c.qualified_name is None and c.entity_kind is None for c in chunks))
        self.assertTrue(all(c.document_id == c.owner_id and c.owner_kind == "document" for c in chunks))

    def test_tilde_fences_and_long_fence_closure(self):
        self.write("README.md", "# A\n~~~~\n## inside\n~~~\n## still inside\n~~~~\n# B\n")
        self.assertEqual([c.heading for c in document_chunks(read_documents(self.root, "repo"))], ["A", "B"])

    def test_deterministic_ids_and_manifest_change_delete_rename(self):
        self.write("README.md", "# Initial\nText\n")
        first = read_documents(self.root, "repo")
        self.assertEqual(first, read_documents(self.root, "repo"))
        self.write("README.md", "# Changed\nText\n")
        second = read_documents(self.root, "repo")
        self.assertEqual(first.items[0].id, second.items[0].id)
        self.assertNotEqual(first.digest, second.digest)
        (self.root / "README.md").rename(self.root / "guide.md")
        third = read_documents(self.root, "repo")
        self.assertNotEqual(second.items[0].id, third.items[0].id)
        (self.root / "guide.md").unlink()
        self.assertFalse(read_documents(self.root, "repo").items)

    def test_bad_encoding_and_binary_content_fail_explicitly(self):
        for raw in [b"\xff", b"binary\x00content"]:
            (self.root / "README.md").write_bytes(raw)
            with self.assertRaises(IngestionError):
                read_documents(self.root, "repo")

    def test_empty_document_has_owner_but_no_fabricated_chunk(self):
        self.write("README.md", "")
        docs = read_documents(self.root, "repo")
        self.assertEqual(len(docs.items), 1)
        self.assertEqual(document_chunks(docs), [])

    def test_documentation_ids_are_repository_scoped(self):
        self.write("README.md", "# Title\n")
        self.assertNotEqual(read_documents(self.root, "one").items[0].id,
                            read_documents(self.root, "two").items[0].id)


if __name__ == "__main__":
    unittest.main()
