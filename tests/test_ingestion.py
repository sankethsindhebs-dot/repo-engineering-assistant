"""Deterministic offline tests; database persistence is tested separately."""

import ast
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.config import Settings
from backend.graph.store import GraphWriteError, write_snapshot
from backend.ingestion.repository import IngestionError, parse_repository
from scripts.ingest import main


FIXTURE = Path(__file__).parent / "fixtures/python_repo"


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def parse(self, source, others=None):
        files = {"example.py": source, **(others or {})}
        for name, contents in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        return parse_repository(self.root, "unit")

    @staticmethod
    def targets(snapshot, kind="CALLS"):
        names = {entity.id: entity.qualified_name for entity in snapshot.entities}
        return [(ref.expression, names.get(ref.target_id)) for ref in snapshot.references if ref.kind == kind]

    def test_files_modules_classes_functions_methods_and_nested_containment(self):
        snapshot = parse_repository(FIXTURE, "fixture")
        by_name = {entity.qualified_name: entity for entity in snapshot.entities}
        self.assertEqual(sum(entity.kind == "Module" for entity in snapshot.entities), 4)
        self.assertEqual(by_name["sample.service.Service"].kind, "Class")
        self.assertEqual(by_name["sample.service.Service.run"].kind, "Method")
        self.assertEqual(by_name["sample.service.pipeline"].kind, "Function")
        child = by_name["sample.service.pipeline.<locals>.finish"]
        self.assertEqual(child.kind, "Function")
        self.assertTrue(any(edge.kind == "CONTAINS" and edge.source_id == by_name["sample.service.pipeline"].id
                            and edge.target_id == child.id for edge in snapshot.relationships))

    def test_fixture_local_imports_inheritance_and_call_chain(self):
        snapshot = parse_repository(FIXTURE, "fixture")
        calls = self.targets(snapshot)
        self.assertIn(("helpers.decorate(item)", "sample.helpers.decorate"), calls)
        self.assertIn(("normalize(value)", "sample.helpers.normalize"), calls)
        self.assertIn(("finish(clean(value))", "sample.service.pipeline.<locals>.finish"), calls)
        self.assertIn(("Base", "sample.base.Base"), self.targets(snapshot, "INHERITS"))
        self.assertIn(("from .base import Base", "sample.base.Base"), self.targets(snapshot, "IMPORTS"))

    def test_unresolved_calls_and_external_import_are_retained(self):
        snapshot = parse_repository(FIXTURE, "fixture")
        calls = self.targets(snapshot)
        self.assertIn(("self.clean(cleaned)", None), calls)
        self.assertIn(("json.dumps(cleaned)", None), calls)
        self.assertIn(("value.strip()", None), calls)
        self.assertIn(("json", None), self.targets(snapshot, "IMPORTS"))
        expected = sum(sum(isinstance(node, ast.Call) for node in ast.walk(ast.parse(path.read_text())))
                       for path in FIXTURE.rglob("*.py"))
        self.assertEqual(sum(ref.kind == "CALLS" for ref in snapshot.references), expected)

    def test_unresolved_external_and_expression_bases(self):
        snapshot = self.parse("from external import Base\nclass A(Base): pass\nclass B(factory()): pass\n")
        self.assertEqual(dict(self.targets(snapshot, "INHERITS")), {"Base": None, "factory()": None})
        self.assertIn(("factory()", None), self.targets(snapshot))

    def test_relative_and_aliased_module_imports(self):
        snapshot = self.parse("import package.helpers as h\ndef run(): return h.work()\n", {
            "package/__init__.py": "", "package/helpers.py": "def work(): return 1\n",
        })
        self.assertIn(("h.work()", "package.helpers.work"), self.targets(snapshot))

    def test_relative_import_without_parent_package_stays_unresolved(self):
        snapshot = self.parse("from .helper import work\nwork()\n", {
            "helper.py": "def work(): pass\n",
        })
        self.assertIn(("from .helper import work", None), self.targets(snapshot, "IMPORTS"))
        self.assertIn(("work()", None), self.targets(snapshot))

    def test_explicit_dotted_import_with_namespace_package(self):
        snapshot = self.parse("import package.helpers\ndef run(): return package.helpers.work()\n", {
            "package/helpers.py": "def work(): return 1\n",
        })
        self.assertIn(("package.helpers.work()", "package.helpers.work"), self.targets(snapshot))

    def test_nonimportable_filename_is_indexed_without_inventing_an_import(self):
        snapshot = self.parse("import unusual.name\ndef run(): return unusual.name.work()\n", {
            "unusual.name.py": "def work(): return 1\n",
        })
        self.assertEqual(len(snapshot.entities), 4)
        self.assertIn(("unusual.name", None), self.targets(snapshot, "IMPORTS"))
        self.assertIn(("unusual.name.work()", None), self.targets(snapshot))

    def test_reexports_and_cyclic_imports(self):
        snapshot = self.parse("from facade import work\ndef run(): return work()\n", {
            "facade.py": "from helpers import work\n", "helpers.py": "def work(): return 1\n",
            "a.py": "from b import loop\n", "b.py": "from a import loop\n",
        })
        self.assertIn(("work()", "helpers.work"), self.targets(snapshot))
        self.assertTrue(any(ref.reason == "cyclic_import_or_alias" for ref in snapshot.references))

    def test_shadowed_rebound_conditional_and_duplicate_bindings_are_unresolved(self):
        cases = [
            "def work(): pass\ndef run(work): return work()\n",
            "def work(): pass\nwork = other\nwork()\n",
            "if flag:\n def work(): pass\nwork()\n",
            "def work(): pass\ndef work(): pass\nwork()\n",
            "from other import *\ndef work(): pass\nwork()\n",
            "def work(): pass\ndef run():\n global work\n work = other\nwork()\n",
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertIn(("work()", None), self.targets(self.parse(source)))

    def test_assignment_aliases_and_runtime_arguments_are_not_guessed(self):
        snapshot = self.parse("def work(): pass\nalias = work\nalias()\ndef run(callback):\n callback()\n")
        self.assertEqual(dict(self.targets(snapshot)), {"alias()": None, "callback()": None})

    def test_nested_calls_at_same_start_column_have_distinct_evidence(self):
        snapshot = self.parse("def factory(): pass\nfactory()()\n")
        calls = [ref for ref in snapshot.references if ref.kind == "CALLS"]
        self.assertEqual(len({ref.id for ref in calls}), 2)
        self.assertIn(("factory()", "example.factory"), self.targets(snapshot))
        self.assertIn(("factory()()", None), self.targets(snapshot))

    def test_repeated_import_names_keep_both_reference_sites(self):
        snapshot = self.parse("from helper import work, work\n", {"helper.py": "def work(): pass\n"})
        imports = [ref for ref in snapshot.references if ref.kind == "IMPORTS"]
        self.assertEqual(len({ref.id for ref in imports}), 2)
        self.assertTrue(all(ref.target_id for ref in imports))

    def test_class_namespace_is_not_method_lexical_scope(self):
        snapshot = self.parse("class A:\n def work(self): pass\n def run(self): return work()\n")
        self.assertIn(("work()", None), self.targets(snapshot))

    def test_nested_class_inside_function_and_nested_function_inside_method(self):
        snapshot = self.parse("def outer():\n class Inner:\n  def run(self):\n   def inner(): pass\n   inner()\n")
        kinds = {entity.qualified_name: entity.kind for entity in snapshot.entities}
        self.assertEqual(kinds["example.outer.<locals>.Inner.run"], "Method")
        self.assertEqual(kinds["example.outer.<locals>.Inner.run.<locals>.inner"], "Function")
        self.assertIn(("inner()", "example.outer.<locals>.Inner.run.<locals>.inner"), self.targets(snapshot))

    def test_decorated_targets_are_unresolved(self):
        snapshot = self.parse("def decorator(fn): return fn\n@decorator\ndef work(): pass\nwork()\n")
        self.assertIn(("work()", None), self.targets(snapshot))

    def test_rebound_import_alias_attribute_is_not_resolved(self):
        snapshot = self.parse("import facade\nimport helper\nfacade.job = other\nfacade.job()\nhelper.work()\n", {
            "facade.py": "from helper import work as job\n", "helper.py": "def work(): pass\n",
        })
        self.assertIn(("facade.job()", None), self.targets(snapshot))
        self.assertIn(("helper.work()", "helper.work"), self.targets(snapshot))
        call = next(ref for ref in snapshot.references if ref.expression == "facade.job()")
        self.assertEqual(call.reason, "attribute_rebinding_present")

    def test_unidentified_attribute_receiver_does_not_invalidate_lexical_call(self):
        snapshot = self.parse("def work(): pass\nmodule.work = other\nwork()\n")
        self.assertIn(("work()", "example.work"), self.targets(snapshot))

    def test_unrelated_attribute_write_in_another_file_preserves_lexical_call(self):
        snapshot = self.parse("", {
            "a.py": "def work(): pass\nwork()\n", "b.py": "obj.work = other\n",
        })
        self.assertIn(("work()", "a.work"), self.targets(snapshot))

    def test_unrelated_attribute_write_preserves_imported_module_call(self):
        snapshot = self.parse("", {
            "a.py": "def work(): pass\n", "b.py": "import a\na.work()\n",
            "c.py": "obj.work = other\n",
        })
        self.assertIn(("a.work()", "a.work"), self.targets(snapshot))

    def test_aliased_module_writes_only_invalidate_that_module_binding(self):
        for mutation in ("alias.work = other", "del alias.work", "alias.work += other"):
            with self.subTest(mutation=mutation):
                snapshot = self.parse(
                    f"import a as alias\n{mutation}\nalias.work()\ndef work(): pass\nwork()\n",
                    {"a.py": "def work(): pass\n"},
                )
                self.assertIn(("alias.work()", None), self.targets(snapshot))
                self.assertIn(("work()", "example.work"), self.targets(snapshot))

    def test_parameter_receiver_does_not_acquire_imported_module_identity(self):
        snapshot = self.parse(
            "import a\ndef run(a):\n a.work = other\n a.work()\na.work()\n",
            {"a.py": "def work(): pass\n"},
        )
        self.assertIn(("a.work()", None), self.targets(snapshot))
        self.assertIn(("a.work()", "a.work"), self.targets(snapshot))

    def test_dotted_module_mutations_are_scoped_to_the_imported_path(self):
        cases = [("package.helper.work = other", None), ("package.helper = other", None),
                 ("import example\nexample.package = other", None),
                 ("obj.helper = other\nobj.work = other", "package.helper.work")]
        for mutation, target in cases:
            with self.subTest(mutation=mutation):
                snapshot = self.parse(
                    f"import package.helper\n{mutation}\npackage.helper.work()\n",
                    {"package/helper.py": "def work(): pass\n"},
                )
                self.assertIn(("package.helper.work()", target), self.targets(snapshot))

    def test_wildcard_package_does_not_invent_submodule_binding(self):
        snapshot = self.parse("from package import helper\nhelper.work()\n", {
            "package/__init__.py": "from external import *\n", "package/helper.py": "def work(): pass\n",
        })
        self.assertIn(("helper.work()", None), self.targets(snapshot))

    def test_calls_before_definition_and_same_line_are_conservative(self):
        snapshot = self.parse("work()\ndef work(): pass\nwork()\n")
        calls = sorted((ref for ref in snapshot.references if ref.kind == "CALLS"), key=lambda ref: ref.start_line)
        self.assertIsNone(calls[0].target_id)
        self.assertIsNotNone(calls[1].target_id)

    def test_recursion_and_async_functions(self):
        snapshot = self.parse("async def work():\n return await work()\n")
        self.assertIn(("work()", "example.work"), self.targets(snapshot))

    def test_defaults_are_parent_calls_and_annotations_are_explicitly_unresolved(self):
        snapshot = self.parse("def make(): pass\ndef work(x: make() = make()) -> make(): pass\n")
        refs = [ref for ref in snapshot.references if ref.kind == "CALLS"]
        self.assertEqual(len(refs), 3)
        self.assertEqual(sum(ref.target_id is not None for ref in refs), 1)
        self.assertEqual(sum(ref.reason == "annotation_context" for ref in refs), 2)
        module = next(entity for entity in snapshot.entities if entity.kind == "Module")
        self.assertTrue(all(ref.owner_id == module.id for ref in refs))

    def test_lambda_comprehension_and_dynamic_import_calls_are_preserved(self):
        snapshot = self.parse("def work(): pass\na = lambda: work()\nb = [work() for x in items]\n__import__('external')\n")
        refs = [ref for ref in snapshot.references if ref.kind == "CALLS"]
        self.assertEqual(len(refs), 3)
        self.assertTrue(all(ref.target_id is None for ref in refs))
        self.assertEqual({ref.reason for ref in refs}, {"lambda_scope", "comprehension_scope", "unknown_or_external_name"})

    def test_duplicate_declarations_get_distinct_ids(self):
        snapshot = self.parse("def work(): pass\ndef work(): pass\n")
        definitions = [entity for entity in snapshot.entities if entity.kind == "Function"]
        self.assertEqual(len({entity.id for entity in definitions}), 2)

    def test_stable_ids_across_locations_and_blank_line_edits(self):
        snapshot = self.parse("def work():\n return 1\n")
        with tempfile.TemporaryDirectory() as other:
            Path(other, "example.py").write_text("def work():\n return 1\n")
            second = parse_repository(other, "unit")
        self.assertEqual(snapshot.to_json(), second.to_json())
        changed = self.parse("\n\ndef work():\n return 1\n")
        self.assertEqual([entity.id for entity in snapshot.entities], [entity.id for entity in changed.entities])
        self.assertNotEqual(snapshot.digest, changed.digest)
        separate = parse_repository(self.root, "other-repository")
        self.assertFalse({entity.id for entity in snapshot.entities} & {entity.id for entity in separate.entities})

    def test_same_repository_snapshot_has_identical_records_on_rerun(self):
        first = parse_repository(FIXTURE, "fixture")
        second = parse_repository(FIXTURE, "fixture")
        self.assertEqual(first, second)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(len(first.relationships), len({edge.id for edge in first.relationships}))

    def test_source_line_hash_and_utf8_byte_column_metadata(self):
        source = "# note\ndef work():\n    return 1\n\nx = 'é'; work()\n"
        snapshot = self.parse(source)
        function = next(entity for entity in snapshot.entities if entity.kind == "Function")
        self.assertEqual((function.start_line, function.end_line), (2, 3))
        self.assertEqual(function.source, "def work():\n    return 1\n")
        self.assertEqual(function.source_hash, hashlib.sha256((self.root / "example.py").read_bytes()).hexdigest())
        ref = next(ref for ref in snapshot.references if ref.kind == "CALLS")
        self.assertEqual((ref.start_line, ref.end_line, ref.start_column, ref.end_column), (5, 5, 10, 16))
        self.assertTrue(all(entity.path == "example.py" for entity in snapshot.entities))
        self.assertNotIn(str(self.root), snapshot.to_json())

    def test_decorator_source_span_and_empty_module(self):
        snapshot = self.parse("@wrapper\ndef work():\n pass\n", {"empty.py": ""})
        function = next(entity for entity in snapshot.entities if entity.kind == "Function")
        self.assertEqual((function.start_line, function.end_line), (1, 3))
        self.assertTrue(function.source.startswith("@wrapper"))
        empty = next(entity for entity in snapshot.entities if entity.path == "empty.py")
        self.assertEqual((empty.start_line, empty.end_line, empty.source), (1, 1, ""))

    def test_encoding_cookie_is_respected(self):
        Path(self.root, "encoded.py").write_bytes(b"# coding: latin-1\nname = 'caf\xe9'\n")
        snapshot = parse_repository(self.root, "unit")
        self.assertIn("café", snapshot.entities[0].source)

    def test_newline_normalization_and_unicode_separator_do_not_break_spans(self):
        raw = "def work():\r\n    return 'a\u2028b'\r\n\r\n".encode()
        Path(self.root, "example.py").write_bytes(raw)
        snapshot = parse_repository(self.root, "unit")
        module = next(entity for entity in snapshot.entities if entity.kind == "Module")
        function = next(entity for entity in snapshot.entities if entity.kind == "Function")
        self.assertEqual(module.end_line, 3)
        self.assertEqual(function.end_line, 2)
        self.assertEqual(function.source, "def work():\n    return 'a\u2028b'\n")
        self.assertEqual(module.source_hash, hashlib.sha256(raw).hexdigest())

    def test_match_capture_exception_and_loop_targets_shadow_names(self):
        cases = [
            "def work(): pass\nfor work in items:\n work()\n",
            "def work(): pass\ntry: pass\nexcept Exception as work: work()\n",
            "def work(): pass\nmatch value:\n case {'x': work}: work()\n",
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertIn(("work()", None), self.targets(self.parse(source)))

    def test_parsing_never_executes_source(self):
        snapshot = self.parse("raise RuntimeError('must not execute')\n")
        self.assertEqual(len(snapshot.entities), 1)
        self.assertIn(("RuntimeError('must not execute')", None), self.targets(snapshot))

    def test_malformed_compile_invalid_and_unsupported_python_rejected(self):
        for source in ("def broken(\n", "return 1\n", "print 'Python 2'\n", "a = '\x00'\n"):
            with self.subTest(source=source), self.assertRaises(IngestionError):
                self.parse(source)

    def test_module_collision_rejected(self):
        with self.assertRaisesRegex(IngestionError, "Ambiguous module"):
            self.parse("", {"same.py": "", "same/__init__.py": ""})

    def test_source_root_and_exclusions(self):
        self.parse("", {"src/sample.py": "def work(): pass\n", "src/.venv/broken.py": "not python !"})
        snapshot = parse_repository(self.root, "unit", "src")
        self.assertEqual({entity.path for entity in snapshot.entities}, {"src/sample.py"})
        self.assertEqual({entity.module_name for entity in snapshot.entities}, {"sample"})
        self.assertEqual(snapshot.skipped_paths, ("src/.venv/",))
        with self.assertRaises(IngestionError):
            parse_repository(self.root, "unit", "../outside")

    def test_symlinks_are_not_followed(self):
        self.parse("def work(): pass\n")
        try:
            Path(self.root, "linked.py").symlink_to(self.root / "example.py")
        except OSError:
            self.skipTest("Creating symlinks is unavailable")
        snapshot = parse_repository(self.root, "unit")
        self.assertIn("linked.py", snapshot.skipped_paths)
        self.assertEqual(len(snapshot.entities), 2)

    def test_empty_repository_and_invalid_id_rejected(self):
        for identity in ("unit", "../bad", "/absolute", ""):
            with self.assertRaises(IngestionError):
                parse_repository(self.root, identity)

    def test_invalid_snapshot_rejected_before_driver_is_opened(self):
        snapshot = parse_repository(FIXTURE, "fixture")
        invalid = replace(snapshot, entities=snapshot.entities[:-1])
        with patch("backend.graph.store.GraphDatabase.driver") as driver:
            with self.assertRaises(ValueError):
                write_snapshot(Settings(_env_file=None), invalid)
        driver.assert_not_called()

    def test_unconfigured_write_is_explicit_and_offline(self):
        with patch.dict("os.environ", {}, clear=True), patch("backend.graph.store.GraphDatabase.driver") as driver:
            with self.assertRaisesRegex(GraphWriteError, "not configured"):
                write_snapshot(Settings(_env_file=None), parse_repository(FIXTURE, "fixture"))
        driver.assert_not_called()

    def test_cli_dry_run_does_not_create_settings_or_open_database(self):
        with patch("scripts.ingest.Settings") as settings, patch("scripts.ingest.write_snapshot") as writer:
            with redirect_stdout(io.StringIO()) as output:
                code = main([str(FIXTURE), "--repository-id", "fixture", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn('"mode": "parse_only"', output.getvalue())
        settings.assert_not_called()
        writer.assert_not_called()

    def test_cli_malformed_input_never_calls_writer(self):
        Path(self.root, "broken.py").write_text("def broken(\n")
        with patch("scripts.ingest.write_snapshot") as writer, redirect_stderr(io.StringIO()):
            self.assertEqual(main([str(self.root), "--repository-id", "unit"]), 1)
        writer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
