"""Build a complete, deterministic snapshot before opening a graph write transaction."""

import ast
import hashlib
import io
import keyword
import os
from pathlib import Path, PurePosixPath
import re
import tokenize

from backend.ingestion.models import Entity, Snapshot, stable_id
from backend.ingestion.parser import Extractor
from backend.ingestion.resolver import Resolver


EXCLUDED_DIRECTORIES = frozenset({".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist"})


class IngestionError(ValueError):
    """Input could not be completely indexed; existing graph data must be retained."""


def discover(root: Path, source: Path) -> tuple[list[Path], list[str]]:
    files, skipped = [], []

    def failed_walk(error: OSError) -> None:
        raise IngestionError("Source directory could not be read") from None

    for directory, names, filenames in os.walk(source, followlinks=False, onerror=failed_walk):
        current = Path(directory)
        for name in sorted(names):
            path = current / name
            if name in EXCLUDED_DIRECTORIES or path.is_symlink():
                skipped.append(path.relative_to(root).as_posix() + "/")
        names[:] = sorted(name for name in names if name not in EXCLUDED_DIRECTORIES and not (current / name).is_symlink())
        for name in sorted(filenames):
            if name.endswith(".py"):
                path = current / name
                if path.is_symlink():
                    skipped.append(path.relative_to(root).as_posix())
                elif not path.is_file():
                    raise IngestionError(f"{path.relative_to(root).as_posix()}: not a regular source file")
                else:
                    files.append(path)
    return sorted(files), sorted(skipped)


def parse_file(path: Path, root: Path, source_root: Path, repository_id: str) -> Extractor:
    relative = path.relative_to(root).as_posix()
    parts = list(path.relative_to(source_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    module_name = ".".join(parts)
    if not all(part.isidentifier() and not keyword.iskeyword(part) for part in parts):
        module_name = f"<file:{relative}>"
    try:
        raw = path.read_bytes()
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        source = raw.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
        tree = ast.parse(source, filename=relative)
        # ast.parse alone permits e.g. return outside a function. Compilation checks
        # scope/syntax but does not execute the resulting code object.
        compile(tree, relative, "exec", dont_inherit=True)
    except (SyntaxError, UnicodeError, LookupError, ValueError, OSError, RecursionError) as exc:
        line = getattr(exc, "lineno", None)
        location = f"{relative}:{line}" if line else relative
        raise IngestionError(f"{location}: cannot parse Python ({type(exc).__name__})") from None
    entity = Entity(
        stable_id("module-v1", repository_id, relative), repository_id, relative, "Module",
        module_name or "<root>", module_name or "<root>", 1,
        max(1, source.count("\n") + int(bool(source) and not source.endswith("\n"))),
        hashlib.sha256(raw).hexdigest(), source, module_name,
    )
    extractor = Extractor(entity)
    try:
        extractor.visit(tree)
    except RecursionError:
        raise IngestionError(f"{relative}: AST exceeds analysis recursion limit") from None
    return extractor


def parse_repository(repository: Path | str, repository_id: str, source_root: str = ".") -> Snapshot:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", repository_id):
        raise IngestionError("Repository ID must be 1-128 letters, digits, dots, underscores or hyphens")
    source_path = PurePosixPath(source_root)
    if source_path.is_absolute() or ".." in source_path.parts or "\\" in source_root or ":" in source_root:
        raise IngestionError("Source root must be a repository-relative POSIX directory")
    root = Path(repository).resolve()
    source = root.joinpath(*source_path.parts)
    if not root.is_dir() or not source.is_dir() or not source.resolve().is_relative_to(root):
        raise IngestionError("Repository and source root must be existing directories within the repository")
    if any(path.is_symlink() for path in [source, *source.parents] if path != root and path.is_relative_to(root)):
        raise IngestionError("Source root must not traverse symlinks")
    files, skipped = discover(root, source)
    if not files:
        raise IngestionError("No Python files found; refusing to replace an existing graph with an empty snapshot")
    extractors = [parse_file(path, root, source, repository_id) for path in files]
    names = [item.module.module_name for item in extractors]
    if len(names) != len(set(names)):
        raise IngestionError("Ambiguous module names in source root")
    references, edges = Resolver(extractors).resolve()
    edges.extend(edge for item in extractors for edge in item.edges)
    snapshot = Snapshot(
        repository_id, source_path.as_posix(),
        tuple(sorted((entity for item in extractors for entity in item.entities.values()), key=lambda entity: entity.id)),
        tuple(sorted(edges, key=lambda edge: edge.id)),
        tuple(sorted(references, key=lambda reference: reference.id)), tuple(skipped),
    )
    snapshot.validate()
    return snapshot
