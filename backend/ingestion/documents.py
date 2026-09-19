"""Repository documentation as source data, without inferred program structure."""

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import re

from backend.ingestion.models import stable_id
from backend.ingestion.repository import EXCLUDED_DIRECTORIES, IngestionError


@dataclass(frozen=True)
class Document:
    id: str
    repository_id: str
    path: str
    format: str
    source: str
    source_hash: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class Documents:
    repository_id: str
    items: tuple[Document, ...]
    skipped_paths: tuple[str, ...] = ()

    @property
    def digest(self) -> str:
        return stable_id("documents-v1", [asdict(item) for item in self.items], self.skipped_paths)


def read_documents(repository: Path | str, repository_id: str) -> Documents:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", repository_id):
        raise IngestionError("Invalid repository ID")
    root = Path(repository).resolve()
    if not root.is_dir():
        raise IngestionError("Documentation repository is not a directory")
    documents, skipped = [], []

    def failed_walk(error: OSError) -> None:
        raise IngestionError("Documentation directory could not be read") from None

    for directory, names, filenames in os.walk(root, followlinks=False, onerror=failed_walk):
        current = Path(directory)
        for name in sorted(names):
            path = current / name
            if name in EXCLUDED_DIRECTORIES or path.is_symlink():
                skipped.append(path.relative_to(root).as_posix() + "/")
        names[:] = sorted(name for name in names if name not in EXCLUDED_DIRECTORIES
                          and not (current / name).is_symlink())
        for name in sorted(filenames):
            markdown = name.lower().endswith(".md")
            if not markdown and name.lower() not in {"readme", "readme.txt"}:
                continue
            path = current / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                skipped.append(relative)
                continue
            if not path.is_file():
                raise IngestionError(f"{relative}: not a regular documentation file")
            try:
                raw = path.read_bytes()
                source = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
                if "\x00" in source:
                    raise ValueError("Binary content")
            except (OSError, UnicodeError, ValueError) as exc:
                raise IngestionError(f"{relative}: cannot read UTF-8 documentation ({type(exc).__name__})") from None
            documents.append(Document(
                stable_id("document-v1", repository_id, relative), repository_id, relative,
                "markdown" if markdown else "text", source, hashlib.sha256(raw).hexdigest(), 1,
                max(1, source.count("\n") + int(bool(source) and not source.endswith("\n"))),
            ))
    return Documents(repository_id, tuple(sorted(documents, key=lambda item: item.path)), tuple(sorted(skipped)))
