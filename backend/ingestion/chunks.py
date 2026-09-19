"""Contiguous, source-owned evidence units. Splitting never executes source."""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import re
from typing import Literal

from backend.ingestion.documents import Documents
from backend.ingestion.models import Snapshot, stable_id


DEFAULT_CHUNK_BYTES = 1536
INPUT_FORMAT = "evidence-input-v1"
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def policy_id(max_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    if not 256 <= max_bytes <= 16384:
        raise ValueError("Chunk input budget must be between 256 and 16384 bytes")
    return f"owned-lines-atx-v1-bytes-{max_bytes}"


def source_lines(source: str) -> list[str]:
    # Only LF separates physical lines; Unicode paragraph separators are source.
    return re.findall(r"[^\n]*\n|[^\n]+$", source)


def lexical_terms(text: str) -> list[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    expanded = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", expanded)
    whole = re.findall(r"\w+", text.casefold())
    parts = re.findall(r"[^\W_]+", expanded.casefold())
    return sorted(set(whole + parts))


@dataclass(frozen=True)
class Chunk:
    id: str
    repository_id: str
    owner_kind: Literal["entity", "document"]
    owner_id: str
    entity_id: str | None
    document_id: str | None
    path: str
    entity_kind: str | None
    qualified_name: str | None
    heading: str | None
    start_line: int
    end_line: int
    text: str
    text_hash: str
    file_source_hash: str
    embedding_input_hash: str
    chunk_policy_id: str
    embedding_input: str
    terms: list[str]


def _units(repository_id: str, owner_id: str, owner_kind: str, path: str,
           kind: str | None, qualified_name: str | None, file_hash: str,
           lines: list[str], regions: list[tuple[int, int, str | None]], max_bytes: int) -> list[Chunk]:
    policy = policy_id(max_bytes)
    chunks = []
    for region_number, (start, end, heading) in enumerate(regions):
        while start <= end and not lines[start - 1].strip():
            start += 1
        while end >= start and not lines[end - 1].strip():
            end -= 1
        header = f"{DOCUMENT_PREFIX}{path}\n{kind or 'Document'}: {qualified_name or heading or path}\n"
        available = max_bytes - len(header.encode("utf-8"))
        cursor, part = start, 0
        while cursor <= end:
            stop, size = cursor, 0
            while stop <= end:
                next_size = len(lines[stop - 1].encode("utf-8"))
                if size + next_size > available:
                    break
                size += next_size
                stop += 1
            if stop == cursor:
                raise ValueError(f"{path}:{cursor}: source line or metadata exceeds chunk input budget")
            text = "".join(lines[cursor - 1:stop - 1])
            if text.strip():
                payload = header + text
                chunks.append(Chunk(
                    stable_id("chunk-v1", repository_id, owner_id, policy, region_number, part),
                    repository_id, owner_kind, owner_id,
                    owner_id if owner_kind == "entity" else None,
                    owner_id if owner_kind == "document" else None,
                    path, kind, qualified_name, heading, cursor, stop - 1, text,
                    hashlib.sha256(text.encode("utf-8")).hexdigest(), file_hash,
                    hashlib.sha256(payload.encode("utf-8")).hexdigest(), policy, payload,
                    lexical_terms(f"{path}\n{qualified_name or ''}\n{heading or ''}\n{text}"),
                ))
            cursor, part = stop, part + 1
    return chunks


def python_chunks(snapshot: Snapshot, max_bytes: int = DEFAULT_CHUNK_BYTES) -> list[Chunk]:
    children = defaultdict(list)
    entities = {entity.id: entity for entity in snapshot.entities}
    files = {entity.path: source_lines(entity.source) for entity in snapshot.entities if entity.kind == "Module"}
    for edge in snapshot.relationships:
        if edge.kind == "CONTAINS":
            children[edge.source_id].append(entities[edge.target_id])
    chunks = []
    for entity in snapshot.entities:
        cursor = entity.start_line
        regions = []
        for child in sorted(children[entity.id], key=lambda item: (item.start_line, item.id)):
            if child.start_line > cursor:
                regions.append((cursor, child.start_line - 1, None))
            cursor = max(cursor, child.end_line + 1)
        end = min(entity.end_line, len(files[entity.path]))
        if cursor <= end:
            regions.append((cursor, end, None))
        chunks.extend(_units(snapshot.repository_id, entity.id, "entity", entity.path, entity.kind,
                             entity.qualified_name, entity.source_hash, files[entity.path], regions, max_bytes))
    return sorted(chunks, key=lambda item: (item.path, item.start_line, item.id))


def markdown_regions(lines: list[str]) -> list[tuple[int, int, str | None]]:
    start, heading, fence = 1, None, None
    regions = []
    for number, line in enumerate(lines, 1):
        stripped = line.rstrip("\n")
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", stripped)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
            continue
        if marker:
            fence = marker[1]
            continue
        title = re.match(r"^ {0,3}#{1,6}(?:[ \t]+(.*?)|[ \t]*)$", stripped)
        if title:
            if number > start:
                regions.append((start, number - 1, heading))
            start, heading = number, (title[1] or "").strip()
    if lines:
        regions.append((start, len(lines), heading))
    return regions


def document_chunks(documents: Documents, max_bytes: int = DEFAULT_CHUNK_BYTES) -> list[Chunk]:
    chunks = []
    for document in documents.items:
        lines = source_lines(document.source)
        regions = markdown_regions(lines) if document.format == "markdown" else [(1, len(lines), None)]
        chunks.extend(_units(document.repository_id, document.id, "document", document.path, None, None,
                             document.source_hash, lines, regions, max_bytes))
    return sorted(chunks, key=lambda item: (item.path, item.start_line, item.id))
