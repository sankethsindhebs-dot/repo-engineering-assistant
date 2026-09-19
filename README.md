# Repo Engineering Assistant

Phase 0 services, Phase 1A deterministic Python AST ingestion, and Phase 1B internal
GraphRAG retrieval. Retrieval combines source/document seeds with bounded traversal
of the stored structural graph and returns evidence with provenance. Agent execution,
public agent tools, generated tools, answer generation, SSE and the frontend remain
outside this implementation.

## Local prerequisites

- Python 3.11+; the existing developer environment is Python 3.12.10.
- Docker Desktop with Linux containers and the `docker compose` plugin for Neo4j.
- For the optional candidate experiment: native Ollama on Windows.
  [Official Windows setup](https://docs.ollama.com/windows).

Only Neo4j runs in Compose during Phase 0. Python and the candidate provider run
on the host. This avoids making Windows GPU/container configuration a prerequisite
for assessing a local model. It does not select Ollama permanently.

## Setup from the repository root (PowerShell)

Use the existing virtual environment, or create it with `py -3.12 -m venv .venv`.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Copy `.env.example` to `.env` only if you do not already have a local `.env`.
Set your own `NEO4J_PASSWORD` (at least eight characters). There is no supplied
password. Keep `.env` private; it is ignored by Git. Do not paste its contents
or expanded Docker configuration into an issue, recording or commit.

```powershell
.\.venv\Scripts\python.exe scripts\preflight.py --checks config app
docker compose config --quiet
docker compose up -d --wait --wait-timeout 180 neo4j
.\.venv\Scripts\python.exe scripts\preflight.py --checks neo4j
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
```

`/health` reports process liveness only. `/ready` performs an authenticated,
read-only Neo4j query and requires server version **5.26.30**. It returns HTTP
503 if the database is unconfigured, unreachable or the wrong version. Neither
endpoint runs a model. Model feasibility is assessed separately by preflight.

The official [5.26 release archive](https://neo4j.com/release-notes/database/)
listed [5.26.30](https://neo4j.com/release-notes/database/neo4j-5-26-30/),
released 26 August 2026, as the latest 5.26 patch checked on 18 September 2026.
Compose pins `neo4j:5.26.30-community`. Phase 1A adds the structural schema below;
Phase 1B adds evidence and a native vector index. Neither phase uses APOC/GDS or
Neo4j 2026/Cypher 25 functionality.

Neo4j 5.26.30 is the pinned database server; neo4j==5.28.6 is the separate Python client driver and is compatible with Neo4j 5.x servers.

On Linux/macOS, use `.venv/bin/python` and `curl` for the equivalent commands.

## Optional local model candidate experiment

By default `MODEL_PROVIDER=disabled`, and no provider is contacted. To test the
candidate, install/start native Ollama and explicitly download:

```powershell
ollama --version
ollama pull qwen2.5-coder:3b
ollama pull nomic-embed-text
ollama list
```

In your private `.env`, set `MODEL_PROVIDER=ollama`,
`MODEL_GENERATION_MODEL=qwen2.5-coder:3b` and `MODEL_EMBEDDING_MODEL=nomic-embed-text`.
The older `7b` example remains a candidate for other hardware; the development
laptop with 7.6 GB RAM verified `3b` with Ollama 0.34.2. This is a configuration
choice, with no change to the provider boundary. Keep the loopback base URL.
No hosted fallback exists. The adapter uses `/api/chat`, `/api/embed` and
`/api/tags` for embedding-model identity; it needs neither an agent framework nor
the Ollama SDK. Retrieval/indexing can use an embedding-only configuration. The
Phase 0 `models` preflight still exercises both generation and embedding.

```powershell
.\.venv\Scripts\python.exe scripts\preflight.py --checks models
.\.venv\Scripts\python.exe scripts\preflight.py
```

The model probe checks two embedding requests for finite, nonzero vectors and
consistent dimensions. It then checks two structured generation responses against
simple source-reading questions with independently known answers. It reports actual
request durations. Generated text is never executed or registered as a tool.

Record your machine's RAM/GPU, Ollama version, model IDs from `ollama list`, and
probe timings when deciding whether this candidate is usable. These small probes
do not establish quality, memory needs or latency for the future full agent workload.
If no local provider is available, this remains a **LOCAL VERIFICATION ITEM**.

## Repeatable verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q backend scripts tests
.\.venv\Scripts\python.exe -m pip check
git check-ignore .env
```

Tests use mock provider responses and do not prove that a real model works.
Preflight exit codes: `0` = selected checks passed; `1` = a selected check failed;
`2` = a selected check is blocked/unconfigured. Passing a subset is not Phase 0
sign-off. Docker configuration acceptance also does not prove container startup.

Configuration is loaded from the repository-root `.env`; environment variables take
precedence. Timeouts must be positive. At least one model capability must be named
when enabled; each operation checks that its own model is configured.
Credentials are separate from URIs and errors suppress input/server response values.

## Troubleshooting and boundaries

- If Compose is unavailable, install/enable Docker Desktop and its Compose plugin.
  YAML parsing alone cannot verify Docker behaviour.
- Ports 7474/7687 are published on loopback only. If occupied, stop the conflicting
  local service or explicitly coordinate port/config changes.
- The `neo4j_data` volume persists database credentials. Editing the initial password
  in `.env` does not change credentials in an existing database. Do not delete an
  existing volume to work around an authentication error.
- A model timeout is a failed feasibility probe, not permission to switch providers.
  Diagnose native Ollama, model availability and hardware before changing the plan.
- No real `.env`, model weights, database data or credentials belong in the repository.
- The pinned FastAPI version provides first-party SSE. A later phase should use
  `fastapi.sse.EventSourceResponse` and `ServerSentEvent`; Phase 0 adds no SSE routes.

The development laptop has completed Phase 0 local verification, including all
27 tests, Compose startup, authenticated Neo4j 5.26.30 queries, HTTP `/health` and
`/ready`, and consolidated preflight with the model configuration above. A different
machine must run its own service/model checks; Phase 1A does not contact a model.

## Phase 1A: Python structural ingestion

Run from the repository root after configuring Neo4j through the existing typed
settings. Parsing reads source as data and never imports or executes the target
repository. The default source root is `.`; use `--source-root src` for a `src/`
layout. One source root is supported per snapshot. Paths stored in the graph remain
relative to the repository, including `src/` when selected.
Choose the parent of a package directory as its import root; relative imports from
a module with no indexed parent package stay unresolved.

```powershell
.\.venv\Scripts\python.exe scripts\ingest.py tests\fixtures\python_repo --repository-id sample --dry-run
.\.venv\Scripts\python.exe scripts\ingest.py tests\fixtures\python_repo --repository-id sample
.\.venv\Scripts\python.exe scripts\ingest.py tests\fixtures\python_repo --repository-id sample
```

Choose a stable logical `--repository-id` for each repository. Reusing an ID
**replaces that repository's entire structural snapshot**, including removed files
and old relationships. Different IDs are isolated. `--dry-run` parses and reports
counts without loading settings or opening a database. Normal execution writes;
its JSON output reports `mode: persisted` only after commit. The fixture produces
12 entities, 19 relationships and 4 unresolved references. No HTTP ingestion route
is added. The Phase 0 readiness/probe path remains read-only.

Python analysis selects `.py` files beneath the source root. Phase 1B also scans
repository-root documentation independently, as described below. Symlinks are not followed.
Directories named `.git`, `.venv`, `venv`, `__pycache__`, `node_modules`, `build` and
`dist` are excluded. Skipped paths are reported. This does not implement `.gitignore`
rules. Use a source root that contains the code you intend to ingest. Empty source
trees, module-name collisions, unreadable files, invalid encodings and Python that
the running interpreter cannot parse/compile fail before any graph write. Use the
same Python minor version for reproducible analysis; Python 3.12 is the tested target.
No partial snapshot is installed when a file fails. Parsing happens in memory;
large repositories and files changing during ingestion are outside the tested scale.
Ingest a stable checkout. This is a local development foundation, not a hardened
untrusted-code service.

### Actual graph schema

| Node labels | Meaning |
| --- | --- |
| `Entity:File:Module` | One physical Python file and its module share one stable identity. `kind=Module`; no duplicate file/module nodes. |
| `Entity:Class` | Class declaration. |
| `Entity:Function` | Top-level or nested function, including `async def`. |
| `Entity:Function:Method` | Function declared directly in a class; `kind=Method`. |
| `IngestionState` | One internal metadata/transaction-lock record per repository, outside the public source entity model. |

`CONTAINS` links lexical owners to declarations. `IMPORTS` links an importing
scope to an indexed module or explicitly imported declaration. `CALLS` links the
scope containing an explicit AST call to a statically resolved function or class;
a class target denotes construction syntax, not a guessed `__init__` call.
`INHERITS` links a class to a resolved local base class. Each reference-backed edge
has its own stable site ID and source span, so two calls to one target remain two
distinct pieces of evidence. An import or call inside a function belongs to that
function. Defaults, decorators and class bases are evaluated in their enclosing
scope; base references themselves belong to the derived class.

Every entity stores `id`, `repository_id`, POSIX relative `path`, `kind`, `name`,
`qualified_name`, `module_name`, one-based `start_line`/`end_line`, `source_hash`,
and captured `source`. Function/class source includes decorators. File source is
decoded using Python's encoding cookie and normalised to LF; `source_hash` is
SHA-256 of the original file bytes, shared by declarations from that file.
Empty files use the documented span `1..1`. Reference columns are zero-based
UTF-8 byte offsets with an exclusive end, as specified by
[Python's AST metadata](https://docs.python.org/3.12/library/ast.html#ast.AST).

IDs use SHA-256 over canonical JSON components with versioned prefixes. Module
IDs use repository ID plus relative path. Declaration IDs use the lexical owner
ID, kind, name and same-name/kind occurrence ordinal. They survive blank-line edits
and moving a checkout; renames/moves or reordering duplicate declarations can change
them. Reference IDs include owner, relationship kind, line/column and import-name
discriminator; moving a site changes its reference ID. Relationship IDs derive
from endpoints or reference IDs. No ID uses Neo4j internal IDs or absolute paths.
The snapshot hash also covers the skipped-path inventory; changing excluded
directory presence can change that hash without changing source entity IDs.

Every explicit call, imported name and base expression is retained in the owner's
`references_json` property as a deterministic JSON array. Records contain `id`,
`owner_id`, `kind`, `path`, `expression`, line/column spans, nullable `target_id` and
a resolution `reason`. A null target is explicitly unresolved. This internal record
is needed because an unresolved call cannot honestly have a target edge. It avoids
adding public CallSite/ExternalModule nodes. Inspect these records alongside edges;
external or unindexed imports and bases remain visible. No placeholder targets are
invented. `references_json` is evidence storage, not a graph relationship.

The writer creates two uniqueness constraints: `Entity.id` and
`IngestionState.repository_id`; one range index on `Entity.repository_id` supports
scoped replacement. Neo4j also supplies the constraints' backing indexes. There are
no structural relationship indexes or plugins. Phase 1B's evidence/vector schema
is additive. A version check requires
Neo4j 5.26.30. Schema setup is idempotent and separate from data writes. A single
transaction locks the repository metadata record, deletes only that repository's
entities, recreates the validated snapshot and updates its hash. Failure rolls the
data transaction back; schema setup may already have succeeded. Internal Neo4j node
IDs can change on rerun. Concurrent writers for one repository serialize. There is
no incremental ingestion or history store. If the connection is lost during commit,
the outcome may be uncertain; rerunning the same snapshot is safe. The CLI graph
transaction timeout defaults to 60 seconds (`--timeout-seconds`, maximum 600).

### Static analysis limits

Resolution requires unique lexical declarations or explicit local imports under
the chosen import root. It supports relative imports, aliases, simple reexports
and explicit dotted module imports. It rejects ambiguous/rebound names, conditional
bindings, decorated definitions, explicit metaclasses, wildcard-import scopes and
attribute rebinding tied to an identifiable local module/import binding. Mutations
are keyed by module identity and attribute, including explicit namespace-import
prefixes; rebinding `facade.job` does not invalidate an unrelated `work` binding or
the original function reexported as `job`. Unknown runtime receivers such as `obj`
do not invalidate other symbols merely because an attribute name matches. Their
calls remain unresolved under the existing rules. Mutation tracking uses declared
imports and is flow-insensitive; it does not infer runtime object identity or
prove mutation order/reachability. Class namespaces are not treated as method closures.
Local inheritance is resolved with the same evidence; external bases stay unresolved.

This is not a runtime call graph or proof of execution. It does not model dynamic
dispatch (including `self.method()` and `super()`), monkey patching in general,
reflection, runtime imports, decorators that alter behaviour, metaprogramming,
assignment/value aliasing, function arguments supplied at runtime, or import-loader/
`sys.path` effects. Class-qualified methods and inherited method lookup stay
unresolved. Lambda, comprehension, annotation and type-alias call sites are retained
but not resolved. Implicit calls (operator overloads, descriptor access, decorator
application without an AST Call) are not enumerated. Flow/path feasibility and
complete import-cycle execution are not analysed. Builtins and installed dependencies
are unindexed. Conservative invalidation may omit valid static edges. **A missing
resolved CALLS edge is never proof that a runtime call does not exist.**

### Phase 1A verification

The standard test command above runs Phase 0 plus offline ingestion tests. Live
integration tests are deliberately separate and require explicit opt-in. They use
the existing typed Neo4j configuration, create unique temporary repository IDs,
and clean only their own entities/metadata. Schema constraints/indexes remain.

```powershell
$env:PHASE1A_NEO4J_TESTS = "1"
.\.venv\Scripts\python.exe -m unittest discover -s tests/integration -v
Remove-Item Env:PHASE1A_NEO4J_TESTS
```

Live tests cover labels/evidence, a stored local call chain, repeated ingestion,
changed snapshots, repository isolation, malformed input, transaction rollback,
concurrent first ingestion and the CLI. Offline tests are not evidence of database
persistence. Phase 1B's separate integration coverage is described below.

## Phase 1B: internal evidence retrieval

### Indexing and additive upgrade

For an existing Phase 1A repository in Neo4j, run the indexing command directly
against its unchanged checkout and the same logical repository ID. It verifies
the stored Python snapshot, adds retrieval metadata, documents and chunks, and
preserves existing structural nodes, relationships and `references_json`. No volume
reset or ingestion of unrelated repositories is needed. If Python source differs,
run structural ingestion first. The indexer obtains `source_root` from stored state.

Enable `MODEL_PROVIDER=ollama` and `MODEL_EMBEDDING_MODEL=nomic-embed-text` in your
private configuration. Generation may remain unset. Native Ollama stays outside
Compose. With the embedding model already pulled, the complete new-repository
workflow is:

```powershell
.\.venv\Scripts\python.exe scripts\ingest.py tests\fixtures\retrieval_repo --repository-id retrieval-demo
.\.venv\Scripts\python.exe scripts\index_repository.py --repository tests\fixtures\retrieval_repo --repository-id retrieval-demo
.\.venv\Scripts\python.exe scripts\index_repository.py --repository tests\fixtures\retrieval_repo --repository-id retrieval-demo
```

The second unchanged indexing run reports `embedded: 0`; identity checks still
contact the provider. Structural ingestion never contacts a model. Indexing reads
and rechecks the checkout before publication; use a stable checkout throughout.
The CLI emits `READY` only after successful publication or verified compatible reuse.

### Source and documentation evidence

Python chunks belong to existing Function, Method, Class or Module/File entities.
A parent owns only the lines outside its direct child declarations, so declaration
bodies are not repeatedly embedded in every ancestor. Regions are contiguous and
line-aligned; blank boundary lines do not create standalone chunks. Empty owners
can be exact seeds without invented source evidence.

Documentation discovery starts at the repository root, irrespective of Python's
source root. It includes case-insensitive `.md`, extensionless `README` and
`README.txt`, using the same excluded directory names and no symlink traversal.
UTF-8/BOM input is normalized to LF; invalid encoding/NUL fails explicitly.
Markdown uses ATX headings outside backtick/tilde fenced blocks. Fenced text stays
source evidence; Markdown links and code examples create no structural edges.
Plain README files use line-aligned splitting. This is not a full Markdown parser.

`Document` is separate from `Entity`, `File` and `Module`. Its ID hashes a versioned
prefix, repository ID and relative path. `Chunk` stores owner discrimination,
owner/entity/document IDs, path, kind/name where applicable, actual line span, exact
normalized text, text hash, original-file hash, embedding-input hash and policy ID.
Chunk IDs hash repository, owner, policy, owned-region ordinal and split ordinal.
They are deterministic across machines; changed region/split layout can change IDs.
Text changes can keep an ID while changing its input hash. Original-file hashes
can change without requiring re-embedding unchanged normalized input.

`RETRIEVAL_CHUNK_BYTES` defaults to 1536 UTF-8 bytes, including owner/path metadata
and the `search_document: ` prefix. Whole physical lines are split into bounded
units. An overlong individual line/header fails; nothing is silently truncated.
The budget is not a tokenizer estimate or a guarantee of model context acceptance.
Ollama receives `truncate: false`. Query input uses `search_query: ` with the same
input-byte ceiling. Input format, prefixes and chunk policy are versioned.

### Evidence schema and embedding publication

| Addition | Purpose |
| --- | --- |
| `Document` | Repository-scoped documentation source, format, lines and original hash. |
| `Chunk` | Exact source evidence plus the current/cached vector and profile metadata. |
| `Chunk-[:EVIDENCE_FOR]->Entity/Document` | One deterministic owner link; not a structural traversal edge. |
| `IngestionState` properties | Current/published source and documentation identities, revisions, profile, state and attempt token. |

There are unique constraints on `Chunk.id` and `Document.id`, and repository range
indexes for each. Existing structural constraints remain. Schema setup is
idempotent and separate from transactional data publication.

Each repository has one label `REA_Evidence_v1_<repository-hash>` and one native
index `rea_evidence_vec_v1_<repository-hash>` on `embedding`, with 768 dimensions
and cosine similarity. Setup uses `CREATE VECTOR INDEX ... IF NOT EXISTS`, then
checks the actual label, property, dimension, metric and `ONLINE` state. An
incompatible existing index is rejected, never silently replaced. Native queries
use `db.index.vector.queryNodes()`; see the
[Neo4j vector-index manual](https://neo4j.com/docs/cypher-manual/5/indexes/semantic-indexes/vector-indexes/).
Candidate search is approximate. The query requests `min(400, max(20, 4*k))`
candidates, applies current-state guards and stable score/ID ordering, then returns
at most `k`. This improves the observed small-candidate fixture behavior without
claiming exhaustive nearest neighbors or guaranteed recall.

The profile hashes provider, normalized model name, model digest, 768 dimensions,
prefix policy, chunk policy and input-format version. This first retrieval profile
requires local `nomic-embed-text`; another model/profile must be explicitly designed,
not silently substituted. `/api/tags` must provide a valid digest. Batches default
to eight inputs (`EMBEDDING_BATCH_SIZE`), with responses associated by input position.
Every batch must have the right count, 768 finite values per vector and a nonzero norm;
vectors are never padded or shortened. A positional API cannot independently detect
a server that silently swaps correctly shaped outputs. Tests verify our association.

Model requests run outside data transactions. Publication takes the existing
repository write lock, verifies captured revision/hash/attempt identities, and
atomically replaces the complete vector/profile state. Failed publication rolls
back; an obsolete attempt cannot publish or fail a newer attempt. Compatible
unchanged input hashes reuse stored vectors. Changed/deleted/renamed sources
reconcile current chunks and delete stale chunks/links; retained old vectors are
cache data until complete publication succeeds.

Same-schema profile changes reuse the single index and label, overwriting cached
profiles on successful publication. No per-profile index, label or history grows;
failure retains at most the old cache. No application path drops indexes. Index
lifecycle is one per logical repository for vector schema v1, including repositories
whose files are later removed; repository/index deletion is not an automated feature.
Unrelated indexes are untouched.

### Effective semantic readiness

| State | Meaning and permitted retrieval |
| --- | --- |
| `UNINDEXED` | Missing retrieval metadata/publication. Current exact/lexical/graph evidence may be used. |
| `DIRTY` | Source, documentation, policy or requested profile differs from publication. Semantic evidence is withheld. |
| `BUILDING` | An attempt is generating vectors outside database transactions. Semantic evidence is withheld. |
| `READY` | Current/published source revision, structural hash, documentation hash/freshness and profile agree; index is compatible and online. |
| `FAILED` | An indexing attempt failed or the current vector request is unavailable. Semantic evidence is withheld; diagnostics explain request-level failures. |

A stored `READY` string alone is insufficient. Vector lookup checks publication
identities again in Cypher and checks the repository revision before/after reads.
An application-managed concurrent change causes one bounded retry, then an explicit
failure if revisions keep changing. Provider identity is checked before and after
query embedding; no model fallback occurs. Exact-only requests report readiness of
the published profile without contacting the provider.

After source change and structural ingestion, current source evidence is available
while semantic retrieval remains not ready until re-indexing completes. Calling the
low-level structural writer without a documentation snapshot marks documentation
`needs_refresh`; retained documentation cannot appear as current evidence. The CLI
ingestion scans both source and docs. Indexing can refresh documentation alone.
Filesystem edits become visible when these commands run: there is no filesystem
watcher. Queries are grounded in the captured database snapshot, not unscanned disk.
An interrupted attempt may remain `BUILDING`; rerun indexing to supersede it.

### Internal request, traversal and provenance

Use the internal Python entry point; no HTTP retrieval endpoint or public agent tool
is introduced. For example, after configuring Neo4j:

```python
import asyncio
from backend.config import Settings
from backend.retrieval.models import RetrievalRequest, TraversalProfile, TraversalStep
from backend.retrieval.retrieve import retrieve

request = RetrievalRequest(
    repository_id="retrieval-demo", query="worker.transform", channels=("exact",),
    traversal=TraversalProfile(steps=(TraversalStep(relationship="CALLS", direction="incoming"),)),
)
result = asyncio.run(retrieve(Settings(), request))
print(result.model_dump_json(indent=2))
```

Exact matching uses qualified name, normalized relative path and symbol name;
ambiguous short names retain distinct candidates within the seed limit. Lexical
matching counts shared source/path/identifier terms, including snake/camel parts.
It is a repository-scoped in-memory scan. Vector hits map chunks to their owner.
Each owner receives one rank per channel; direct fusion sums `1/(60 + rank)` with
exact precedence and fixed ties. Raw scores/ranks and matching chunk IDs remain
visible. Vector similarity is not answer confidence. Graph discoveries do not
receive a fabricated semantic score.

Every structural relationship supports outgoing and incoming traversal. Requests
choose explicit relationship/direction pairs; no natural-language inference occurs.
Defaults use outgoing CALLS, IMPORTS, INHERITS and CONTAINS, plus terminal incoming
CONTAINS parent context. Incoming CALLS finds callers, IMPORTS finds dependent
owners, and INHERITS finds derived classes. No reverse relationships are stored.
`parent_context="ascend"` explicitly permits continued parent traversal.

BFS defaults: 5 seeds, depth 2, 40 admitted entities including roots, 80 examined
relationship rows including duplicates/cycles, and 12 rows per expanded entity.
Validated hard ceilings are 20/6/200/1000/100 respectively. Relationship priority is
CALLS, INHERITS, IMPORTS, CONTAINS; outgoing precedes incoming, then target/edge IDs.
Each entity is expanded once; all distinct encountered provenance paths are kept,
including parallel call sites. This is bounded discovery, not enumeration of every
possible graph path. Documentation seeds never enter structural expansion.

Only stored structural edges can be traversed. Unresolved `references_json` entries
produce diagnostics, never guessed targets. Up to 100 unresolved records are
returned; expressions longer than 512 characters are explicitly marked truncated.
Graph provenance preserves seed, hop count, ordered entity path, stored relationship
ID/type/endpoints, followed direction, and reference expression/path/coordinates.
Source spans refer to exact returned text; AST columns retain UTF-8 byte semantics.

Defaults reserve up to 10 direct and 10 graph evidence chunks, two chunks per owner,
and 32 KiB of source text. Evidence is deduplicated by chunk ID without discarding
encountered direct/graph origins. Graph evidence orders by seed rank, hop distance
and stable ID. Source budgets skip complete chunks with diagnostics. Bounds indicate
that a cap was reached, not proof that more neighbors exist. A 15-second request
deadline and database query timeouts limit work; timeout/connection errors fail
explicitly. Whole-repository reads and in-memory processing target small local
repositories, not hard real-time or production-scale service guarantees.

### Phase 1B verification

The offline command in the earlier section includes all source, document, model
schema, traversal and existing regression tests. Real database tests are separate:

```powershell
$env:PHASE1A_NEO4J_TESTS = "1"
$env:PHASE1B_NEO4J_TESTS = "1"
.\.venv\Scripts\python.exe -m unittest discover -s tests/integration -v
Remove-Item Env:PHASE1A_NEO4J_TESTS
Remove-Item Env:PHASE1B_NEO4J_TESTS
```

These tests use deterministic test embeddings in a real Neo4j vector index.
They prove source/vector persistence, retrieval and traversal causality, not a
real model's relevance quality. The fixture's entry-point evidence omits the
downstream `casefold` implementation with graph disabled, discovers it through CALLS
with graph enabled, and loses it when that stored edge is removed. Equivalent tests
cover lexical seeds, reverse directions, inheritance/dependency edge intervention,
unresolved callbacks, bounds, source/doc updates, rollback and additive upgrade.
Tests create unique repository IDs and remove only their own data/vector indexes;
shared schema remains. They never call a generation model.

For the separately gated real local embedding check, keep native Ollama running
with `nomic-embed-text` installed and run:

```powershell
$env:PHASE1B_NEO4J_TESTS = "1"
$env:PHASE1B_OLLAMA_TESTS = "1"
.\.venv\Scripts\python.exe -m unittest discover -s tests/integration -p test_retrieval.py -k test_real_ollama -v
Remove-Item Env:PHASE1B_NEO4J_TESTS
Remove-Item Env:PHASE1B_OLLAMA_TESTS
```

No agent loop, public `search_repository`/`traverse_graph`/`read_evidence` tools,
runtime code generation, generated-code execution, SCC tooling, SSE, frontend,
Mermaid rendering or final answer generation is part of Phase 1B. Static-analysis
limits above continue to apply to every retrieved path.
