# Repo Engineering Assistant

Phase 0 service foundation plus Phase 1A deterministic Python AST ingestion and
structural Neo4j persistence. GraphRAG, agent execution, generated tools, SSE activity
and the frontend are not implemented.

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
Compose pins `neo4j:5.26.30-community`. Phase 1A adds only the structural schema
described below. No APOC/GDS, vector index or Neo4j 2026/Cypher 25 functionality is introduced.

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
No hosted fallback exists. The adapter talks to `/api/chat` and `/api/embed`;
it does not require an agent framework or the Ollama SDK.

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
precedence. Timeouts must be positive. Model names must be explicit when enabled.
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

Only `.py` files beneath the source root are selected. Symlinks are not followed.
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
no relationship indexes, vector indexes or plugins. A version check requires
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
persistence. No Phase 1B retrieval or later functionality is included.
