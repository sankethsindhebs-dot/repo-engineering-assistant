# Repo Engineering Assistant

Phase 0 foundation only: typed settings, FastAPI liveness, Neo4j readiness,
and optional local model feasibility probes. Repository ingestion, GraphRAG,
agent execution, generated tools, SSE activity and the frontend are not implemented.

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
Compose pins `neo4j:5.26.30-community`. No APOC/GDS, graph schema, vector index,
or Neo4j 2026/Cypher 25 functionality is introduced.

Neo4j 5.26.30 is the pinned database server; neo4j==5.28.6 is the separate Python client driver and is compatible with Neo4j 5.x servers.

On Linux/macOS, use `.venv/bin/python` and `curl` for the equivalent commands.

## Optional local model candidate experiment

By default `MODEL_PROVIDER=disabled`, and no provider is contacted. To test the
candidate, install/start native Ollama and explicitly download:

```powershell
ollama --version
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
ollama list
```

In your private `.env`, change `MODEL_PROVIDER` to `ollama` and uncomment the two
candidate model names already shown in `.env.example`. Keep the loopback base URL.
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

Phase 0 must remain unapproved for completion until required local service/model
checks have actually been run and the provider feasibility decision has been reviewed.
