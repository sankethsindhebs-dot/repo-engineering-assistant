"""Repeatable Phase 0 probes. No ingestion, tool execution or database writes."""

import argparse
import asyncio
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

# Also supports `python scripts/preflight.py` from a different working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from backend.config import PROJECT_ROOT, Settings
from backend.graph.connection import Neo4jProbeError, check_neo4j
from backend.llm import ProviderError, create_provider
from backend.main import create_app


class GenerationProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    function_name: str
    result: int


async def check_app(settings: Settings) -> None:
    application = create_app(settings)
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://phase0.test"
        ) as client:
            response = await client.get("/health")
    if response.status_code != 200 or response.json() != {"status": "ok", "phase": 0}:
        raise RuntimeError("Unexpected liveness response")


async def check_models(settings: Settings) -> str:
    provider = create_provider(settings)
    measurements = []
    try:
        start = perf_counter()
        vectors = await provider.embed(["A Python function adds two numbers.", "A database stores a graph."])
        dimension = len(vectors[0])
        repeat = await provider.embed(["A Python function adds two numbers."])
        if len(repeat[0]) != dimension:
            raise ProviderError("Embedding dimension changed between requests")
        measurements.append(f"embedding_dimension={dimension}; two requests={perf_counter() - start:.2f}s")
        for left, right in ((2, 3), (7, 11)):
            start = perf_counter()
            answer = await provider.generate(
                "Read this Python source without executing it: def add(a, b): return a + b. "
                f"What does add({left}, {right}) return? Return JSON with function_name and integer result.",
                GenerationProbe,
            )
            if answer.function_name != "add" or answer.result != left + right:
                raise ProviderError("Generation probe returned a schema-valid but incorrect answer")
            measurements.append(f"generation_request={perf_counter() - start:.2f}s")
    finally:
        await provider.aclose()
    return "; ".join(measurements) + "; basic probes only, not workload/quality acceptance"


def check_compose() -> tuple[str, str]:
    with (PROJECT_ROOT / "compose.yaml").open(encoding="utf-8") as handle:
        yaml.safe_load(handle)
    if shutil.which("docker") is None:
        return "BLOCKED", "YAML parsed; Docker CLI unavailable; Compose validation/container NOT verified"
    result = subprocess.run(
        ["docker", "compose", "-f", str(PROJECT_ROOT / "compose.yaml"), "config", "--quiet"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        return "FAIL", "docker compose config --quiet failed; check Docker Compose and local configuration"
    return "PASS", "Docker Compose configuration accepted; this does not prove container startup"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checks", nargs="+", choices=["config", "app", "compose", "neo4j", "models"],
        default=["config", "app", "compose", "neo4j", "models"],
    )
    args = parser.parse_args(argv)
    try:
        settings = Settings()
    except ValidationError as exc:
        fields = sorted({".".join(map(str, error["loc"])) or "settings" for error in exc.errors()})
        print("FAIL config: invalid settings in " + ", ".join(fields) + "; values suppressed")
        return 1

    statuses = []
    for check in dict.fromkeys(args.checks):
        status = "PASS"
        try:
            if check == "config":
                detail = "settings validated; service availability is checked separately"
            elif check == "app":
                asyncio.run(check_app(settings))
                detail = "application startup and /health=200; liveness only"
            elif check == "compose":
                status, detail = check_compose()
            elif check == "neo4j":
                if settings.neo4j_password is None:
                    status, detail = "BLOCKED", "set a local NEO4J_PASSWORD before probing Neo4j"
                else:
                    version = asyncio.run(check_neo4j(settings))
                    detail = f"authenticated query succeeded; server version={version}"
            elif settings.model_provider == "disabled":
                status, detail = "BLOCKED", "provider disabled; model feasibility remains a LOCAL VERIFICATION ITEM"
            else:
                detail = asyncio.run(check_models(settings))
        except (Neo4jProbeError, ProviderError) as exc:
            status, detail = "FAIL", str(exc)
        except Exception as exc:
            status, detail = "FAIL", f"check raised {type(exc).__name__}; details suppressed"
        print(f"{status} {check}: {detail}", flush=True)
        statuses.append(status)

    print("Selected checks only; results do not automatically establish Phase 0 completion.")
    if "FAIL" in statuses:
        return 1
    return 2 if "BLOCKED" in statuses else 0


if __name__ == "__main__":
    raise SystemExit(main())
