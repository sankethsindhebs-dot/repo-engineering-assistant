"""Phase 0 liveness and Neo4j readiness endpoints only."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.config import Settings
from backend.graph.connection import Neo4jProbeError, check_neo4j


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.settings = settings if settings is not None else Settings()
        yield

    application = FastAPI(title="Repo Engineering Assistant", lifespan=lifespan)

    @application.get("/health")
    async def health() -> dict[str, str | int]:
        # Liveness does not imply that Neo4j or a model is available.
        return {"status": "ok", "phase": 0}

    @application.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        try:
            version = await check_neo4j(request.app.state.settings)
        except Neo4jProbeError as exc:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "neo4j": str(exc)},
            )
        return JSONResponse({"status": "ready", "neo4j_version": version})

    return application


app = create_app()
