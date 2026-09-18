"""Read-only connectivity/version probe; no graph schema or ingestion."""

import asyncio

from neo4j import AsyncGraphDatabase, Query, READ_ACCESS
from neo4j.exceptions import DriverError, Neo4jError

from backend.config import NEO4J_VERSION, Settings


class Neo4jProbeError(RuntimeError):
    """A deliberately credential-free error for API and preflight output."""


async def check_neo4j(settings: Settings) -> str:
    if settings.neo4j_password is None:
        raise Neo4jProbeError("NEO4J_PASSWORD is not configured")
    try:
        async with asyncio.timeout(settings.neo4j_timeout_seconds):
            async with AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
                connection_timeout=settings.neo4j_timeout_seconds,
                connection_acquisition_timeout=settings.neo4j_timeout_seconds,
                max_transaction_retry_time=0,
            ) as driver:
                await driver.verify_connectivity()
                async with driver.session(
                    database=settings.neo4j_database, default_access_mode=READ_ACCESS
                ) as session:
                    result = await session.run(Query(
                        "CALL dbms.components() YIELD versions RETURN versions[0] AS version",
                        timeout=settings.neo4j_timeout_seconds,
                    ))
                    record = await result.single(strict=True)
                    version = record["version"]
    except (DriverError, Neo4jError, OSError, TimeoutError) as exc:
        raise Neo4jProbeError(f"Neo4j connectivity/query failed ({type(exc).__name__})") from None
    if version != NEO4J_VERSION:
        raise Neo4jProbeError(f"Neo4j version mismatch: expected {NEO4J_VERSION}, received {version}")
    return version
