"""Offline contract tests. Mock HTTP responses do NOT establish model feasibility."""

import asyncio
from contextlib import redirect_stdout
import io
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from pydantic import ValidationError
import yaml

from backend.config import NEO4J_VERSION, PROJECT_ROOT, Settings
from backend.graph.connection import Neo4jProbeError, check_neo4j
from backend.llm import OllamaProvider, ProviderError, create_provider
from backend.main import create_app
from scripts.preflight import GenerationProbe, check_models, main


def candidate_settings(**overrides):
    values = {
        "model_provider": "ollama",
        "model_generation_model": "candidate-code:7b",
        "model_embedding_model": "candidate-embed:latest",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def app_response(path):
    async def request():
        application = create_app(Settings(_env_file=None))
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application), base_url="http://phase0.test"
            ) as client:
                return await client.get(path)
    return asyncio.run(request())


class CleanEnvironment(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)


class ConfigurationTests(CleanEnvironment):
    def test_provider_is_opt_in(self):
        settings = Settings(_env_file=None)
        self.assertEqual(settings.model_provider, "disabled")
        self.assertIsNone(settings.neo4j_password)
        with self.assertRaises(ProviderError):
            create_provider(settings)

    def test_environment_is_typed(self):
        with patch.dict(os.environ, {"NEO4J_TIMEOUT_SECONDS": "2.5"}):
            self.assertEqual(Settings(_env_file=None).neo4j_timeout_seconds, 2.5)

    def test_invalid_timeouts_rejected(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                Settings(_env_file=None, neo4j_timeout_seconds=value)

    def test_provider_requires_both_model_names(self):
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, model_provider="ollama")

    def test_remote_or_credentialed_model_endpoint_rejected(self):
        for url in ("https://example.com", "http://user:private@127.0.0.1", "http://127.0.0.1/api"):
            with self.subTest(url=url), self.assertRaises(ValidationError):
                Settings(_env_file=None, model_base_url=url)

    def test_invalid_neo4j_uri_rejected_without_exposing_input(self):
        with self.assertRaises(ValidationError) as caught:
            Settings(_env_file=None, neo4j_uri="bolt://user:private-value@localhost")
        self.assertNotIn("private-value", str(caught.exception))

    def test_password_empty_placeholder_and_redaction(self):
        self.assertIsNone(Settings(_env_file=None, neo4j_password="").neo4j_password)
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, neo4j_password="replace-your-password")
        settings = Settings(_env_file=None, neo4j_password="unit-test-only")
        self.assertNotIn("unit-test-only", repr(settings))

    def test_cloud_model_is_not_a_local_candidate(self):
        with self.assertRaises(ValidationError):
            candidate_settings(model_generation_model="something:cloud")

    def test_compose_pins_same_version_and_only_neo4j(self):
        document = yaml.safe_load((PROJECT_ROOT / "compose.yaml").read_text())
        self.assertEqual(set(document["services"]), {"neo4j"})
        service = document["services"]["neo4j"]
        self.assertEqual(service["image"], f"neo4j:{NEO4J_VERSION}-community")
        self.assertTrue(all(port.startswith("127.0.0.1:") for port in service["ports"]))
        self.assertIn("$${NEO4J_AUTH#*/}", service["healthcheck"]["test"][1])


class ApplicationTests(CleanEnvironment):
    def test_liveness_without_services(self):
        response = app_response("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "phase": 0})

    def test_unconfigured_database_is_not_ready(self):
        response = app_response("/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "not_ready")

    def test_readiness_success_contract_with_mock_probe(self):
        with patch("backend.main.check_neo4j", AsyncMock(return_value=NEO4J_VERSION)):
            response = app_response("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["neo4j_version"], NEO4J_VERSION)

    def test_readiness_failure_contract_with_mock_probe(self):
        with patch("backend.main.check_neo4j", AsyncMock(side_effect=Neo4jProbeError("unavailable"))):
            response = app_response("/ready")
        self.assertEqual(response.status_code, 503)


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def provider(self, handler, **overrides):
        provider = OllamaProvider(candidate_settings(**overrides), transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(provider.aclose)
        return provider

    @staticmethod
    def chat_body(**overrides):
        body = {
            "model": "candidate-code:7b", "done": True,
            "message": {"role": "assistant", "content": '{"function_name":"add","result":5}'},
        }
        body.update(overrides)
        return body

    async def test_generation_request_and_response_contract(self):
        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(request.url.path, "/api/chat")
            self.assertFalse(payload["stream"])
            self.assertIn("result", payload["format"]["properties"])
            return httpx.Response(200, json=self.chat_body())
        answer = await self.provider(handler).generate("probe", GenerationProbe)
        self.assertEqual(answer.result, 5)

    async def test_generation_invalid_schema_rejected(self):
        for content in ('{"function_name":"add","result":"5"}', '{"wrong":1}', 'not JSON'):
            with self.subTest(content=content):
                body = self.chat_body(message={"role": "assistant", "content": content})
                provider = self.provider(lambda request: httpx.Response(200, json=body))
                with self.assertRaises(ProviderError):
                    await provider.generate("probe", GenerationProbe)

    async def test_incomplete_wrong_role_or_wrong_model_rejected(self):
        cases = [
            self.chat_body(done=False), self.chat_body(model="different-model"),
            self.chat_body(message={"role": "user", "content": "{}"}),
        ]
        for body in cases:
            provider = self.provider(lambda request: httpx.Response(200, json=body))
            with self.assertRaises(ProviderError):
                await provider.generate("probe", GenerationProbe)

    async def test_embedding_batch_contract(self):
        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(request.url.path, "/api/embed")
            self.assertFalse(payload["truncate"])
            self.assertEqual(len(payload["input"]), 2)
            return httpx.Response(200, json={
                "model": "candidate-embed:latest", "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            })
        self.assertEqual(len(await self.provider(handler).embed(["first", "second"])), 2)

    async def test_embedding_invalid_vectors_rejected(self):
        for vectors in ([], [[]], [[0.0, 0.0]], [[float("nan")]], [[float("inf")]], [[True]], [[0.1], [0.2, 0.3]]):
            with self.subTest(vectors=vectors):
                body = {"model": "candidate-embed:latest", "embeddings": vectors}
                provider = self.provider(lambda request: httpx.Response(200, content=json.dumps(body)))
                with self.assertRaises(ProviderError):
                    await provider.embed(["text"])

    async def test_empty_embedding_input_does_not_send_request(self):
        def handler(request):
            self.fail("Empty input must fail before HTTP")
        with self.assertRaises(ProviderError):
            await self.provider(handler).embed([])

    async def test_error_response_body_is_not_exposed(self):
        provider = self.provider(lambda request: httpx.Response(500, text="private-provider-detail"))
        with self.assertRaises(ProviderError) as caught:
            await provider.generate("private-prompt", GenerationProbe)
        self.assertEqual(str(caught.exception), "Model endpoint returned HTTP 500")

    async def test_invalid_envelope_rejected(self):
        for body in ([], {"error": "private-provider-detail"}):
            provider = self.provider(lambda request: httpx.Response(200, json=body))
            with self.assertRaises(ProviderError) as caught:
                await provider.generate("probe", GenerationProbe)
            self.assertNotIn("private-provider-detail", str(caught.exception))

    async def test_total_request_deadline(self):
        async def handler(request):
            await asyncio.sleep(0.1)
            return httpx.Response(200, json=self.chat_body())
        provider = self.provider(handler, model_timeout_seconds=0.01)
        with self.assertRaisesRegex(ProviderError, "TimeoutError"):
            await provider.generate("probe", GenerationProbe)

    async def test_live_probe_rejects_schema_valid_wrong_answer(self):
        provider = AsyncMock()
        provider.embed.return_value = [[0.1, 0.2]]
        provider.generate.return_value = GenerationProbe(function_name="add", result=99)
        with patch("scripts.preflight.create_provider", return_value=provider):
            with self.assertRaisesRegex(ProviderError, "incorrect answer"):
                await check_models(candidate_settings())
        provider.aclose.assert_awaited_once()

    async def test_database_probe_requires_password(self):
        with self.assertRaisesRegex(Neo4jProbeError, "not configured"):
            await check_neo4j(Settings(_env_file=None))


class PreflightTests(CleanEnvironment):
    def setUp(self):
        super().setUp()
        # A user's real .env must never turn these offline tests into service calls.
        settings = patch("scripts.preflight.Settings", side_effect=lambda: Settings(_env_file=None))
        settings.start()
        self.addCleanup(settings.stop)

    def test_selected_offline_checks_pass(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--checks", "config", "app"])
        self.assertEqual(code, 0)
        self.assertIn("liveness only", output.getvalue())

    def test_disabled_services_do_not_produce_success_exit(self):
        with redirect_stdout(io.StringIO()):
            code = main(["--checks", "neo4j", "models"])
        self.assertEqual(code, 2)

    def test_bad_configuration_is_failure_and_sanitized(self):
        output = io.StringIO()
        with patch.dict(os.environ, {"NEO4J_URI": "bolt://user:private-value@localhost"}):
            with redirect_stdout(output):
                code = main(["--checks", "config"])
        self.assertEqual(code, 1)
        self.assertNotIn("private-value", output.getvalue())


if __name__ == "__main__":
    unittest.main()
