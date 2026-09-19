import json
import math
import unittest

import httpx
from pydantic import ValidationError

from backend.config import Settings
from backend.graph.evidence_store import effective_readiness, validate_vectors
from backend.llm import EmbeddingIdentity, OllamaProvider, ProviderError
from backend.retrieval.models import EmbeddingProfile
from tests.support_retrieval import unit_vector


class EmbeddingTests(unittest.IsolatedAsyncioTestCase):
    def provider(self, handler, **overrides):
        settings = Settings(_env_file=None, model_provider="ollama", model_embedding_model="nomic-embed-text", **overrides)
        provider = OllamaProvider(settings, transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(provider.aclose)
        return provider

    async def test_identity_normalizes_tag_and_captures_digest(self):
        def handler(request):
            self.assertEqual((request.method, request.url.path), ("GET", "/api/tags"))
            return httpx.Response(200, json={"models": [{"name": "nomic-embed-text:latest", "digest": "sha256:" + "a" * 64}]})
        self.assertEqual(await self.provider(handler).embedding_identity(), EmbeddingIdentity("ollama", "nomic-embed-text:latest", "a" * 64))

    async def test_identity_missing_duplicate_or_bad_digest_is_rejected(self):
        valid = {"name": "nomic-embed-text:latest", "digest": "a" * 64}
        for models in [None, [], [valid, valid], [{**valid, "digest": "bad"}]]:
            with self.subTest(models=models), self.assertRaises(ProviderError):
                await self.provider(lambda request: httpx.Response(200, json={"models": models})).embedding_identity()

    async def test_embedding_only_never_contacts_chat(self):
        def handler(request):
            self.assertEqual(request.url.path, "/api/embed")
            payload = json.loads(request.content)
            self.assertFalse(payload["truncate"])
            self.assertEqual(payload["input"], ["search_document: first", "search_document: second"])
            return httpx.Response(200, json={"model": "nomic-embed-text:latest", "embeddings": [unit_vector(0), unit_vector(1)]})
        provider = self.provider(handler)
        self.assertEqual(await provider.embed(["search_document: first", "search_document: second"]), [unit_vector(0), unit_vector(1)])
        with self.assertRaisesRegex(ProviderError, "Generation model"):
            await provider.generate("unused", EmbeddingProfile)

    def test_retrieval_vectors_validate_every_dimension_and_value(self):
        validate_vectors([unit_vector(1)], 1)
        for bad in [[0.0] * 768, [1.0] * 767, [1.0] * 769,
                    [math.inf] + [1.0] * 767, [math.nan] + [1.0] * 767, [True] + [1.0] * 767]:
            with self.subTest(size=len(bad)), self.assertRaises(ValueError):
                validate_vectors([bad], 1)
        with self.assertRaises(ValueError):
            validate_vectors([unit_vector(1)], 2)

    def test_profile_changes_with_digest_or_chunk_policy(self):
        first = EmbeddingProfile.from_identity(EmbeddingIdentity("ollama", "nomic-embed-text:latest", "a" * 64), 1536)
        second = first.model_copy(update={"digest": "b" * 64})
        self.assertNotEqual(first.id, second.id)
        self.assertNotEqual(first.id, first.model_copy(update={"chunk_policy_id": "other"}).id)
        with self.assertRaises(ValidationError):
            EmbeddingProfile(**(first.model_dump() | {"dimension": 1024}))

    def test_unsupported_model_does_not_silently_use_nomic_prefixes(self):
        with self.assertRaises(ValueError):
            EmbeddingProfile.from_identity(EmbeddingIdentity("ollama", "another-model", "a" * 64), 1536)


class ReadinessTests(unittest.TestCase):
    def ready(self):
        return {"retrieval_schema_version": 1, "retrieval_state": "READY", "source_revision": 2,
                "published_source_revision": 2, "snapshot_hash": "source", "published_snapshot_hash": "source",
                "documentation_hash": "docs", "published_documentation_hash": "docs",
                "documentation_status": "current", "published_profile_id": "profile"}

    def test_legacy_state_is_unindexed(self):
        self.assertEqual(effective_readiness({"snapshot_hash": "source"}).state, "UNINDEXED")

    def test_ready_requires_every_current_publication_identity(self):
        self.assertEqual(effective_readiness(self.ready(), "profile").state, "READY")
        for key in ["source_revision", "snapshot_hash", "documentation_hash", "documentation_status", "published_profile_id"]:
            state = self.ready() | {key: "changed"}
            with self.subTest(key=key):
                self.assertEqual(effective_readiness(state, "profile").state, "DIRTY")

    def test_building_failed_and_dirty_never_satisfy_ready(self):
        for status in ["BUILDING", "FAILED", "DIRTY", "UNINDEXED"]:
            self.assertEqual(effective_readiness(self.ready() | {"retrieval_state": status}, "profile").state, status)


if __name__ == "__main__":
    unittest.main()
