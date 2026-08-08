import json

import httpx
import pytest

from llm_providers.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)


async def test_embeds_batches_and_restores_response_order() -> None:
    request_body: dict[str, object] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        request_body.update(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.2, 0.3]},
                    {"index": 0, "embedding": [0.0, 0.1]},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider = OpenAICompatibleEmbeddingProvider(
            base_url="http://embeddings.test/",
            model="embedding-model",
            dimensions=2,
            max_tokens=4096,
            timeout_seconds=10,
            api_key="test-key",
            client=client,
        )
        embeddings = await provider.embed_documents(["first", "second"])

    assert embeddings == [[0.0, 0.1], [0.2, 0.3]]
    assert request_body == {
        "model": "embedding-model",
        "input": ["first", "second"],
        "encoding_format": "float",
    }


async def test_rejects_wrong_embedding_dimensions() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.1]}]},
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        provider = OpenAICompatibleEmbeddingProvider(
            base_url="http://embeddings.test",
            model="embedding-model",
            dimensions=2,
            max_tokens=4096,
            timeout_seconds=10,
            client=client,
        )
        with pytest.raises(ValueError, match="invalid embedding"):
            await provider.embed_query("hello")
