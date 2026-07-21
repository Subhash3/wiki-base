import httpx

from wiki_base.providers.embeddings.ollama import OllamaEmbeddingProvider


async def test_embeds_a_batch_and_validates_dimensions() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        return httpx.Response(
            200,
            json={"embeddings": [[0.0] * 1024, [1.0] * 1024]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider = OllamaEmbeddingProvider(
            base_url="http://ollama.test",
            model="bge-m3:latest",
            dimensions=1024,
            timeout_seconds=10,
            client=client,
        )

        embeddings = await provider.embed_documents(["first", "second"])

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1024
