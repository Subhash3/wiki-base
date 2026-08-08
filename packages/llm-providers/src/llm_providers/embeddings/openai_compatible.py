import math

import httpx

from llm_providers.embeddings.base import EmbeddingModelInfo


class OpenAICompatibleEmbeddingProvider:
    """Generate embeddings through an OpenAI-compatible HTTP endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimensions: int,
        max_tokens: int,
        timeout_seconds: float,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None
        self._api_key = api_key
        self._model_info = EmbeddingModelInfo(
            model=model,
            dimensions=dimensions,
            max_tokens=max_tokens,
        )

    @property
    def model_info(self) -> EmbeddingModelInfo:
        return self._model_info

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        headers = (
            {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
        )
        response = await self._client.post(
            f"{self._base_url}/v1/embeddings",
            headers=headers,
            json={
                "model": self._model_info.model,
                "input": texts,
                "encoding_format": "float",
            },
        )
        response.raise_for_status()
        data = response.json().get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise ValueError("Embedding API returned an unexpected number of embeddings")

        ordered: list[list[float] | None] = [None] * len(texts)
        for position, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError("Embedding API returned an invalid embedding")
            index = item.get("index", position)
            embedding = item.get("embedding")
            if (
                not isinstance(index, int)
                or not 0 <= index < len(texts)
                or ordered[index] is not None
                or not isinstance(embedding, list)
                or len(embedding) != self._model_info.dimensions
            ):
                raise ValueError("Embedding API returned an invalid embedding")
            vector = [float(value) for value in embedding]
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("Embedding API returned a non-finite embedding value")
            ordered[index] = vector
        if any(embedding is None for embedding in ordered):
            raise ValueError("Embedding API returned incomplete embeddings")
        return [embedding for embedding in ordered if embedding is not None]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
