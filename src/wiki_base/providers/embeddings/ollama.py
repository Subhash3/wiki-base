import math

import httpx

from wiki_base.providers.embeddings.base import EmbeddingModelInfo


class OllamaEmbeddingProvider:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimensions: int,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None
        self._model_info = EmbeddingModelInfo(
            model=model,
            dimensions=dimensions,
            max_tokens=8192,
        )

    @property
    def model_info(self) -> EmbeddingModelInfo:
        return self._model_info

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = await self._client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model_info.model, "input": texts},
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise ValueError("Ollama returned an unexpected number of embeddings")

        validated: list[list[float]] = []
        for embedding in embeddings:
            if not isinstance(embedding, list) or len(embedding) != self._model_info.dimensions:
                raise ValueError(
                    f"Expected {self._model_info.dimensions} embedding dimensions"
                )
            vector = [float(value) for value in embedding]
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("Ollama returned a non-finite embedding value")
            validated.append(vector)
        return validated

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
