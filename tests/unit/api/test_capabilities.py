from httpx import AsyncClient


async def test_capabilities(client: AsyncClient) -> None:
    response = await client.get(
        "/capabilities",
        headers={"Origin": "http://localhost:8080"},
    )

    assert response.status_code == 200
    assert [item["extension"] for item in response.json()["document_formats"]] == [
        ".pdf",
        ".docx",
        ".pptx",
    ]
    assert response.headers["access-control-allow-origin"] == "http://localhost:8080"
