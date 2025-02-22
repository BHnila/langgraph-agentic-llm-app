import httpx

async def post_query_lightrag(query_text: str,query_mode: str):
    url = "http://lightrag:9621/query"
    headers = {"Content-Type": "application/json"}
    payload = {
        "query": query_text,
        "param": {
            "mode": query_mode
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raises an exception for HTTP errors
        return response.json()