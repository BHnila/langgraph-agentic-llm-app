import httpx
import json

async def post_query_lightrag(query_text: str, query_mode: str):
    url = "http://lightrag:9621/query"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {
        "query": query_text,
        "param": {
            "mode": query_mode
        }
    }

    timeout = httpx.Timeout(60.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)
        
        return response