import httpx

"""

A module for communication with LightRAG server API.

"""


async def post_query_lightrag(query_text: str, query_mode: str):
    """
    Asynchronously sends a POST request to the Lightrag API with the given query text and mode.

    This endpoint is used to send a retrieval query to the Lightrag server.

    Args:
        - query_text (str): The text of the query to be sent to the Lightrag API.
        - query_mode (str): The mode parameter for the query.
    Returns:
        - httpx.Response: The response object from the Lightrag API.

    """

    # The URL of the Lightrag API endpoint
    url = "http://lightrag:9621/query"

    # The headers and payload for the POST request
    headers = {"Content-Type": "application/json",
               "Accept": "application/json"}

    # The payload for the POST request
    payload = {
        "query": query_text,
        "param": {
            "mode": query_mode
        }
    }

    # Set the timeout for the request (LightRAG response can take a while)
    timeout = httpx.Timeout(60.0, connect=10.0)

    # Send the POST request to the Lightrag API
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)

        # Return the response object
        return response
