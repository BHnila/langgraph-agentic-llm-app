from utils.lightrag import post_query_lightrag
from asyncio import run
from langchain_core.tools import tool

@tool
def retriever(user_question: str) -> str:
    """
    Retrieves docs based on the user's question.

    Args:
        user_question (str): The question posed by the user.

    Returns:
        response: The retreived docs.
    """

    response = run(post_query_lightrag(user_question, "hybrid"))
    return response.text
