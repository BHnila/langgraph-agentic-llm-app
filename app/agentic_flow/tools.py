from utils.lightrag import post_query_lightrag
from asyncio import run
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


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


@tool
def fei_stu_web_search(user_question: str) -> str:
    """
    Perform a web search.

    Args:
        user_question (str): The search query provided by the user.
    Returns:
        str: The search results obtained from the DuckDuckGo API.
    """

    user_question = user_question + " site:stuba.sk OR site:fei.stuba.sk"

    wrapper = DuckDuckGoSearchAPIWrapper(region="sk-sk", time="y", max_results=10)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper)

    results = search.invoke(user_question)

    return results
    
