from asyncio import run

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from utils.lightrag_api import post_query_lightrag

"""

A module that contains Langchain tools for LLM agents.

"""


@tool
def fei_stu_retriever(user_question: str) -> str:
    """
    Retrieves documents related to FEI STU based on the user's question.

    Args:
        user_question (str): The question asked by the user.

    Returns:
        response: The retrieved documents in text format.
    """

    response = run(post_query_lightrag(user_question, "hybrid"))
    return response.text


@tool
def fei_stu_web_search(user_question: str) -> str:
    """
    Performs a web search about FEI STU and related topics.

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
    
