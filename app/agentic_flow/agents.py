import os
import operator
from typing import TypedDict, Annotated

from langchain_core.messages import SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

from agentic_flow.tools import fei_stu_retriever, fei_stu_web_search

"""

A module for defining the LLM agents for the agentic workflow.

"""


class AgentState(TypedDict):
    """
    AgentState is a TypedDict that represents the state of an agent in agentic workflow.

    Attributes:
        messages (list[BaseMessage]): A list of BaseMessage objects. Operator add is used to 
        concatenate lists.
    """
    messages: Annotated[list[BaseMessage], operator.add]


### OPENROUTER API CLIENT

# Client for LLMS provided by OpenRouter
# llm = ChatOpenAI(
#     model="openai/gpt-4o-mini",  # Specify the desired model
#     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
#     openai_api_base="https://openrouter.ai/api/v1",
#     temperature=0.4,  # Adjust the `temperature as needed
# )

### OPENAI API CLIENT

# Client for LLMS hosted on OpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Specify the desired model
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.4,  # Adjust the `temperature as needed
)


def retrieve_or_respond(state: AgentState) -> AgentState:
    """
    A LLM agent that determines whether to retrieve additional documents or respond 
    based on the given state.

    Args:
        -state (AgentState): The current state of the agent, containing messages and 
        other relevant information.

    Returns:
        -dict: A dictionary containing the response message.

    """

    # Filter out the relevant conversation messages
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    # Construct the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant, assisting students and employees "
                "of Faculty of Electrical Engineering and Informatics (FEI STU)."
                "Decide whether to retrieve additional docs or respond."
                "Keep the answer concise."
            ),
            MessagesPlaceholder(variable_name="conversation_messages"),
        ]
    )

    # Build the agent
    search_or_respond_agent = prompt.partial() | llm.bind_tools([
        fei_stu_retriever])

    # Generate the response
    response = search_or_respond_agent.invoke(conversation_messages)
    return {"messages": [response]}


def search_or_respond(state: AgentState) -> AgentState:
    """
    A LLM agent that determines whether to search the web or respond based on the 
    provided documents and conversation history.

    Args:
        -state (AgentState): The current state of the agent, including messages and 
        other relevant information.

    Returns:
        -dict: A dictionary containing the response message generated by the agent.

    """

    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Extract the content of the ToolMessages - retrieved documents
    docs_content = "\n\n".join(doc.content for doc in tool_messages)

    # Filter out the relevant conversation messages
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    # Construct the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant, assisting students and employees "
                "of Faculty of Electrical Engineering and Informatics (FEI STU)."
                "Decide whether to search the web or respond based on the provided docs. "
                "If information in provided docs is uncertain, search the web."
                "Keep the answer concise."
                "\n\n"
                "{docs_content}"
            ),
            MessagesPlaceholder(variable_name="conversation_messages"),
        ]
    )

    # Build the agent
    search_or_respond_agent = prompt.partial(
        docs_content=docs_content) | llm.bind_tools([fei_stu_web_search])

    # Generate the response
    response = search_or_respond_agent.invoke(conversation_messages)
    return {"messages": [response]}


def respond(state: AgentState) -> AgentState:
    """
    The last agent in a flow - generates the final response based on the given 
    agent state.

    Output from fei_stu_retriever and fei_stu_web_search tools is used 
    to generate the response.

    Args:
        - state (AgentState): The current state of the agent, containing messages
                            and other relevant information.
    Returns:
        - dict: A dictionary containing the generated response message.

    """

    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        if len(recent_tool_messages) == 2:
            break
    tool_messages = recent_tool_messages[::-1]

    # Extract the content of the ToolMessages - retrieved documents
    docs_content = "\n\n".join(doc.content for doc in tool_messages)

    # Prepare the prompt
    system_message_content = (
        "You are an assistant, assisting students and employees "
        "of Faculty of Electrical Engineering and Informatics (FEI STU) "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Keep the answer concise."
        "\n\n"
        f"{docs_content}"
    )

    # Filter out the relevant conversation messages
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    # Construct the prompt
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Generate the response
    response = llm.invoke(prompt)
    return {"messages": [response]}
