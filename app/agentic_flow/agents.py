import os
from langchain_core.messages import SystemMessage, BaseMessage
from typing import TypedDict, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

from agentic_flow.tools import retriever, fei_stu_web_search


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


llm = ChatOpenAI(
    model="openai/gpt-4o-mini",  # Specify the desired model
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.4,  # Adjust the `temperature as needed
)


def generate(state: AgentState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        if len(recent_tool_messages) == 2:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant, assisting students and employees "
        "of Faculty of Electrical Engineering and Informatics (FEI STU) "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Keep the answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}


def search_or_respond(state: AgentState):
    """Generate tool call for retrieval or respond."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    docs_content = "\n\n".join(doc.content for doc in tool_messages)

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant, assisting students and employees "
                "of Faculty of Electrical Engineering and Informatics (FEI STU)."
                "Decide whether to search the web or respond based on the provided docs. "
                "Keep the answer concise."
                "\n\n"
                "{docs_content}"
            ),
            MessagesPlaceholder(variable_name="conversation_messages"),
        ]
    )

    search_or_respond_agent = prompt.partial(docs_content=docs_content) | llm.bind_tools([fei_stu_web_search])

    response = search_or_respond_agent.invoke(conversation_messages)
    return {"messages": [response]}


def retrieve_or_respond(state: AgentState):
    """Generate tool call for retrieval or respond."""

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

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

    search_or_respond_agent = prompt.partial() | llm.bind_tools([retriever])

    response = search_or_respond_agent.invoke(conversation_messages)
    return {"messages": [response]}
