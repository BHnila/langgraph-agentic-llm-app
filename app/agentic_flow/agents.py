import os
import streamlit as st
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_openai import ChatOpenAI

from agentic_flow.tools import retriever, fei_stu_web_search


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


llm = ChatOpenAI(
    model="openai/gpt-4o-mini",  # Specify the desired model
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.4,  # Adjust the `temperature as needed
)


def retrieve_or_respond(state: AgentState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retriever])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def search_or_respond(state: AgentState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([fei_stu_web_search])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def generate(state: AgentState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
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