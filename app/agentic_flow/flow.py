from langgraph.graph.state import StateGraph, CompiledStateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from langfuse.callback import CallbackHandler

from agentic_flow.agents import AgentState, retrieve_or_respond, search_or_respond, respond
from agentic_flow.tools import fei_stu_retriever, fei_stu_web_search

"""

A module that defines the agentic LLM workflow.

"""


# A callback handler for LangFuse LLM monitoring
langfuse_handler = CallbackHandler(secret_key="sk-lf-eaca059f-5d00-4fd9-a4da-b0ab0a689b02",
                                   public_key="pk-lf-57d0dc66-7822-439c-b35f-7d52c6c66aca",
                                   host="http://langfuse-web:8030")


def build_graph() -> StateGraph:
    """
    Builds a state graph for the agent workflow.

    The workflow consists of the following nodes:
    - "retrieve_or_respond": Initial state to decide whether to retrieve information or respond directly.
    - "retriever_runner": Executes the retrieval tool.
    - "search_or_respond": Decides whether to perform a web search or respond directly.
    - "web_search_runner": Executes the web search tool.
    - "respond": Generates the final response.

    The workflow transitions are defined as:
    - Entry point is "retrieve_or_respond".
    - Conditional edges from "retrieve_or_respond" based on `tools_condition`:
        - END: Ends the workflow.
        - "tools": Transitions to "retriever_runner".
    - Edge from "retriever_runner" to "search_or_respond".
    - Conditional edges from "search_or_respond" based on `tools_condition`:
        - END: Ends the workflow.
        - "tools": Transitions to "web_search_runner".
    - Edge from "web_search_runner" to "generate".
    - Edge from "generate" to END.

    Returns:
        StateGraph: The constructed workflow state graph.

    """

    # Spawn the State Graph
    workflow = StateGraph(AgentState)

    # Define the tool execution nodes
    retriever_runner = ToolNode([fei_stu_retriever])
    web_search_runner = ToolNode([fei_stu_web_search])

    # Add the nodes to the workflow
    workflow.add_node("retrieve_or_respond", retrieve_or_respond)
    workflow.add_node("retriever_runner", retriever_runner)
    workflow.add_node("search_or_respond", search_or_respond)
    workflow.add_node("web_search_runner", web_search_runner)
    workflow.add_node("respond", respond)

    # Set entry point
    workflow.set_entry_point("retrieve_or_respond")
    
    # Define the workflow transitions
    workflow.add_conditional_edges(
        "retrieve_or_respond",
        tools_condition,
        {END: END, "tools": "retriever_runner"},
    )
    workflow.add_edge("retriever_runner", "search_or_respond")
    workflow.add_conditional_edges(
        "search_or_respond",
        tools_condition,
        {END: END, "tools": "web_search_runner"},
    )
    workflow.add_edge("web_search_runner", "generate")
    workflow.add_edge("generate", END)

    # Return the constructed workflow - graph
    return workflow


def compile_graph(graph_memory: MemorySaver) -> CompiledStateGraph:
    """
    Compiles a workflow state graph using the provided graph memory as a checkpointer.

    Args:
        - graph_memory: An object used as a checkpointer for the workflow compilation.

    Returns:
        - The compiled graph object.

    """

    # Build the workflow state graph
    workflow = build_graph()

    # Compile the workflow
    graph = workflow.compile(checkpointer=graph_memory)

    # Return the compiled graph
    return graph
    

def invoke_graph(graph : StateGraph, messages: list[BaseMessage]) -> str:
    """
    Invokes a graph with the given messages and returns the content of the last message.

    This function configures the graph with specific settings, including a thread ID,
    recursion limit, and a list of callbacks. It then invokes the graph with the provided
    messages and returns the content of the last message in the output.

    Args:
        - graph: The compiled graph to be invoked.
        - messages: A list of messages to be passed as input to the graph.

    Returns:
        str: The content of the last message in the output.
    """

    graph = graph.with_config({"configurable": {"thread_id": 1},
                       "recursion_limit": 100, "callbacks": [langfuse_handler]})

    # Invoke the graph
    output = graph.invoke(
        input={
            "messages": messages,
        }
    )

    return output["messages"][-1].content
