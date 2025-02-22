from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langfuse.callback import CallbackHandler

from agentic_flow.agents import query_or_respond, generate
from agentic_flow.tools import retriever


langfuse_handler = CallbackHandler(secret_key="",
                                    public_key="",
                                    host="http://langfuse-web:8030")


def create_graph():

    workflow = StateGraph(MessagesState)

    tools = ToolNode([retriever])

    workflow.add_node(query_or_respond)
    workflow.add_node(tools)
    workflow.add_node(generate)

    workflow.set_entry_point("query_or_respond")
    workflow.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    workflow.add_edge("tools", "generate")
    workflow.add_edge("generate", END)

    return workflow


def compile_graph():
    # Add memory
    memory = MemorySaver()

    workflow = create_graph()
    graph = workflow.compile(checkpointer=memory)

    return graph


def invoke_graph(graph, messages):
    

    # Invoke the graph
    output = graph.invoke(
            input={
                "messages": messages,
            },
            config={"configurable": {"thread_id": 1},
                    "recursion_limit": 100, "callbacks": ([langfuse_handler])},
        )
    
    return output["messages"][-1].content
