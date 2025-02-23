from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langfuse.callback import CallbackHandler
from asyncio import run as arun
from agentic_flow.agents import AgentState, retrieve_or_respond, search_or_respond, generate
from agentic_flow.tools import retriever, fei_stu_web_search


langfuse_handler = CallbackHandler(secret_key="sk-lf-eaca059f-5d00-4fd9-a4da-b0ab0a689b02",
                                   public_key="pk-lf-57d0dc66-7822-439c-b35f-7d52c6c66aca",
                                   host="http://langfuse-web:8030")


def create_graph():

    workflow = StateGraph(AgentState)

    retriever_runner = ToolNode([retriever])
    web_search_runner = ToolNode([fei_stu_web_search])

    workflow.add_node("retrieve_or_respond", retrieve_or_respond)
    workflow.add_node("retriever_runner", retriever_runner)
    workflow.add_node("search_or_respond", search_or_respond)
    workflow.add_node("web_search_runner", web_search_runner)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("retrieve_or_respond")
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

    return workflow


def compile_graph(graph_memory):

    workflow = create_graph()
    graph = workflow.compile(checkpointer=graph_memory)

    return graph
    

def invoke_graph(graph, messages):

    graph = graph.with_config({"configurable": {"thread_id": 1},
                       "recursion_limit": 100, "callbacks": [langfuse_handler]})

    # Invoke the graph
    output = graph.invoke(
        input={
            "messages": messages,
        }
    )

    return output["messages"][-1].content
