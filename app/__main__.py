import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from agentic_flow.flow import compile_graph, invoke_graph
from utils import streamhandler
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)


class Main:
    def __init__(self):
        st.session_state.messages = []
        st.session_state.graph_memory = MemorySaver()
        st.session_state.graph = compile_graph(st.session_state.graph_memory)

    def chat(self):

        #Display chat messages of current conversation on app rerun
        for message in st.session_state.messages:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("user"):
                    st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("What is on your mind, dear AI enjoyer?", key="chat_input"):
         
            # Add user message to chat history
            st.session_state.messages.append(
                HumanMessage(content=prompt)
            )
           
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container in form of a stream
            with st.chat_message("assistant"):

                #Create a stream handler for streaming of assistant response
                stream_handler = streamhandler.StreamHandler(
                    st.empty(), display_method="markdown"
                )

                stream_handler.on_static_string("Researching...")

                # Send user query to LLM and receive response
                response = invoke_graph(st.session_state.graph, st.session_state.messages)

                stream_handler.on_static_string(response, erase=True)

                # Append assistant response to chat history
                st.session_state.messages.append(
                    AIMessage(content=response)
                )


if __name__ == "__main__":
    main = Main()
    main.chat()