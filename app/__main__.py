
import streamlit as st

from agentic_flow.flow import compile_graph, invoke_graph
from utils import streamhandler
from utils.lightrag import spawn_lightrag_instance, feed_documents_to_lightrag


class Main:
    def __init__(self):
        st.session_state.messages = []
        st.session_state.rag = spawn_lightrag_instance()
        st.session_state.graph = compile_graph()

    def chat(self):

        with st.sidebar:
            st.button("Ingest docs", on_click=feed_documents_to_lightrag, args=("/app/docs", st.session_state.rag,))

        # Accept user input
        if prompt := st.chat_input("What is on your mind, dear AI enjoyer?", key="chat_input"):
         
            # Add user message to chat history
            st.session_state.messages.append(
                {"role": "user", "content": prompt})
           
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container in form of a stream
            with st.chat_message("assistant"):

                # Create a stream handler for streaming of assistant response
                stream_handler = streamhandler.StreamHandler(
                    st.empty(), display_method="markdown"
                )

                # Send user query to LLM and receive response
                response = invoke_graph(st.session_state.graph, st.session_state.messages)

                stream_handler.on_static_string(response)

                # Append assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )


if __name__ == "__main__":
    main = Main()
    main.chat()