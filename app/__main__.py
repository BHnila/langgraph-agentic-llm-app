import streamlit as st

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from agentic_flow.flow import compile_graph, invoke_graph
from utils.streamhandler import StreamHandler


### A module that sets up an application (Initialize Streamlit session).


class Main:
    """
    Main class for handling chat interactions in the application.
    Methods
    -------
    __init__():
        Initializes the session state with messages, graph memory, and compiled graph.
    chat():
        Manages the chat interface, displaying messages, accepting user input, and generating responses from the AI.
    """

    def __init__(self):
        """
        Initializes the session state for the application.

        This method sets up the initial state for the Streamlit session, including:
        - Initializing an empty list for messages.
        - Creating a MemorySaver instance for graph memory.
        - Compiling the graph using the graph memory.

        Attributes:
            st.session_state.messages (list): A list to store messages.
            st.session_state.graph_memory (MemorySaver): An instance to save graph memory.
            st.session_state.graph: The compiled graph based on the graph memory.
        """
        st.session_state.messages = []
        st.session_state.graph_memory = MemorySaver()
        st.session_state.graph = compile_graph(st.session_state.graph_memory)

    def chat(self):
        """
        Handles the chat functionality of the application.

        This method performs the following tasks:
        1. Displays chat messages of the current conversation on app rerun.
        2. Accepts user input and adds the user message to the chat history.
        3. Displays the user message in the chat message container.
        4. Sends the user query to the LLM (Language Learning Model) and receives a response.
        5. Displays the assistant response in the chat message container in the form of a stream.
        6. Appends the assistant response to the chat history.
        The method uses the Streamlit library for displaying messages and handling user input.
        It also utilizes a stream handler for streaming the assistant's response.

        Args:
            None

        Returns:
            None

        """

        # Display chat messages of current conversation on app rerun
        for message in st.session_state.messages:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)
            else:
                with st.chat_message("user"):
                    st.markdown(message.content)

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

                # Create a stream handler for streaming of assistant response
                stream_handler = StreamHandler(
                    st.empty(), display_method="markdown"
                )

                # Stream waiting message
                stream_handler.on_static_string("Researching...")

                # Send user query to LLM and receive response
                response = invoke_graph(
                    st.session_state.graph, st.session_state.messages)

                # Stream assistant response
                stream_handler.on_static_string(response, erase=True)

                # Append assistant response to chat history
                st.session_state.messages.append(
                    AIMessage(content=response)
                )


# Initialize the main class and start the chat
if __name__ == "__main__":

    # Ensures that the Main class is instantiated only once
    if "main" not in st.session_state:
        st.session_state.main = Main()

    # Start the chat
    st.session_state.main.chat()
