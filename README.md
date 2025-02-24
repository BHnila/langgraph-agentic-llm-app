# LangGraph Agentic LLM App

## Overview
LangGraph Agentic LLM App is a robust application designed to harness the power of large language models (LLMs) for diverse tasks. Built on the LangGraph agentic LLM framework, it integrates advanced technologies such as LightRAG for graph retrieval, Streamlit for the frontend, and MongoDB Atlas for both standard and vector storage. Additionally, it includes the LangFuse open-source LLM monitoring service.

This app has been created as demo for my presentation at the Second AI Meetup at FEI STU, 27. 2. 2025. The current conversational flow is tailored to provide information about FEI STU. You can customize the agent's system prompts to suit your specific requirements.

## Table of Contents

- [Important](#important)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Building a Custom .env File](#building-a-custom-env-file)
- [License](#license)

## Important

This app uses OpenRouter API for chat flow and OpenAI API for LightRAG.

**Why OpenRouter?**

OpenRouter provides the flexibility to switch seamlessly between different LLMs. Additionally, using OpenSource model APIs through OpenRouter can be more cost-effective compared to the OpenAI API.

If you want to use just OpenAI API, replace `llm` variable in `app/agentic_flow/agents.py` with:

```
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.4
)
```

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You obtained your personal OpenAI API key. (Optionally: OpenRouter API key)
- You have installed Docker with Docker Compose support on yout PC.

## Installation

1. **Clone the repository:**

    ```bash
    git clone git@github.com:BHnila/langgraph-agentic-llm-app.git
    cd langgraph-agentic-llm-app
    ```

2. **Create a custom `.env` file with your API keys:**

    Follow the instructions in the [Building a Custom .env File](#building-a-custom-env-file) section to create and configure your `.env` file.

3. **Run the application using Docker Compose:**

    ```bash
    docker compose up --build -d
    ```


## Usage


1. **Ingest your files through LightRAG server web interface:**

    **⚠️ ATTENTION: LightRAG ingest may be costly! ⚠️**

    Open your web browser and navigate to `http://localhost:9621/webui`.

1. **Access the chat application:**

    Open your web browser and navigate to `http://localhost`.

2. **Access the Langfuse monitoring dashboard to review conversations:**

    Open your web browser and navigate to `http://localhost:8030`.



## Building a Custom .env File

The `.env` file is used to configure environment variables for the application. Follow these steps to create a custom `.env` file:

1. **Create a new file named `.env` in the root directory of the project.**

2. **Add the following environment variables to the `.env` file:**

    ```plaintext
    # Example .env file
    OPENAI_API_KEY={your personal key for official OpenAI API}
    OPENROUTER_API_KEY={your personal key for OpenRouter API}
    ```

    Replace the placeholder values with your actual configuration details.

3. **Save the `.env` file.**

## License

This project is licensed under the MIT License.