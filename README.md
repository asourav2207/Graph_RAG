# Graph RAG with Ollama

## Setup

1.  **Dependencies**: A virtual environment has been created in `.venv`.
    Dependencies are installed.

2.  **Ollama**:
    Ensure you have [Ollama](https://ollama.com/) installed and running.
    Pull the default model (or change it in the UI sidebar):
    ```bash
    ollama pull mistral
    ```

## Usage

Run the Streamlit app:

```bash
./.venv/bin/streamlit run app.py
```

1.  **Upload**: Upload a text file (`.txt`) containing the content you want to graph.
2.  **Process**: Click "Process & Build Graph".
3.  **Explore**:
    - View the interactive graph visualization.
    - Chat with your data in the query interface.
