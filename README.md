# Lightweight Local RAG Q&A Chatbot

## Project Overview
This project is a lightweight Retrieval-Augmented Generation (RAG) system that answers questions based on a small set of personal documents. It retrieves relevant document chunks and generates answers using a local language model—no external API costs.

## Steps

### 1. Document Selection
- Choose relevant documents: research notes, project docs, or articles.
- Convert them to clean text (plain text, Markdown, etc.).

### 2. Embedding Creation
- Use a small embedding model, e.g., Sentence Transformers.
- Split your documents into chunks (paragraphs or sections).
- Convert each chunk into an embedding vector.

### 3. Vector Database Setup
- Install and set up FAISS (or another local vector store).
- Store all chunk embeddings in the vector database.

### 4. Local LLM Setup
- Choose a small local model, e.g., a lightweight LLaMA or Mistral variant.
- Use tools like Ollama or LM Studio to run it locally.

### 5. Query Pipeline
- When a question is asked, create an embedding for the query.
- Retrieve the top relevant chunks from the vector database.
- Feed the chunks plus the query into your local LLM.
- Generate an answer based on the retrieved context.

## Tools and Libraries
- Embeddings: Sentence Transformers
- Vector Store: FAISS (or similar)
- Local LLM: LLaMA/Mistral (via Ollama/LM Studio)

## Python Environment
This repository is set up to use a local virtual environment in `.venv`.

### Recommended Python Version
- Preferred: Python 3.11 or 3.12 for the widest package compatibility.
- Current local fallback: Python 3.13 works with the included dependency set by using `chromadb` instead of `faiss-cpu`.

### Create and Activate the Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt
```

### Dependency Notes
- `faiss-cpu` installs automatically on Python versions below 3.13.
- `chromadb` is used automatically on Python 3.13 and newer as the vector store fallback.

## Project Structure
```text
CortexRAG/
├── data/
│   ├── raw/          # Source documents
│   ├── processed/    # Cleaned text extracted from documents
│   └── chunks/       # Chunked text ready for embedding
├── notebooks/        # Experiments and one-off exploration
├── prompts/          # Prompt templates for local generation
├── scripts/          # Helper scripts for ingestion or maintenance
├── src/
│   └── cortex_rag/
│       ├── ingestion/   # Loading and chunking
│       ├── retrieval/   # Embeddings and vector search
│       ├── generation/  # Local model interaction
│       ├── pipeline/    # End-to-end orchestration
│       ├── cli.py
│       └── config.py
├── storage/
│   └── chroma/       # Local vector database files
└── tests/            # Automated tests
```

## Next Steps
1. Gather documents and preprocess them.
2. Install FAISS and create embeddings.
3. Set up a local LLM environment.
4. Build the retrieval-and-generate pipeline.
5. Test with simple questions and iterate!
