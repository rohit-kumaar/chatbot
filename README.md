# 📄 PDF Document Chatbot (RAG Pipeline)

A Retrieval-Augmented Generation (RAG) application built with **Streamlit**, **LangChain**, **Ollama**, and **FAISS**. This application allows users to upload PDF documents, automatically chunk and embed their text, store embeddings in a local vector database, and ask questions to receive accurate answers from a local LLM based strictly on document context.

---

## 🏗️ System Architecture

The project implements an end-to-end local **RAG (Retrieval-Augmented Generation)** architecture. All data processing, vector storage, and model inference run locally on your machine.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   DOCUMENT PROCESSING                                   │
│                                                                                         │
│   ┌───────────────┐     pdfplumber     ┌────────────────┐     TextSplitter    ┌───────┐ │
│   │ Uploaded PDF  │ ─────────────────> │ Raw Text Stream│ ──────────────────> │ Chunks│ │
│   └───────────────┘                    └────────────────┘                     └───┬───┘ │
└───────────────────────────────────────────────────────────────────────────────────┼─────┘
                                                                                    │
┌───────────────────────────────────────────────────────────────────────────────────┼─────┐
│                                 VECTOR INDEXING                                   │
│                                                                                   ▼     │
│   ┌─────────────────┐    OllamaEmbeddings    ┌───────────────────┐    Insert    ┌─────┐ │
│   │ nomic-embed-text│ <───────────────────── │  Chunk Vectors    │ ───────────> │FAISS│ │
│   └─────────────────┘                        └───────────────────┘              │  DB │ │
└─────────────────────────────────────────────────────────────────────────────────┴──┬──┘ │
                                                                                     │    │
┌────────────────────────────────────────────────────────────────────────────────────┼────┘
│                                RETRIEVAL & GENERATION                              │
│                                                                                    │
│   ┌───────────────┐     Embed Question      ┌───────────────────┐    Similarity    │
│   │ User Question │ ──────────────────────> │ Question Vector   │ ──── Search ─────┘
│   └───────┬───────┘                         └───────────────────┘         │
│           │                                                               ▼
│           │                                                    ┌────────────────────┐
│           │                                                    │  Top K Context     │
│           │                                                    │  Retrieved Chunks  │
│           │                                                    └─────────┬──────────┘
│           │                                                              │
│           └─────────────────────────┬────────────────────────────────────┘
│                                     ▼
│                          ┌────────────────────┐
│                          │ ChatPromptTemplate │
│                          └──────────┬─────────┘
│                                     ▼
│                          ┌────────────────────┐
│                          │   ChatOllama LLM   │ (llama3.2)
│                          └──────────┬─────────┘
│                                     ▼
│                          ┌────────────────────┐
│                          │    LLM Response    │
│                          └────────────────────┘
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Architectural Workflow Step-by-Step

### 1. Document Upload & Extraction
* **File Processing:** Uploaded via Streamlit file uploader (`pdf`).
* **Text Extraction:** Uses `pdfplumber` to extract text page-by-page from the uploaded PDF binary buffer (`io.BytesIO`).

### 2. Text Chunking
* **Splitter:** Uses LangChain's `RecursiveCharacterTextSplitter`.
* **Configurable Parameters:**
  * `Chunk Size`: Maximum characters per chunk (default: `1000`).
  * `Chunk Overlap`: Character overlap between consecutive chunks (default: `200`).
  * `Separators`: Custom separators parsed from UI input (default: `\n\n`, `\n`, `' '`, `''`).

### 3. Vector Embeddings & Indexing
* **Embedding Model:** Connected to local Ollama via `OllamaEmbeddings` (default model: `nomic-embed-text`).
* **Vector Store:** Uses **FAISS (Facebook AI Similarity Search)** to index text chunks and store vector representations directly in Streamlit session state (`st.session_state`).
* **Caching Strategy:** Generates a unique state signature based on file metadata and chunking configurations. Embeddings are recomputed only when settings or files change.

### 4. Query & Vector Search (Retrieval)
* **Similarity Search:** When a user asks a question, FAISS performs similarity search with `similarity_search_with_score` to retrieve the top $K$ context chunks (configurable $K=1$ to $10$).
* **Context Display:** Retrieved context chunks along with dissimilarity scores are rendered in an expandable UI component.

### 5. Prompt Engineering & Response Generation
* **Prompt Engineering:** Formats context and user question using LangChain's `ChatPromptTemplate` with system instructions restricting answers strictly to the provided document context.
* **LLM Generation:** Sends the prompt to `ChatOllama` (default model: `llama3.2`) to generate concise and context-grounded responses parsed with `StrOutputParser`.

---

## 🛠️ Tech Stack

* **Frontend / UI:** [Streamlit](https://streamlit.io/)
* **Document Parsing:** `pdfplumber`
* **RAG Orchestration:** [LangChain](https://www.langchain.com/) (`langchain-core`, `langchain-community`, `langchain-ollama`, `langchain-text-splitters`)
* **Vector Database:** [FAISS (faiss-cpu)](https://github.com/facebookresearch/faiss)
* **Local LLM & Embeddings:** [Ollama](https://ollama.com/) (`nomic-embed-text`, `llama3.2`)

---

## 🚀 Getting Started

### Prerequisites

1. **Python 3.10+** installed.
2. **Ollama** installed and running locally:
   * Download and install from [Ollama's Website](https://ollama.com/).
   * Pull required models in your terminal:
     ```bash
     ollama pull nomic-embed-text
     ollama pull llama3.2
     ```
   * Ensure Ollama is running at `http://localhost:11434`.

### Installation

1. Clone the repository or navigate to your project directory:
   ```bash
   cd c:\02_my_learning\08_chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirments.txt
   ```

### Running the App

Launch the Streamlit dashboard:
```bash
streamlit run main.py
```

Open your browser at `http://localhost:8501`.

---

## ⚙️ Configuration Options (Sidebar)

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| **Chunk Size** | `1000` | Max characters in a single chunk |
| **Chunk Overlap** | `200` | Overlap characters between chunks |
| **Separators** | `\n\n, \n, ' ', ''` | Priority separators for splitting text |
| **Embedding Model** | `nomic-embed-text` | Ollama model for vector embeddings |
| **LLM Model** | `llama3.2` | Ollama model for generating answers |
| **Ollama Base URL** | `http://localhost:11434` | Endpoint for local Ollama server |
| **Max Output Tokens**| `512` | Maximum length of LLM output response |
| **Top K Chunks** | `3` | Number of context chunks retrieved for RAG |

---

## 📁 Repository Structure

```
.
├── main.py              # Streamlit application & RAG pipeline
├── requirments.txt      # Project dependencies
└── README.md            # Project documentation & architecture guide
```
