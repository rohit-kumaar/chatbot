# Plan: Generate Embeddings using Ollama Free Model

## Goal
Integrate Ollama embeddings into the Streamlit application (`main.py`) to generate embeddings for extracted text chunks using a free local Ollama model (e.g., `nomic-embed-text` or `all-minilm`).

---

## Steps to Implement

### 1. Install Dependencies & Prepare Ollama Model
1. **Install `langchain-ollama`** in the virtual environment:
   ```powershell
   .\.venv\Scripts\pip.exe install langchain-ollama
   ```
2. **Pull a free Ollama embedding model** (e.g., `nomic-embed-text`):
   ```powershell
   ollama pull nomic-embed-text
   ```

---

## 2. Update `main.py`
1. **Import `OllamaEmbeddings`**:
   ```python
   from langchain_ollama import OllamaEmbeddings
   ```
2. **Add Embedding Controls in Sidebar**:
   - Embedding Model selector input (default: `nomic-embed-text`, alternative: `all-minilm`, `mxbai-embed-large`).
   - Ollama Base URL input (default: `http://localhost:11434`).
   - "🧠 Generate Embeddings" button.

3. **Generate & Display Embeddings**:
   - When the user clicks "Generate Embeddings", instantiate `OllamaEmbeddings(model=embedding_model, base_url=base_url)`.
   - Batch/loop embed chunks with progress feedback (`st.progress` / `st.spinner`).
   - Display embedding results:
     - Vector dimension (e.g. 768 dimensions for `nomic-embed-text`).
     - Total embedded vectors count.
     - Interactive view showing a preview of the generated embedding vector (e.g. first 10 values of float array) alongside each chunk or in an expander.
   - Error handling: handle cases where Ollama service is offline or the requested model is not downloaded.

---

## Verification
1. Run `ollama list` to verify model availability.
2. Launch Streamlit: `streamlit run main.py`.
3. Upload a sample PDF, chunk it, and click "Generate Embeddings".
4. Confirm embeddings generated successfully and verify vector dimensions & preview values in Streamlit UI.
