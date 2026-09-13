import io
import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Page Configuration
st.set_page_config(
    page_title="PDF Q&A Chatbot with Ollama & FAISS",
    page_icon="🤖",
    layout="wide"
)

def format_file_size(size_in_bytes):
    if size_in_bytes < 1024:
        return f"{size_in_bytes} Bytes"
    elif size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.2f} KB"
    else:
        return f"{size_in_bytes / (1024 * 1024):.2f} MB"

def parse_separators(sep_string):
    """Parses comma-separated separators string into a python list with escaped character support."""
    if not sep_string.strip():
        return ["\n\n", "\n", " ", ""]

    parts = [s.strip().strip("'").strip('"') for s in sep_string.split(",")]
    parsed = []
    for p in parts:
        parsed.append(p.replace("\\n", "\n").replace("\\t", "\t"))
    return parsed

# Main Title & Header
st.title("📄 PDF Document Chatbot")
st.header("RAG Pipeline: Chunking ➔ FAISS Vector Search ➔ Ollama LLM")

# Sidebar Controls
st.sidebar.title("📌 Navigation")
st.sidebar.info("Supported file format: **PDF only**")

# File Uploader in Sidebar
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# Chunking Controls in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Chunk Settings")
chunk_size = st.sidebar.number_input("Chunk Size", min_value=100, max_value=5000, value=1000, step=100)
chunk_overlap = st.sidebar.number_input("Chunk Overlap", min_value=0, max_value=1000, value=200, step=50)
separators_input = st.sidebar.text_input("Separators (comma-separated)", value=r"\n\n, \n, ' ', ''")

# Ollama Models Settings in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Ollama Settings")
embedding_model = st.sidebar.text_input("Embedding Model", value="nomic-embed-text", help="Ollama embedding model")
llm_model = st.sidebar.text_input("LLM Model", value="llama3.2", help="Ollama LLM model for answering questions (e.g. llama3.2, mistral, llama3)")
ollama_url = st.sidebar.text_input("Ollama Base URL", value="http://localhost:11434")
max_tokens = st.sidebar.number_input("Max Output Tokens", min_value=64, max_value=8192, value=512, step=64, help="Maximum number of tokens to generate in response")

# Display File Details in Sidebar & Main Workflow
if uploaded_file is not None:
    formatted_size = format_file_size(uploaded_file.size)

    st.sidebar.markdown("---")
    st.sidebar.success("✅ File uploaded!")
    st.sidebar.subheader("📋 File Details")
    st.sidebar.markdown(f"**📄 Name:** `{uploaded_file.name}`")
    st.sidebar.markdown(f"**🏷️ Type:** `{uploaded_file.type}`")
    st.sidebar.markdown(f"**📏 Size:** `{formatted_size}`")

    # Extract and concatenate all text from PDF
    try:
        pdf_bytes = io.BytesIO(uploaded_file.read())
        all_text = ""
        with pdfplumber.open(pdf_bytes) as pdf:
            st.sidebar.markdown(f"**📑 Pages:** `{len(pdf.pages)}`")
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n\n"

        if all_text.strip():
            # Parse separators from sidebar input
            separators_list = parse_separators(separators_input)

            # Chunk all_text with sidebar parameters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators_list,
                length_function=len
            )
            chunks = text_splitter.split_text(all_text)

            st.sidebar.markdown(f"**🧩 Total Chunks:** `{len(chunks)}`")

            # Create a signature to track file and settings state
            current_signature = (
                uploaded_file.name,
                uploaded_file.size,
                chunk_size,
                chunk_overlap,
                separators_input,
                embedding_model,
                ollama_url
            )

            # Automatically generate FAISS vector store if not created or if settings changed
            if st.session_state.get("file_signature") != current_signature:
                with st.spinner(f"Generating embeddings & indexing into FAISS DB via `{embedding_model}`..."):
                    try:
                        embeddings_service = OllamaEmbeddings(
                            model=embedding_model,
                            base_url=ollama_url
                        )
                        # Create FAISS vector store directly from chunks
                        vector_store = FAISS.from_texts(
                            texts=chunks,
                            embedding=embeddings_service
                        )
                        st.session_state["vector_store"] = vector_store
                        st.session_state["file_signature"] = current_signature
                        st.sidebar.success(f"⚡ Saved {len(chunks)} vectors into FAISS Vector DB!")
                    except Exception as e:
                        st.sidebar.error(f"Failed to create FAISS vector store: {e}")
            else:
                st.sidebar.success(f"⚡ Saved {len(chunks)} vectors into FAISS Vector DB!")

            # Main Area: Q&A Chatbot (RAG)
            st.subheader("🤖 Ask Questions About Your PDF")

            if "vector_store" in st.session_state and st.session_state.get("file_signature") == current_signature:
                vector_store = st.session_state["vector_store"]

                # User input question
                user_question = st.text_input("Ask a question based on the document:", placeholder="e.g. What is the main conclusion of this report?")
                top_k = st.slider("Top K Retrieved Chunks (Context)", min_value=1, max_value=min(10, len(chunks)), value=3)

                if user_question:
                    # Step 1: Query converted to embedding & similarity search in FAISS
                    with st.spinner("1️⃣ Searching FAISS Vector DB for relevant chunks..."):
                        retrieved_docs_with_scores = vector_store.similarity_search_with_score(user_question, k=top_k)
                        retrieved_chunks = [doc.page_content for doc, _ in retrieved_docs_with_scores]
                        context_text = "\n\n---\n\n".join(retrieved_chunks)

                    # Show retrieved context in expandable section
                    with st.expander(f"📚 View {len(retrieved_chunks)} Retrieved Context Chunks from FAISS"):
                        for i, (doc, score) in enumerate(retrieved_docs_with_scores, 1):
                            st.markdown(f"**Chunk {i} (Dissimilarity Score: {score:.4f}):**")
                            st.text(doc.page_content)
                            st.markdown("---")

                    # Step 2: Pass context + question to LLM
                    with st.spinner(f"2️⃣ Generating answer with Ollama LLM (`{llm_model}`)..."):
                        try:
                            prompt_template = ChatPromptTemplate.from_messages([
                                ("system", "You are an AI assistant helping users analyze uploaded documents. "
                                           "Answer the user's question accurately using ONLY the provided context below. "
                                           "If the context doesn't contain the answer, say 'I cannot find the answer in the provided document.'\n\n"
                                           "Context:\n{context}"),
                                ("human", "{question}")
                            ])

                            llm = ChatOllama(
                                model=llm_model,
                                base_url=ollama_url,
                                temperature=0.2,
                                num_predict=max_tokens
                            )

                            chain = prompt_template | llm | StrOutputParser()
                            response = chain.invoke({
                                "context": context_text,
                                "question": user_question
                            })

                            # Display LLM Answer
                            st.markdown("### 💡 LLM Response")
                            st.success(response)

                        except Exception as e:
                            st.error(f"Error generating answer with Ollama LLM: {e}")
            else:
                st.warning("⚠️ Vector store initialization incomplete. Please check Ollama settings and try re-uploading.")

        else:
            st.info("No readable text found in the uploaded PDF file.")

    except Exception as e:
        st.error(f"Error reading PDF file: {e}")

else:
    st.info("👈 Please upload a PDF file from the sidebar.")
