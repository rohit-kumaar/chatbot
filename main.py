import io
import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Page Configuration
st.set_page_config(
    page_title="PDF Text Extractor & Chunking",
    page_icon="📄",
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
st.title("📄 PDF Document Portal")
st.header("Extracted & Chunked PDF Text")

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

# Display File Details in Sidebar & Extracted Text/Chunks on Main Area
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

            # Display the chunks on the right side with visible chunk separators
            for idx, chunk in enumerate(chunks, start=1):
                st.caption(f"--- Chunk {idx} (Length: {len(chunk)} characters) ---")
                st.text(chunk)
                st.markdown("---")
        else:
            st.info("No readable text found in the uploaded PDF file.")

    except Exception as e:
        st.error(f"Error reading PDF file: {e}")

else:
    st.info("👈 Please upload a PDF file from the sidebar.")
