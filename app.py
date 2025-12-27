import streamlit as st
import os

from ingest_runtime import ingest_file
from rag import ask_question
from security import sanitize_query, redact_sensitive_data
from cleanup import cleanup_old_files

# ---------- Page Config ----------
st.set_page_config(
    page_title="Enterprise Knowledge Bot",
    layout="wide"
)

UPLOAD_DIR = "uploads"
VECTOR_DB_DIR = "vectordb"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# ---------- Sidebar ----------
st.sidebar.title("📁 Document Center")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF / DOCX / TXT",
    type=["pdf", "docx", "txt"]
)

if uploaded_file:
    doc_id = uploaded_file.name.replace(".", "_")
    doc_upload_dir = os.path.join(UPLOAD_DIR, doc_id)
    doc_vector_dir = os.path.join(VECTOR_DB_DIR, doc_id)

    os.makedirs(doc_upload_dir, exist_ok=True)
    os.makedirs(doc_vector_dir, exist_ok=True)

    file_path = os.path.join(doc_upload_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ingest_file(file_path, doc_vector_dir)
    st.sidebar.success("✅ Document indexed")

docs = os.listdir(VECTOR_DB_DIR) if os.path.exists(VECTOR_DB_DIR) else []
selected_doc = st.sidebar.selectbox("Select Document", docs)

# ---------- Main Chat ----------
st.title("🧠 Enterprise AI Knowledge Chatbot")

if selected_doc:
    user_query = st.chat_input("Ask a question from this document")

    if user_query:
        if not sanitize_query(user_query):
            st.error("🚫 Query blocked for security reasons")
        else:
            response = ask_question(
                user_query,
                os.path.join(VECTOR_DB_DIR, selected_doc),
                selected_doc
            )

            # 🔥 SAFE ANSWER HANDLING (FIXED)
            raw_answer = response.get("answer", "").strip()

            # Apply redaction
            answer = redact_sensitive_data(raw_answer)

            # Final fallback (VERY IMPORTANT)
            if not answer or not answer.strip():
                answer = "I couldn't generate an answer from this document."

            # ---------- Show Answer ----------
            with st.chat_message("assistant"):
                st.write(answer)

            # ---------- Show Sources ----------
            with st.expander("📎 Sources"):
                for src in response.get("source_documents", []):
                    st.write(src.metadata.get("source", "Unknown source"))

else:
    st.info("Upload and select a document to start chatting")

# ---------- Cleanup ----------
cleanup_old_files(UPLOAD_DIR, VECTOR_DB_DIR)
