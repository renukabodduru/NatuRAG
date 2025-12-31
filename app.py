import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFium2Loader

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --------------------------------------------------
# ENV & STREAMLIT CONFIG
# --------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Fast PDF Chatbot", layout="wide")
st.title("⚡ Fast PDF Chatbot – NatuRAG")

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4,
    api_key=GROQ_API_KEY
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --------------------------------------------------
# BUILD VECTORSTORE
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        pdf_path = tmp.name

    loader = PyPDFium2Loader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)

# --------------------------------------------------
# SIDEBAR – PDF UPLOAD
# --------------------------------------------------
st.sidebar.title("📄 Upload PDF")
pdf = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if pdf:
    with st.spinner("Processing PDF..."):
        st.session_state.vectorstore = build_vectorstore(pdf.read())
    st.sidebar.success("PDF uploaded successfully!")

# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --------------------------------------------------
# BOTTOM-STABLE CHAT INPUT
# --------------------------------------------------
query = st.chat_input("Ask a question…")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --------------------------------------------------
# CHAT LOGIC (SMART SWITCH)
# --------------------------------------------------
if query and query.strip():

    # 🟢 BEFORE PDF → NATURAL CHAT
    if st.session_state.vectorstore is None:
        with st.spinner("Thinking..."):
            answer = llm.invoke(query).content

    # 🔵 AFTER PDF → RAG MODE
    else:
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question using ONLY the context below.
            If the answer is not in the context, say "I don't know".

            Context:
            {context}

            Question:
            {question}
            """
        )

        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )

        with st.spinner("Searching document..."):
            answer = chain.invoke(query).content

    # SAVE & DISPLAY MESSAGE
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    st.chat_message("assistant").write(answer)
