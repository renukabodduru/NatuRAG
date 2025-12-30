import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile 
from
langchain_community.document_loaders 
import PyPDFLoader

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit config
st.set_page_config(page_title="Fast PDF Chatbot", layout="wide")
st.title("⚡ Fast PDF Chatbot")

# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=GROQ_API_KEY
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

@st.cache_resource(show_spinner=False)
def build_vectorstore(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path=tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)

# Sidebar
st.sidebar.title("Upload PDF")
pdf = st.sidebar.file_uploader("PDF file", type="pdf")

if pdf:
    with st.spinner("Processing PDF..."):
        st.session_state.vectorstore = build_vectorstore(pdf.read())
    st.sidebar.success("PDF ready")

# Chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

query = st.chat_input("Ask a question")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        if st.session_state.vectorstore:

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

            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

            chain = (
                {
                    "context": retriever,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
            )

            answer = chain.invoke(query).content

        else:
            answer = llm.invoke(query).content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

