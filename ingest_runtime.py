from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")

def ingest_file(file_path: str, persist_dir: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        loader = TextLoader(file_path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_dir
    )
    db.persist()