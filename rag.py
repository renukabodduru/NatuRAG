from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from memory import get_memory

embeddings = OllamaEmbeddings(model="nomic-embed-text")

llm = Ollama(
    model="llama3",
    temperature=0.1
)

def ask_question(query: str, persist_dir: str, doc_id: str):
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    memory = get_memory(doc_id)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )

    return qa({"question": query})