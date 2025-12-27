from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

embeddings = OllamaEmbeddings(model="nomic-embed-text")

llm = Ollama(
    model="llama3.2:3b",
    temperature=0.1
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Use the context below to answer the question.
If the answer is not present, say:
"I don't know based on the document."

Context:
{context}

Question:
{question}

Answer:
""")

def ask_question(query: str, persist_dir: str, doc_id: str):
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10
    }
)

    docs = retriever.invoke(query)

    if not docs:
        return {
            "answer": "No relevant information found in the document.",
            "source_documents": []
        }

    MAX_CHARS = 3000
    context = ""
    for d in docs:
        if len(context) < MAX_CHARS:
            context += d.page_content + "\n\n"

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": query
    })

    if not answer.strip():
        answer = "I couldn't generate an answer from this document."

    return {
        "answer": answer,
        "source_documents": docs
    }
