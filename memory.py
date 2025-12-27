from langchain.memory import ConversationBufferMemory

_memory_store = {}

def get_memory(doc_id: str):
    if doc_id not in _memory_store:
        _memory_store[doc_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return _memory_store[doc_id]
