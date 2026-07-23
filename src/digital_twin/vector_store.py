from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from .metrics import VECTOR_STORE_DOCUMENTS
from langchain_core.documents import Document


def create_or_get_vector_store(
    embedding: OllamaEmbeddings,
    chunks: list[Document],
    chroma_collection: str,
    chroma_directory: str,
    reset_vector_store: bool = False,
) -> Chroma:
    """Create or Get a local Chroma vector store."""

    vector_store = Chroma(
        collection_name=chroma_collection,
        embedding_function=embedding,
        persist_directory=chroma_directory,
    )
    if reset_vector_store:
        print("Resetting the existing vector-store collection.")
        vector_store.reset_collection()

    if vector_store._collection.count() > 0:
        print(f"ids: {vector_store.get()["ids"]}")
        return vector_store

    vector_store.add_documents(chunks)

    VECTOR_STORE_DOCUMENTS.set(len(vector_store.get()["ids"]))

    return vector_store
