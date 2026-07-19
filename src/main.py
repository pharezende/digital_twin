from operator import itemgetter
from chromadb import Documents
from prometheus_client import start_http_server
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from .prompts import PROMPTS
from .metrics import VECTOR_STORE_DOCUMENTS
from .ui import create_ui

from .config import (
    CHROMA_COLLECTION,
    CHROMA_DIRECTORY,
    EMBEDDING_MODEL,
    K,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    PDF_PATH,
    RESET_VECTOR_STORE,
)


def print_context(retriever: VectorStoreRetriever, question: str):
    """Print the chunks selected by the retriever"""
    retrieved_documents = retriever.invoke(question)

    print("\nRetrieved documents:")

    for index, document in enumerate(
        retrieved_documents,
        start=1,
    ):
        print(f"\n--- Chunk {index} ---")
        print(document.page_content[:500])
        print("Metadata:", document.metadata)


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller overlapping chunks."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    return text_splitter.split_documents(documents)


def create_or_get_vector_store(
    embedding: OllamaEmbeddings, reset_vector_store: bool = False
) -> Chroma:
    """Create or Get a local Chroma vector store."""

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embedding,
        persist_directory=str(CHROMA_DIRECTORY),
    )

    if reset_vector_store:
        print("Resetting the existing vector-store collection.")
        vector_store.reset_collection()

    if vector_store._collection.count() > 0:
        VECTOR_STORE_DOCUMENTS.set(len(vector_store.get()["ids"]))
        return vector_store

    documents = PyPDFLoader(PDF_PATH).load()
    chunks = split_documents(documents)
    vector_store.add_documents(chunks)

    VECTOR_STORE_DOCUMENTS.set(len(vector_store.get()["ids"]))

    return vector_store


def format_docs(documents: list[Documents]) -> str:
    """Convert retrieved Document objects into clean prompt context."""

    return "\n\n".join(
        (
            f"Source: {document.metadata.get('source', 'unknown')}\n"
            f"{document.page_content}"
        )
        for document in documents
    )


def create_rag_chains(retriever: VectorStoreRetriever) -> dict[str, Runnable]:
    """Create the retrieval-augmented generation chains, one for each language."""

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,  # Low temperature to be less creative
    )

    rag_chains = {
        language: {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
        for language, prompt in PROMPTS.items()
    }

    return rag_chains


def main():
    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    vector_store = create_or_get_vector_store(embedding, RESET_VECTOR_STORE)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": K,
        },
    )

    rag_chain = create_rag_chains(retriever)

    demo = create_ui(rag_chain)

    start_http_server(
        port=8000,
        addr="0.0.0.0",
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )


if __name__ == "__main__":
    main()
