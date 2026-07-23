from digital_twin.config import CHROMA_DIRECTORY
from digital_twin.rag import create_rag_chains
from digital_twin.vector_store import create_or_get_vector_store
from prometheus_client import start_http_server
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from .ui import create_ui
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


from .config import (
    EMBEDDING_MODEL,
    K,
    OLLAMA_BASE_URL,
    RESET_VECTOR_STORE,
    PDF_PATH,
    CHROMA_COLLECTION,
    CHROMA_DIRECTORY,
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


def main():
    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    documents = PyPDFLoader(PDF_PATH).load()

    split_documents = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True,
    ).split_documents(documents)

    vector_store = create_or_get_vector_store(
        embedding,
        split_documents,
        CHROMA_COLLECTION,
        CHROMA_DIRECTORY,
        RESET_VECTOR_STORE,
    )

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
