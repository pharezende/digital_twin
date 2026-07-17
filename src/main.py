from operator import itemgetter
from pprint import pprint
from tkinter import RADIOBUTTON
from chromadb import Documents
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser

from config import (
    CHROMA_COLLECTION,
    CHROMA_DIRECTORY,
    EMBEDDING_MODEL,
    K,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    PDF_PATH,
)
from prompts import RAG_PROMPT
from ui import create_ui


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
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True,
    )

    return text_splitter.split_documents(documents)


def create_or_get_vector_store(embedding: OllamaEmbeddings) -> Chroma:
    """Create or Get a local Chroma vector store."""

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embedding,
        persist_directory=str(CHROMA_DIRECTORY),
    )

    if vector_store._collection.count() > 0:
        return vector_store

    documents = PyPDFLoader(PDF_PATH).load()
    chunks = split_documents(documents)
    vector_store.add_documents(chunks)

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


def create_rag_chain(retriever: VectorStoreRetriever) -> Runnable[str, str]:
    """Create the retrieval-augmented generation chain."""

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,  # Low temperature to be less creative
    )

    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain


def main():
    system_initial_message = "Como posso te ajudar?"
    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    vector_store = create_or_get_vector_store(embedding)

    chat_history = []

    # question = input("\nVocê: ").strip()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": K,
        },
    )

    rag_chain = create_rag_chain(retriever)

    demo = create_ui(rag_chain)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )

    # answer = rag_chain.invoke(
    #     {
    #         "question": question,
    #         "chat_history": chat_history,
    #     }
    # )

    # chat_history.append(HumanMessage(content=question))
    # chat_history.append(AIMessage(content=answer))

    #     print(f"\nAgent: {answer}")

    # print("\Resposta:")
    # pprint(answer)


if __name__ == "__main__":
    main()
