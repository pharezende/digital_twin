from pprint import pprint
from chromadb import Documents
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from config import (
    CHROMA_COLLECTION,
    CHROMA_DIRECTORY,
    EMBEDDING_MODEL,
    K,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    PDF_PATH,
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


def create_vector_store(chunks: list[Document], embedding: OllamaEmbeddings) -> Chroma:
    """Create or Get a local Chroma vector store."""

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embedding,
        persist_directory=str(CHROMA_DIRECTORY),
    )

    if vector_store._collection.count() == 0:
        if not chunks:
            raise ValueError(
                "The Chroma collection is empty and no chunks were provided."
            )

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

    client = Client()

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(
        """
    Responda à pergunta usando exclusivamente o contexto fornecido.
    Não invente informações.
    Responda diretamente ao que foi perguntado.
    Se a resposta não estiver no contexto, diga que não sabe.

    Pergunta:
    {question}

    Contexto:
    {context}

    Resposta:
    """
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def main():
    system_initial_message = "Como posso te ajudar?"

    documents = PyPDFLoader(PDF_PATH).load()
    chunks = split_documents(documents)
    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    vector_store = create_vector_store(chunks, embedding)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": K,
        },
    )
    rag_chain = create_rag_chain(retriever)

    # question = "Fale me sobre Diana"
    question = "Qual o nome completo de Diana? Ela tem quantos anos?"

    answer = rag_chain.invoke(question)

    print("\Resposta:")
    pprint(answer)


if __name__ == "__main__":
    main()
