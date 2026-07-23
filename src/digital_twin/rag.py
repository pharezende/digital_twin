from chromadb import Documents
from operator import itemgetter
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from .prompts import PROMPTS

from .config import (
    LLM_MODEL,
    OLLAMA_BASE_URL,
)


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
        temperature=0.0,  # Low temperature to be less creative
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
