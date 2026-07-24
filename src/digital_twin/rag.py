from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from chromadb import Documents
from operator import itemgetter
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.output_parsers import StrOutputParser


def format_docs(documents: list[Documents]) -> str:
    """Convert retrieved Document objects into clean prompt context."""

    return "\n\n".join(
        (
            f"Source: {document.metadata.get('source', 'unknown')}\n"
            f"{document.page_content}"
        )
        for document in documents
    )


def create_rag_chain(
    retriever: VectorStoreRetriever,
    prompt: ChatPromptTemplate,
    llm: BaseChatModel,
) -> Runnable:
    """Create a retrieval-augmented generation chain."""

    return (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
