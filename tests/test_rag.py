from digital_twin.prompt import PROMPT
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.runnables import RunnableLambda

from digital_twin.rag import (
    create_rag_chain,
    format_docs,
)


def test_format_docs_includes_source() -> None:
    documents = [
        Document(
            page_content="First document",
            metadata={"source": "first.pdf"},
        ),
        Document(
            page_content="Second document",
            metadata={"source": "second.pdf"},
        ),
    ]

    result = format_docs(documents)

    assert result == (
        "Source: first.pdf\n"
        "First document"
        "\n\n"
        "Source: second.pdf\n"
        "Second document"
    )


def test_format_docs_with_empty_list() -> None:
    assert format_docs([]) == ""


def test_create_rag_chain_returns_answer_and_context() -> None:
    documents = [
        Document(page_content="Digital twins modelam sistemas físicos."),
        Document(page_content="Eles podem usar dados operacionais em tempo real."),
    ]

    retriever = RunnableLambda(lambda _: documents)

    llm = FakeListLLM(
        responses=["Digital twins modelam sistemas físicos usando dados operacionais."]
    )

    rag_chain = create_rag_chain(
        retriever=retriever,
        prompt=PROMPT,
        llm=llm,
    )

    chain_input = {
        "question": "O que digital twins fazem?",
        "chat_history": [],
    }

    answer = rag_chain.invoke(chain_input)

    assert answer == (
        "Digital twins modelam sistemas físicos usando dados operacionais."
    )


def test_rag_chain_handles_no_retrieved_documents() -> None:
    retriever = RunnableLambda(lambda _: [])

    llm = FakeListLLM(responses=["Nenhuma informação relevante foi encontrada."])

    rag_chain = create_rag_chain(
        retriever=retriever,
        prompt=PROMPT,
        llm=llm,
    )

    chain_input = {
        "question": "Pergunta desconhecida.",
        "chat_history": [],
    }

    assert rag_chain.invoke(chain_input) == (
        "Nenhuma informação relevante foi encontrada."
    )


def test_rag_chain_includes_chat_history() -> None:
    documents = [
        Document(
            page_content="O sistema usa Chroma.",
            metadata={"source": "arquitetura.pdf"},
        )
    ]

    retriever = RunnableLambda(lambda _: documents)

    captured_prompts = []

    def fake_llm_with_prompt_capture(prompt_value):
        captured_prompts.append(prompt_value)
        return "Ele usa o Chroma."

    llm = RunnableLambda(fake_llm_with_prompt_capture)

    rag_chain = create_rag_chain(
        retriever=retriever,
        prompt=PROMPT,
        llm=llm,
    )

    result = rag_chain.invoke(
        {
            "question": "Qual mesmo?",
            "chat_history": [
                HumanMessage(content="Qual BD de vetor o sistema usa?"),
                AIMessage(content="Ele usa Chroma."),
            ],
        }
    )

    assert result == "Ele usa o Chroma."

    prompt_text = captured_prompts[0].to_string()

    assert "Source: arquitetura.pdf" in prompt_text
    assert "O sistema usa Chroma." in prompt_text
    assert "Qual BD de vetor o sistema usa?" in prompt_text
    assert "Ele usa Chroma." in prompt_text
    assert "Qual mesmo?" in prompt_text
