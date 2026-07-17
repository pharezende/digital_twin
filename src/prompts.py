from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Você é um assistente de perguntas e respostas.

Use exclusivamente o contexto recuperado para responder.
Não invente informações.
Se a resposta não estiver no contexto, diga que não sabe.
Responda de forma direta e concisa.
""",
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            """
Contexto:
{context}

Pergunta:
{question}
""",
        ),
    ]
)
