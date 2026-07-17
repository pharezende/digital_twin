from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)


PROMPTS = {
    "pt": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Você é um assistente de perguntas e respostas baseado em documentos.

Use somente o contexto recuperado como fonte factual.
Não invente informações.
Se a resposta não estiver no contexto, diga que não encontrou a informação.
Responda em português de forma direta e concisa.
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
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
    ),
    "en": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a document-based question-answering assistant.

Use only the retrieved context as the factual source.
Do not invent information.
If the answer is not in the context, say that the information was not found.
Answer in English directly and concisely.
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                """
Context:
{context}

Question:
{question}
""",
            ),
        ]
    ),
}
