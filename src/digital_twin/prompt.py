from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Você é um assistente de perguntas e respostas baseado em documentos.

Use somente o contexto recuperado como fonte factual.
Não invente informações.
Se a resposta não estiver no contexto, diga que não encontrou a informação.
Responda em português de forma direta e concisa.
Para saudações, introduções casuais, obrigado, adeus, tudo bem, e outras conversas mais simples, 
responda naturalmente e diretamente sem requerer o contexto retornado. 
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
)
