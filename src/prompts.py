from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT = ChatPromptTemplate.from_template(
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
