import gradio as gr
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
import os

CSS_PATH = os.path.join(
    os.path.dirname(__file__),
    "static",
    "styles.css",
)

PREDEFINED_QUESTIONS = [
    "Qual é o nome completo de Dina?",
    "Qual é a idade de Dina?",
    "Como é a aparência física de Dina?",
]


def load_css() -> str:
    with open(CSS_PATH, encoding="utf-8") as file:
        return file.read()


def convert_history(history: list) -> list:
    """Converte o histórico do Gradio para mensagens LangChain."""

    messages = []

    for item in history:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if not isinstance(content, str):
            continue

        if role == "user":
            messages.append(HumanMessage(content=content))

        elif role == "assistant":
            messages.append(AIMessage(content=content))

    return messages


def create_ui(rag_chain: Runnable) -> gr.Blocks:
    """Cria a interface Gradio do assistente RAG."""

    def respond(message: str, history: list) -> str:
        question = message.strip()

        if not question:
            return "Digite uma pergunta sobre o documento."

        try:
            return rag_chain.invoke(
                {
                    "question": question,
                    "chat_history": convert_history(history),
                }
            )

        except Exception as error:
            print(f"Erro durante a execução do RAG: {error}")

            return (
                "Não foi possível processar a pergunta. "
                "Verifique se o Ollama e o banco vetorial estão disponíveis."
            )

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(
        theme=theme,
        css=load_css(),
        title="Assistente de Documentos",
    ) as demo:

        gr.HTML(
            """
            <header id="app-header">
                <h1>Assistente de Documentos</h1>
                <p>
                    Consulte o conteúdo do PDF usando recuperação semântica
                    e respostas fundamentadas no documento.
                </p>
            </header>
            """
        )

        with gr.Column(elem_id="assistant-card"):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                height=520,
                placeholder=(
                    "<strong>Olá!</strong><br>"
                    "Escolha uma pergunta sugerida ou escreva a sua."
                ),
            )

            textbox = gr.Textbox(
                elem_id="chat-input",
                placeholder="Digite uma pergunta sobre o documento...",
                container=False,
                lines=1,
                max_lines=4,
            )

            gr.HTML('<div id="examples-title">Perguntas sugeridas</div>')

            gr.ChatInterface(
                fn=respond,
                chatbot=chatbot,
                textbox=textbox,
                examples=PREDEFINED_QUESTIONS,
                submit_btn="Enviar",
                stop_btn="Parar",
                autofocus=True,
                autoscroll=True,
                show_progress="minimal",
            )

        gr.HTML(
            """
            <footer id="footer-text">
                As respostas são produzidas com base nos trechos
                recuperados do documento indexado.
            </footer>
            """
        )

    return demo
