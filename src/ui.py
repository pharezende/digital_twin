import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from translations import TRANSLATIONS


def convert_history(history: list) -> list:
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


def select_language(language: str):
    """Select the language, can be portuguese or english."""
    texts = TRANSLATIONS[language]

    initial_history = [
        {
            "role": "assistant",
            "content": texts["welcome_message"],
        }
    ]

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        language,
        f"# {texts['title']}",
        texts["description"],
        gr.update(placeholder=texts["placeholder"]),
        initial_history,
    )


def create_ui(
    rag_chains: dict[str, Runnable],
) -> gr.Blocks:
    """Create UI using gradio"""

    def respond(
        message: str,
        history: list,
        language: str,
    ) -> str:
        texts = TRANSLATIONS[language]
        question = message.strip()

        if not question:
            return texts["empty_question"]

        try:
            return rag_chains[language].invoke(
                {
                    "question": question,
                    "chat_history": convert_history(history),
                }
            )
        except Exception as error:
            print(f"RAG error: {error}")
            return texts["error_message"]

    with gr.Blocks(title="Digital Twin Assistant") as demo:
        selected_language = gr.State("pt")

        # Initial screen
        with gr.Column(visible=True) as language_screen:
            gr.Markdown(
                """
                # Bem-vindo / Welcome

                Escolha o idioma para continuar.  
                Choose a language to continue.
                """
            )

            with gr.Row():
                portuguese_button = gr.Button(
                    "🇧🇷 Português",
                    variant="primary",
                )

                english_button = gr.Button(
                    "🇺🇸 English",
                )

        # The chatbox screen, perhaps this can become a new function later.
        with gr.Column(visible=False) as application_screen:
            title = gr.Markdown()
            description = gr.Markdown()

            chatbot = gr.Chatbot(
                elem_id="chatbot",
                height=520,
            )

            textbox = gr.Textbox(
                elem_id="chat-input",
                container=False,
            )

            gr.ChatInterface(
                fn=respond,
                chatbot=chatbot,
                textbox=textbox,
                additional_inputs=[selected_language],
            )

            portuguese_button.click(
                fn=lambda: select_language("pt"),
                inputs=None,
                outputs=[
                    language_screen,
                    application_screen,
                    selected_language,
                    title,
                    description,
                    textbox,
                    chatbot,
                ],
            )

            english_button.click(
                fn=lambda: select_language("en"),
                inputs=None,
                outputs=[
                    language_screen,
                    application_screen,
                    selected_language,
                    title,
                    description,
                    textbox,
                    chatbot,
                ],
            )

    return demo
