import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from .texts import TEXTS
import time

from .metrics import (
    RAG_REQUESTS,
    RAG_PROCESSING_DURATION,
    RAG_RESPONSE_CHARACTERS,
)


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


def create_ui(
    rag_chain: Runnable,
) -> gr.Blocks:
    """Create the Gradio user interface."""

    def respond(
        message: str,
        history: list[dict],
    ) -> str:
        question = message.strip()

        if not question:
            return TEXTS["empty_question"]

        started_at = time.perf_counter()

        try:
            response = rag_chain.invoke(
                {
                    "question": question,
                    "chat_history": convert_history(history),
                }
            )

            RAG_REQUESTS.labels(status="success").inc()
            RAG_RESPONSE_CHARACTERS.observe(len(response))

            return response

        except Exception as error:
            RAG_REQUESTS.labels(status="error").inc()
            print(f"RAG error: {error}")

            return TEXTS["error_message"]

        finally:
            RAG_PROCESSING_DURATION.observe(time.perf_counter() - started_at)

    with gr.Blocks(title="Digital Twin Assistant") as demo:
        gr.Markdown(TEXTS["title"])

        chatbot = gr.Chatbot(
            value=[
                {
                    "role": "assistant",
                    "content": TEXTS["welcome_message"],
                }
            ],
            elem_id="chatbot",
            height=520,
        )

        textbox = gr.Textbox(
            elem_id="chat-input",
            container=False,
            placeholder=TEXTS["placeholder"],
        )

        gr.ChatInterface(
            fn=respond,
            chatbot=chatbot,
            textbox=textbox,
        )

    return demo
