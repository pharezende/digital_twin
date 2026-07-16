import os
from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv


def import_pdf_file(path: str) -> list:
    loader = PyPDFLoader(path)
    return loader.load()


def main():
    system_initial_message = "Como posso te ajudar?"

    load_dotenv(override=True)

    file = import_pdf_file(os.getenv("PDF_PATH"))

    print(f"Loaded pages: {len(file)}")

    # Retrieve Document if not retrieved yet

    # Build or retrieve Chroma DB

    # Loop

    # Call Embedded Model

    # Call LLM
    # llm = ChatOpenRouter(model="tencent/hy3:free", temperature=0.1)

    # RAG Pipeline

    # print("Hello from digital-twin!")


if __name__ == "__main__":
    main()
