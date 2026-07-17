import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(override=True)


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")
CHROMA_DIRECTORY = Path(os.getenv("CHROMA_DIRECTORY"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION")
PDF_PATH = os.getenv("PDF_PATH")
K = int(os.getenv("K"))
