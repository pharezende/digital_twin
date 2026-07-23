import os
from pathlib import Path
from setuptools._distutils.util import strtobool
from dotenv import load_dotenv


load_dotenv(override=True)


def parse_boolean(value: str) -> bool:
    normalized = value.strip().lower()

    if normalized in {"true", "yes", "on", "1"}:
        return True

    if normalized in {"false", "no", "off", "0"}:
        return False

    raise ValueError(
        f"Invalid Boolean value: {value!r}. " "Use true/false, yes/no, on/off, or 1/0."
    )


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")
CHROMA_DIRECTORY = Path(os.getenv("CHROMA_DIRECTORY"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION")
PDF_PATH = os.getenv("PDF_PATH")
K = int(os.getenv("K"))
RESET_VECTOR_STORE = parse_boolean(os.getenv("RESET_VECTOR_STORE"))
