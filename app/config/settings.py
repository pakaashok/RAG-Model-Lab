from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    # ================================
    # LLM Settings
    # ================================
    OLLAMA_HOST: str = os.getenv(
        "OLLAMA_HOST",
        "http://localhost:11434"
    )
    LLM_MODEL: str = os.getenv(
        "LLM_MODEL",
        "mistral"
    )
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "nomic-embed-text"
    )

    # ================================
    # ChromaDB Settings
    # ================================
    CHROMA_HOST: str = os.getenv(
        "CHROMA_HOST",
        "localhost"
    )
    CHROMA_PORT: int = int(os.getenv(
        "CHROMA_PORT",
        "8000"
    ))
    CHROMA_COLLECTION: str = os.getenv(
        "CHROMA_COLLECTION",
        "rag_documents"
    )

    # ================================
    # RAG Settings
    # ================================
    CHUNK_SIZE: int = int(os.getenv(
        "CHUNK_SIZE",
        "500"
    ))
    CHUNK_OVERLAP: int = int(os.getenv(
        "CHUNK_OVERLAP",
        "50"
    ))
    TOP_K_RESULTS: int = int(os.getenv(
        "TOP_K_RESULTS",
        "3"
    ))

    # ================================
    # App Settings
    # ================================
    APP_TITLE: str = os.getenv(
        "APP_TITLE",
        "Local RAG Lab"
    )
    APP_PORT: int = int(os.getenv(
        "APP_PORT",
        "8501"
    ))

settings = Settings()
