import os
import requests
import chromadb
from config.settings import settings

def check_ollama_connection() -> bool:
    """Check if Ollama is running"""
    try:
        response = requests.get(
            f"{settings.OLLAMA_HOST}/api/tags",
            timeout=5
        )
        return response.status_code == 200
    except:
        return False

def check_chromadb_connection() -> bool:
    """Check if ChromaDB is running"""
    try:
        client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            tenant="default_tenant",
            database="default_database"
        )
        client.heartbeat()
        return True
    except Exception:
        # Try without tenant
        try:
            response = requests.get(
                f"http://{settings.CHROMA_HOST}:"
                f"{settings.CHROMA_PORT}/api/v2/heartbeat",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

def get_available_models() -> list:
    """Get list of available Ollama models"""
    try:
        response = requests.get(
            f"{settings.OLLAMA_HOST}/api/tags",
            timeout=5
        )
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except:
        return []

def format_file_size(size_bytes: int) -> str:
    """Format file size to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} GB"

def get_document_count(directory: str) -> dict:
    """Count documents in directory"""
    counts = {
        "pdf": 0,
        "txt": 0,
        "csv": 0,
        "total": 0
    }
    if not os.path.exists(directory):
        return counts
    for root, _, files in os.walk(directory):
        for file in files:
            ext = file.split(".")[-1].lower()
            if ext in counts:
                counts[ext] += 1
                counts["total"] += 1
    return counts
    