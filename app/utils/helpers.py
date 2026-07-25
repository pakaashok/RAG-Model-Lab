import os
import requests
import chromadb
from config.settings import settings


def check_ollama_connection():
    """Check if Ollama is running"""
    try:
        response = requests.get(
            settings.OLLAMA_HOST + "/api/tags",
            timeout=5
        )
        return response.status_code == 200
    except Exception:
        return False


def check_chromadb_connection():
    """Check if ChromaDB is running"""
    try:
        client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT
        )
        client.heartbeat()
        return True
    except Exception:
        return False


def get_available_models():
    """Get list of available Ollama models"""
    try:
        response = requests.get(
            settings.OLLAMA_HOST + "/api/tags",
            timeout=5
        )
        if response.status_code == 200:
            models = response.json().get(
                "models", []
            )
            return [m["name"] for m in models]
        return []
    except Exception:
        return []


def format_file_size(size_bytes):
    """Format file size to human readable"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return str(round(size_bytes, 1)) + " " + unit
        size_bytes /= 1024
    return str(round(size_bytes, 1)) + " GB"


def get_document_count(directory):
    """Count documents in directory"""
    counts = {
        "pdf": 0,
        "txt": 0,
        "csv": 0,
        "total": 0
    }
    if not os.path.exists(directory):
        return counts
    for root, dirs, files in os.walk(directory):
        for file in files:
            ext = file.split(".")[-1].lower()
            if ext in counts:
                counts[ext] += 1
                counts["total"] += 1
    return counts

    