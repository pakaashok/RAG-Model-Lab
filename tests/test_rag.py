import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../app'))

from utils.helpers import (
    check_ollama_connection,
    check_chromadb_connection,
    format_file_size,
    get_document_count
)

def test_format_file_size():
    assert format_file_size(1024) == "1.0 KB"
    assert format_file_size(1048576) == "1.0 MB"
    assert format_file_size(500) == "500.0 B"

def test_get_document_count_empty():
    counts = get_document_count("./nonexistent")
    assert counts["total"] == 0

def test_ollama_connection():
    result = check_ollama_connection()
    assert isinstance(result, bool)

def test_chromadb_connection():
    result = check_chromadb_connection()
    assert isinstance(result, bool)
    