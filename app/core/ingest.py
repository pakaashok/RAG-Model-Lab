from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from config.settings import settings
import chromadb
import os

class DocumentIngester:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            base_url=settings.OLLAMA_HOST,
            model=settings.EMBEDDING_MODEL
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len
        )
        self.chroma_client = self._setup_chromadb()

    def _setup_chromadb(self):
        """Setup ChromaDB with v2 API"""
        try:
            client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT
            )
            # Test connection
            client.heartbeat()
            print("✅ ChromaDB connected!")
            return client
        except Exception as e:
            print(f"❌ ChromaDB error: {e}")
            raise e

    def load_documents(self, directory: str) -> list:
        """Load multiple document types"""
        documents = []

        loaders = {
            "**/*.pdf": PyPDFLoader,
            "**/*.txt": TextLoader,
            "**/*.csv": CSVLoader,
        }

        for glob_pattern, loader_cls in loaders.items():
            try:
                loader = DirectoryLoader(
                    directory,
                    glob=glob_pattern,
                    loader_cls=loader_cls,
                    silent_errors=True
                )
                docs = loader.load()
                if docs:
                    documents.extend(docs)
                    print(
                        f"✅ Loaded {len(docs)} "
                        f"docs with {glob_pattern}"
                    )
            except Exception as e:
                print(f"⚠️ Error loading: {e}")

        return documents

    def ingest(
        self,
        directory: str,
        progress_callback=None
    ) -> dict:
        """Main ingestion pipeline"""
        try:
            # Step 1: Load
            if progress_callback:
                progress_callback(
                    0.2,
                    "📂 Loading documents..."
                )
            documents = self.load_documents(directory)

            if not documents:
                return {
                    "success": False,
                    "error": "No documents found!"
                }

            # Step 2: Split
            if progress_callback:
                progress_callback(
                    0.4,
                    "✂️ Splitting into chunks..."
                )
            chunks = self.text_splitter.split_documents(
                documents
            )

            # Step 3: Embed & Store
            if progress_callback:
                progress_callback(
                    0.6,
                    "🔢 Creating embeddings..."
                )

            Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                client=self.chroma_client,
                collection_name=settings.CHROMA_COLLECTION
            )

            if progress_callback:
                progress_callback(
                    1.0,
                    "✅ Indexing complete!"
                )

            return {
                "success": True,
                "documents": len(documents),
                "chunks": len(chunks)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            