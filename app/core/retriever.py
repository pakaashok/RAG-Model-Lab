from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config.settings import settings
import chromadb
from chromadb.config import Settings as ChromaSettings

class RAGRetriever:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            base_url=settings.OLLAMA_HOST,
            model=settings.EMBEDDING_MODEL
        )
        self.llm = Ollama(
            base_url=settings.OLLAMA_HOST,
            model=settings.LLM_MODEL,
            temperature=0.1
        )
        self.chroma_client = self._setup_chromadb()
        self.chain = self._setup_chain()

    def _setup_chromadb(self):
        """Setup ChromaDB client with tenant"""
        client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            settings=ChromaSettings(
                anonymized_telemetry=False
            )
        )

        # Create tenant if not exists
        try:
            client.get_tenant(
                name="default_tenant"
            )
        except Exception:
            client.create_tenant(
                name="default_tenant"
            )

        # Create database if not exists
        try:
            client.get_database(
                name="default_database",
                tenant="default_tenant"
            )
        except Exception:
            client.create_database(
                name="default_database",
                tenant="default_tenant"
            )

        return client

    def _setup_chain(self):
        """Setup RAG chain"""
        prompt_template = """
        You are a helpful assistant.
        Use the context below to answer the question.
        If answer is not in context, say
        "I cannot find this in the documents."

        Context:
        {context}

        Question: {question}

        Answer:
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=self.embeddings
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.TOP_K_RESULTS}
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def query(self, question: str) -> dict:
        """Query the RAG system"""
        try:
            result = self.chain.invoke(
                {"query": question}
            )
            sources = [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get(
                        "source", "Unknown"
                    ),
                    "page": doc.metadata.get(
                        "page", "N/A"
                    )
                }
                for doc in result.get(
                    "source_documents", []
                )
            ]
            return {
                "success": True,
                "answer": result["result"],
                "sources": sources
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            