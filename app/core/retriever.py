from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config.settings import settings
import chromadb

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
        """Setup ChromaDB with tenant"""
        try:
            # Step 1: Connect as admin client
            admin_client = chromadb.AdminClient(
                chromadb.Settings(
                    chroma_api_impl="chromadb.api.fastapi.FastAPI",
                    chroma_server_host=settings.CHROMA_HOST,
                    chroma_server_http_port=settings.CHROMA_PORT
                )
            )

            # Step 2: Create tenant if not exists
            try:
                admin_client.get_tenant(
                    name="default_tenant"
                )
                print("✅ Tenant exists!")
            except Exception:
                admin_client.create_tenant(
                    name="default_tenant"
                )
                print("✅ Tenant created!")

            # Step 3: Create database if not exists
            try:
                admin_client.get_database(
                    name="default_database",
                    tenant="default_tenant"
                )
                print("✅ Database exists!")
            except Exception:
                admin_client.create_database(
                    name="default_database",
                    tenant="default_tenant"
                )
                print("✅ Database created!")

            # Step 4: Connect as regular client
            client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
                tenant="default_tenant",
                database="default_database"
            )
            print("✅ ChromaDB connected!")
            return client

        except Exception as e:
            print(f"❌ ChromaDB error: {e}")
            raise e

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
                    