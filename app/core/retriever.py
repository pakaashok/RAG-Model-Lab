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
        """Setup ChromaDB client"""
        try:
            client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT
            )
            client.heartbeat()
            print("ChromaDB connected!")
            return client
        except Exception as e:
            print("ChromaDB error: " + str(e))
            raise e

    def _setup_chain(self):
        """Setup RAG chain"""
        prompt_template = (
            "You are a helpful assistant.\n"
            "Use the context below to answer.\n"
            "If not in context say: "
            "I cannot find this in documents.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

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

    def query(self, question):
        """Query the RAG system"""
        try:
            result = self.chain.invoke(
                {"query": question}
            )
            sources = []
            for doc in result.get(
                "source_documents", []
            ):
                sources.append(
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get(
                            "source", "Unknown"
                        ),
                        "page": doc.metadata.get(
                            "page", "N/A"
                        )
                    }
                )
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
            