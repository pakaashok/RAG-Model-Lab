import streamlit as st
import os
from core.ingest import DocumentIngester
from core.retriever import RAGRetriever
from utils.helpers import (
    check_ollama_connection,
    check_chromadb_connection,
    get_available_models,
    get_document_count
)
from config.settings import settings

# ================================
# Page Config
# ================================
st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# Sidebar
# ================================
with st.sidebar:
    st.title("⚙️ RAG Lab Control Panel")
    st.divider()

    # Service Status
    st.subheader("🔌 Service Status")
    col1, col2 = st.columns(2)
    with col1:
        ollama_status = check_ollama_connection()
        st.metric(
            "Ollama",
            "✅ Online" if ollama_status else "❌ Offline"
        )
    with col2:
        chroma_status = check_chromadb_connection()
        st.metric(
            "ChromaDB",
            "✅ Online" if chroma_status else "❌ Offline"
        )

    st.divider()

    # Configuration Info
    st.subheader("📋 Configuration")
    st.info(f"**Model:** {settings.LLM_MODEL}")
    st.info(f"**Embedding:** {settings.EMBEDDING_MODEL}")
    st.info(f"**Chunk Size:** {settings.CHUNK_SIZE}")
    st.info(f"**Top K:** {settings.TOP_K_RESULTS}")

    st.divider()

    # Document Upload
    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your documents",
        accept_multiple_files=True,
        type=["pdf", "txt", "csv"],
        help="Supported: PDF, TXT, CSV"
    )

    if uploaded_files:
        save_dir = "./documents/uploaded"
        os.makedirs(save_dir, exist_ok=True)
        for file in uploaded_files:
            with open(f"{save_dir}/{file.name}", "wb") as f:
                f.write(file.getbuffer())
        st.success(f"✅ {len(uploaded_files)} files uploaded!")

    # Document Stats
    doc_counts = get_document_count("./documents")
    if doc_counts["total"] > 0:
        st.subheader("📊 Document Stats")
        st.write(f"📄 PDF: {doc_counts['pdf']}")
        st.write(f"📝 TXT: {doc_counts['txt']}")
        st.write(f"📊 CSV: {doc_counts['csv']}")
        st.write(f"📁 Total: {doc_counts['total']}")

    st.divider()

    # Index Button
    if st.button(
        "🔄 Index Documents",
        type="primary",
        use_container_width=True
    ):
        if not ollama_status:
            st.error("❌ Ollama is not running!")
        elif not chroma_status:
            st.error("❌ ChromaDB is not running!")
        else:
            with st.spinner("Indexing..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(value, message):
                    progress_bar.progress(value)
                    status_text.text(message)

                ingester = DocumentIngester()
                result = ingester.ingest(
                    "./documents",
                    progress_callback=update_progress
                )

                if result["success"]:
                    st.success(f"""
                        ✅ Indexed Successfully!
                        - 📄 Documents: {result['documents']}
                        - 🧩 Chunks: {result['chunks']}
                    """)
                else:
                    st.error(f"❌ {result['error']}")

    # Clear Chat Button
    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

# ================================
# Main Area
# ================================
st.title(f"🤖 {settings.APP_TITLE}")
st.markdown(
    "Ask questions about your documents. "
    "All processing is done **locally** on your machine!"
)
st.divider()

# Service Warning
if not check_ollama_connection() or \
   not check_chromadb_connection():
    st.warning(
        "⚠️ Some services are offline. "
        "Please check the status in the sidebar."
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 View Sources"):
                for i, source in enumerate(
                    message["sources"], 1
                ):
                    st.markdown(
                        f"**Source {i}:** "
                        f"`{source['source']}` | "
                        f"Page: `{source['page']}`"
                    )
                    st.caption(
                        source["content"][:300] + "..."
                    )
                    st.divider()

# Chat Input
if question := st.chat_input(
    "Ask a question about your documents..."
):
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents..."):
            retriever = RAGRetriever()
            result = retriever.query(question)

        if result["success"]:
            st.markdown(result["answer"])

            if result.get("sources"):
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(
                        result["sources"], 1
                    ):
                        st.markdown(
                            f"**Source {i}:** "
                            f"`{source['source']}` | "
                            f"Page: `{source['page']}`"
                        )
                        st.caption(
                            source["content"][:300] + "..."
                        )
                        st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", [])
            })
        else:
            st.error(f"❌ Error: {result['error']}")
            