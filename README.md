# 🤖 Local RAG Model Lab

A complete, production-ready **Retrieval Augmented Generation (RAG)** system that runs 100% locally on your machine. Perfect for learning, demos, and building private AI applications with your own documents.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-green.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📖 Table of Contents
- [About](#-about)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Usage Guide](#-usage-guide)
- [Sample Documents](#-sample-documents)
- [Sample Questions](#-sample-questions)
- [Commands Reference](#-commands-reference)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 About

**Local RAG Lab** is a self-hosted Retrieval Augmented Generation system that lets you ask questions about your own documents using local Large Language Models. All processing happens on your machine — no data leaves your system.

### Why This Project?
- 🔒 **Privacy First** - Your documents never leave your machine
- 💰 **Zero Cost** - No API fees, completely free
- 🚀 **Learn RAG** - Understand how modern AI Q&A systems work
- 🎓 **Demo Ready** - Perfect for showcasing RAG concepts
- 🛠️ **Production Base** - Solid foundation for real projects

---

## ✨ Features

- 📄 **Multi-format Support** - PDF, TXT, CSV documents
- 💬 **Chat Interface** - Streamlit-based conversational UI
- 🔍 **Semantic Search** - Find answers by meaning, not just keywords
- 📚 **Source Attribution** - See which documents provided the answer
- 🐳 **Docker Deployed** - One command setup with Docker Compose
- 🔄 **Easy Reindexing** - Add new documents anytime
- 📊 **Live Status** - Real-time service monitoring
- ⚡ **Optimized Performance** - Configured for CPU inference
- 🎨 **Clean UI** - Modern chat interface with sources display

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              User (Web Browser)                  │
└──────────────────┬──────────────────────────────┘
                   │
                   ↓ HTTP :8501
┌─────────────────────────────────────────────────┐
│         Streamlit UI (rag-app)                   │
│         - Chat Interface                         │
│         - Document Upload                        │
│         - Source Display                         │
└──────────────────┬──────────────────────────────┘
                   │
                   │ LangChain Framework
                   │
         ┌─────────┴─────────┐
         ↓                   ↓
┌──────────────┐    ┌──────────────┐
│   Ollama     │    │  ChromaDB    │
│   :11434     │    │   :8000      │
│              │    │              │
│  - LLM       │    │  - Vectors   │
│  - Embed     │    │  - Metadata  │
└──────────────┘    └──────────────┘
```

### Data Flow

```
INGESTION FLOW:
Documents → Load → Split → Embed → Store in ChromaDB

QUERY FLOW:
Question → Embed → Search ChromaDB → Get Context → LLM Generate → Return Answer with Sources
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **UI Framework** | Streamlit | Web interface |
| **RAG Framework** | LangChain | Orchestration |
| **LLM Runtime** | Ollama | Local model serving |
| **LLM Model** | Llama 3.2 3B | Answer generation |
| **Embedding Model** | Nomic Embed Text | Text to vectors |
| **Vector Database** | ChromaDB | Vector storage |
| **Container Runtime** | Docker Compose | Orchestration |
| **Language** | Python 3.11 | Core language |

---

## 📋 Prerequisites

### Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **RAM** | 8 GB | 16 GB+ |
| **Disk Space** | 15 GB | 25 GB+ |
| **CPU** | 4 cores | 6+ cores |
| **GPU** | Not required | Nvidia GPU for speed |

### Software Requirements
- Docker (v20.10+)
- Docker Compose (v2.0+)
- Git
- Linux, macOS, or Windows with WSL2

### Optional
- Ubuntu 22.04 LTS (recommended OS)
- VSCode with Remote SSH extension

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/pakaashok/RAG-Model-Lab)
cd RAG-Model-Lab
```

### 2. Setup Environment
```bash
# Copy environment template
cp .env.example .env

# Review and update if needed
cat .env
```

### 3. Start Everything (One Command!)
```bash
make start
```
This will:
- ✅ Build Docker images
- ✅ Start Ollama, ChromaDB, and Streamlit
- ✅ Download AI models automatically
- ✅ Ready to use in ~10 minutes

### 4. Open in Browser
```bash
# Get your machine IP
hostname -I

# Open browser to
http://<your-ip>:8501
```

### 5. Use the App
1. Upload documents via the sidebar.
2. Click **"Index Documents"**.
3. Ask questions in the chat!

---

## 📁 Project Structure

```
RAG-Model-Lab/
├── app/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # App configuration
│   ├── core/
│   │   ├── __init__.py
│   │   ├── ingest.py            # Document ingestion logic
│   │   └── retriever.py         # RAG retrieval logic
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py           # Helper functions
│   ├── Dockerfile               # App container definition
│   ├── main.py                  # Streamlit UI entry point
│   └── requirements.txt         # Python dependencies
├── documents/
│   ├── samples/                 # Sample documents
│   │   ├── sample_ai.txt
│   │   └── devops/              # DevOps documents
│   └── uploaded/                # User uploaded files
├── vectorstore/                 # ChromaDB persistent data
├── tests/
│   └── test_rag.py             # Unit tests
├── .github/
│   └── workflows/
│       └── test.yml            # CI/CD pipeline
├── .env.example                 # Environment template
├── .gitignore
├── docker-compose.yml           # Container orchestration
├── Makefile                     # Command shortcuts
└── README.md                    # This file
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```env
# LLM Settings
OLLAMA_HOST=http://ollama:11434
LLM_MODEL=llama3.2:3b            # Change model here
EMBEDDING_MODEL=nomic-embed-text

# ChromaDB Settings
CHROMA_HOST=chromadb
CHROMA_PORT=8000
CHROMA_COLLECTION=rag_documents

# RAG Tuning
CHUNK_SIZE=300                    # Smaller = faster
CHUNK_OVERLAP=30                  # Context between chunks
TOP_K_RESULTS=2                   # Sources per query

# App Settings
APP_TITLE=Local RAG Lab
APP_PORT=8501
```

### Available LLM Models

| Model | Size | Speed (CPU) | Quality |
|-------|------|-------------|---------|
| `llama3.2:1b` | 1.3GB | Very Fast ⚡⚡⚡ | Good |
| `llama3.2:3b` | 2.0GB | Fast ⚡⚡ | Better ✅ |
| `mistral` | 4.4GB | Medium ⚡ | Good |
| `llama3.1:8b` | 4.7GB | Slow | Best |

Change model in `.env`:
```env
LLM_MODEL=llama3.2:3b
```

---

## 📖 Usage Guide

### Adding Documents

#### Method 1: Web Upload
1. Open the UI.
2. Use the sidebar file uploader.
3. Click **"Index Documents"**.

#### Method 2: Direct Copy
```bash
cp your-document.pdf documents/samples/
# Then click "Index Documents" in UI
```

### Asking Questions
1. Type your question in the chat input.
2. Wait for the response (10-30 seconds).
3. View source documents in the expandable section.

### Best Practices

#### ✅ Good Questions:
- "What is Kubernetes?"
- "Explain the deployment process"
- "Summarize the security policies"

#### ❌ Avoid:
- Questions not in documents
- Real-time information queries
- Math calculations

---

## 📚 Sample Documents

The project includes sample documents to get started:

### AI/ML Documents
- `sample_ai.txt` - AI, ML, DL concepts

### DevOps Documents
- `harness.txt` - Harness CI/CD platform
- `terraform.txt` - Terraform infrastructure as code
- `aws.txt` - AWS services for DevOps
- `github.txt` - GitHub and Git workflows
- `linux.txt` - Linux commands and administration
- `backstage.txt` - Backstage developer portal

---

## 💬 Sample Questions

### AI/ML Questions
- What is Artificial Intelligence?
- What is the difference between AI and ML?
- Explain Deep Learning
- What is RAG?

### DevOps Questions
- What is Terraform?
- What is a Harness Delegate?
- What are AWS storage services?
- How does GitHub Actions work?
- What are Linux file operations?
- What is Backstage software catalog?

### Advanced Questions
- Compare Terraform and CloudFormation
- What is the DevOps workflow with GitHub?
- How does Harness deploy to AWS?
- What are the security best practices?

---

## 🔧 Commands Reference

### Make Commands

```bash
make start        # Start all services + download models
make stop         # Stop all services
make restart      # Restart services
make clean        # Remove everything (models & data)
make logs         # View live logs
make status       # Check container & model status
make pull-models  # Download models manually
make setup        # Initial setup
```

### Docker Commands

```bash
# Container management
docker compose ps                    # List containers
docker compose logs -f              # View logs
docker compose restart rag-app      # Restart app
docker compose down                 # Stop containers
```

---

## 🛠️ Troubleshooting

- **Container failed to start?** Check ports `8501`, `11434`, and `8000` are free on host.
- **Slow generation times?** Ensure `CHUNK_SIZE` isn't too large or switch to `llama3.2:1b`.
- **Out of memory?** Increase your Docker memory limits to at least 8 GB.

---

## 🛣️ Roadmap

- [ ] Multi-user session management
- [ ] Support for audio/video transcript indexing
- [ ] Hybrid BM25 + Vector retrieval reranking
- [ ] Native GPU acceleration toggle

---

