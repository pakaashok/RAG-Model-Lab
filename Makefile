.PHONY: setup start stop clean logs status pull-models restart

# Default target when just typing "make"
.DEFAULT_GOAL := help

# Show help
help:
	@echo "🤖 Local RAG Lab - Commands:"
	@echo ""
	@echo "  make setup      - Initial setup"
	@echo "  make start      - Start everything"
	@echo "  make stop       - Stop everything"
	@echo "  make restart    - Restart"
	@echo "  make status     - Check status"
	@echo "  make logs       - View logs"
	@echo "  make clean      - Remove everything"
	@echo "  make pull-models- Download models"
	@echo ""

# Setup environment
setup:
	@echo "🔧 Setting up..."
	cp .env.example .env
	mkdir -p documents/uploaded vectorstore
	@echo "✅ Setup complete!"

# Pull required models
pull-models:
	@echo "📥 Pulling Ollama models..."
	@echo "   → nomic-embed-text (274MB)..."
	docker exec rag-ollama ollama pull nomic-embed-text
	@echo "   → mistral (4.1GB)..."
	docker exec rag-ollama ollama pull mistral
	@echo "✅ Models pulled!"

# Start all services
start:
	@echo "🚀 Starting RAG Lab..."
	docker compose up -d --build
	@echo "⏳ Waiting for services (30s)..."
	sleep 30
	@make pull-models
	@echo ""
	@echo "════════════════════════════════"
	@echo "✅ RAG Lab is running!"
	@echo "🌐 http://localhost:8501"
	@echo "════════════════════════════════"

# Stop all services
stop:
	@echo "🛑 Stopping RAG Lab..."
	docker compose down
	@echo "✅ Stopped!"

# Clean everything (WARNING: deletes models & data)
clean:
	@echo "⚠️  WARNING: This deletes models & data!"
	@echo "🧹 Cleaning up..."
	docker compose down -v
	rm -rf vectorstore/*
	rm -rf documents/uploaded/*
	@echo "✅ Cleaned!"

# View logs
logs:
	docker compose logs -f

# Check status
status:
	@echo "📊 Container Status:"
	docker compose ps
	@echo ""
	@echo "📊 Model Status:"
	docker exec rag-ollama ollama list

# Restart services
restart:
	@make stop
	@make start
	