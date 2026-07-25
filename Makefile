.PHONY: setup start stop clean logs status pull-models

# Setup environment
setup:
	cp .env.example .env
	mkdir -p documents/uploaded vectorstore
	@echo "✅ Setup complete!"

# Pull required models
pull-models:
	@echo "📥 Pulling Ollama models..."
	docker exec rag-ollama ollama pull mistral
	docker exec rag-ollama ollama pull nomic-embed-text
	@echo "✅ Models pulled!"

# Start all services
start:
	@echo "🚀 Starting RAG Lab..."
	docker compose up -d --build
	@echo "⏳ Waiting for services to be ready..."
	sleep 15
	@make pull-models
	@echo ""
	@echo "✅ RAG Lab is running!"
	@echo "🌐 Open: http://localhost:8501"

# Stop all services
stop:
	@echo "🛑 Stopping RAG Lab..."
	docker compose down
	@echo "✅ Stopped!"

# Clean everything
clean:
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
	docker compose ps

# Restart services
restart:
	@make stop
	@make start
	