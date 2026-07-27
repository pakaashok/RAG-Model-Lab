.PHONY: setup start stop clean logs status pull-models restart

# Setup environment
setup:
	cp .env.example .env
	mkdir -p documents/uploaded vectorstore
	@echo "Setup complete!"

# Pull required models - FAST MODELS
pull-models:
	@echo "Pulling embedding model (274MB)..."
	docker exec rag-ollama ollama pull nomic-embed-text
	@echo "Pulling LLM model (2GB)..."
	docker exec rag-ollama ollama pull llama3.2:3b
	@echo "Models pulled!"

# Start all services
start:
	@echo "Starting RAG Lab..."
	docker compose up -d --build
	@echo "Waiting for services (30s)..."
	sleep 30
	@make pull-models
	@echo ""
	@echo "======================================"
	@echo "RAG Lab is running!"
	@echo "Open: http://localhost:8501"
	@echo "======================================"

# Stop all services
stop:
	@echo "Stopping RAG Lab..."
	docker compose down
	@echo "Stopped!"

# Clean everything
clean:
	@echo "Cleaning up..."
	docker compose down -v
	rm -rf vectorstore/*
	rm -rf documents/uploaded/*
	@echo "Cleaned!"

# View logs
logs:
	docker compose logs -f

# Check status
status:
	@echo "Containers:"
	docker compose ps
	@echo ""
	@echo "Models:"
	docker exec rag-ollama ollama list

# Restart services
restart:
	@make stop
	@make start
	