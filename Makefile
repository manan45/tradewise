.PHONY: run-api run-data-service run-training-service start-services stop-services migrate train test lint clean build

# Development server commands
run-api:
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

run-data-service:
	python -m app.services.data_service

run-training-service:
	python -m app.services.training_service

# Docker commands
start-services:
	docker-compose -f infra/docker-compose.yml up --build -d

stop-services:
	docker-compose -f infra/docker-compose.yml down

build:
	docker-compose -f infra/docker-compose.yml build

# Database commands
migrate:
	alembic upgrade head

migrate-rollback:
	alembic downgrade -1

# Training commands
train:
	curl -X POST http://localhost:8000/train

train-local:
	python -m app.core.ai.train

# Testing and quality commands
test:
	pytest tests/ -v --cov=app

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	flake8 app/
	black app/ --check
	isort app/ --check-only

format:
	black app/
	isort app/

# Utility commands
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/

# Development setup
setup-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

# Help command
help:
	@echo "Available commands:"
	@echo "  Development:"
	@echo "    run-api              - Run API development server"
	@echo "    run-data-service     - Run data service"
	@echo "    run-training-service - Run training service"
	@echo ""
	@echo "  Docker:"
	@echo "    start-services       - Start all services in Docker"
	@echo "    stop-services        - Stop all Docker services"
	@echo "    build               - Build Docker images"
	@echo ""
	@echo "  Database:"
	@echo "    migrate             - Run database migrations"
	@echo "    migrate-rollback    - Rollback last migration"
	@echo ""
	@echo "  Training:"
	@echo "    train               - Trigger model training via API"
	@echo "    train-local         - Run model training locally"
	@echo ""
	@echo "  Testing:"
	@echo "    test                - Run all tests with coverage"
	@echo "    test-unit           - Run unit tests"
	@echo "    test-integration    - Run integration tests"
	@echo ""
	@echo "  Code Quality:"
	@echo "    lint                - Check code style"
	@echo "    format              - Format code"
	@echo ""
	@echo "  Utility:"
	@echo "    clean               - Remove temporary files"
	@echo "    setup-dev           - Setup development environment"
