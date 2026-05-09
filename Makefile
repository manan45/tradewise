.PHONY: api worker dashboard up down build migrate revision test lint format typecheck clean

# --- Local processes ---
api:
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

worker:
	python -m app.main

dashboard:
	streamlit run app/dashboard/Home.py

# --- Docker stack ---
up:
	docker compose -f infra/docker-compose.yml up -d

down:
	docker compose -f infra/docker-compose.yml down

build:
	docker compose -f infra/docker-compose.yml build

logs:
	docker compose -f infra/docker-compose.yml logs -f

# --- Database ---
migrate:
	alembic upgrade head

migrate-rollback:
	alembic downgrade -1

revision:
	alembic revision -m "$(MSG)"

# --- Quality ---
test:
	pytest -v

lint:
	ruff check app/ migrations/

format:
	ruff format app/ migrations/

typecheck:
	pyright

# --- Utility ---
clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
