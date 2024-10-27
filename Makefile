.PHONY: run-api run-data-service run-training-service start-services stop-services migrate train

run-api:
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

run-data-service:
	python -m app.services.data_service

run-training-service:
	python -m app.services.training_service

start-services:
	docker-compose -f infra/docker-compose.yml up --build -d

stop-services:
	docker-compose -f infra/docker-compose.yml down

migrate:
	alembic upgrade head

train:
	curl -X POST http://localhost:8000/train
