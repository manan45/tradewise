.PHONY: run-api run-dashboard run-data-service run-training-service start-services stop-services

run-api:
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	python -m flask run --host 0.0.0.0 --port 8080

run-data-service:
	python -m app.services.data_service

run-training-service:
	python -m app.services.training_service

start-services:
	docker-compose -f infra/docker-compose.yml up --build -d

stop-services:
	docker-compose -f infra/docker-compose.yml down

migrate:
	psql -h localhost -U postgres -d stockdb -f migrations/001_create_tables.sql
