.PHONY: run-server run-training run-data-fetching start-services

run-server:
	python app/api/main.py

run-training:
	python services/training_service.py

run-data-fetching:
	python services/data_service.py

start-services:
	docker-compose -f app/docker-compose.yml up --build
