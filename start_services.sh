#!/bin/bash

# Start only the database and Kafka services using Docker Compose
docker-compose -f app/docker-compose.yml up -d db kafka

# Run the FastAPI server locally
python app/api/main.py
