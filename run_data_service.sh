#!/bin/bash

# Set environment variables
export KAFKA_BOOTSTRAP_SERVERS="kafka:9092"
export DB_HOST="db"
export DB_USER="postgres"
export DB_PASSWORD="password"
export DB_NAME="stockdb"
export DATA_SOURCE="apple"  # or "dhan"

# If using Dhan API, uncomment and set these variables
# export DHAN_CLIENT_ID="your_dhan_client_id"
# export DHAN_ACCESS_TOKEN="your_dhan_access_token"

# Run the data service
python -m app.services.data_service
