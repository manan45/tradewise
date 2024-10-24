#!/bin/bash

# This script starts all the services required for the Stock Trading Application.
# It uses Docker Compose to build and run the services defined in the docker-compose.yml file.

# Usage:
#   ./start_services.sh

echo "Starting all services using Docker Compose..."
docker-compose -f app/docker-compose.yml up --build
