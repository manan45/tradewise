#!/bin/bash

# Simple script to run the application

# Function to display help
function show_help() {
    echo "Usage: ./start.sh [option]"
    echo "Options:"
    echo "  run      - Run the application"
    echo "  help     - Display this help message"
}

# Function to run the application
function run_application() {
    echo "Starting services..."
    
    # Start all services using docker-compose
    docker-compose -f docker-compose.yml up -d    
    echo "Services started successfully."
    echo "You can now access the following:"
    echo "1. API: http://localhost:8000"
    echo "2. Dashboard: http://localhost:8080"
    
    echo "To get trade suggestions, use the following curl command:"
    echo "curl -X POST http://localhost:8000/trade-suggestions -H 'Content-Type: application/json' -d '{\"symbol\": \"AAPL\", \"date\": \"2023-05-01\"}'"
}

# Main script logic
case "$1" in
    run)
        run_application
        ;;
    help|*)
        show_help
        ;;
esac
