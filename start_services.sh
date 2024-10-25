#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  all         Start all services and show logs"
    echo "  api         Start API service and show logs"
    echo "  dashboard   Start Dashboard service and show logs"
    echo "  db          Start Database service and show logs"
    echo "  kafka       Start Kafka and Zookeeper services and show logs"
    echo "  data        Start Data service and show logs"
    echo "  training    Start Training service and show logs"
    echo "  -h, --help  Display this help message"
}

# Function to start a specific service and show logs
start_service() {
    service=$1
    echo "Starting $service service..."
    docker-compose -f infra/docker-compose.yml up -d $service
    docker-compose -f infra/docker-compose.yml logs -f $service
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

# Parse command-line arguments
while [ "$1" != "" ]; do
    case $1 in
        all )
            echo "Starting all services..."
            docker-compose -f infra/docker-compose.yml up -d
            docker-compose -f infra/docker-compose.yml logs -f
            ;;
        api )
            start_service api
            ;;
        dashboard )
            start_service dashboard
            ;;
        db )
            start_service db
            ;;
        kafka )
            docker-compose -f infra/docker-compose.yml up -d kafka zookeeper
            docker-compose -f infra/docker-compose.yml logs -f kafka zookeeper
            ;;
        data )
            start_service data_service
            ;;
        training )
            start_service training_service
            ;;
        -h | --help )
            usage
            read -p "Press Enter to continue..."
            return
            ;;
        * )
            echo "Invalid option: $1"
            usage``
            read -p "Press Enter to continue..."
            return
        esac
    shift
done

echo "Services started. Access the API at http://localhost:8000 and the dashboard at http://localhost:8080"
echo "Showing logs. Press Ctrl+C to exit."
