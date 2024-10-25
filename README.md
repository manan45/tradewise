# Stock Trading Application

## Overview

This application provides stock trading suggestions using AI models. It follows clean architecture principles to ensure maintainability and scalability.

## System Architecture

![Trade Management Architecture](docs/architecture.svg)

## Architecture

Our application follows a modular architecture designed for scalability and maintainability:

1. Data Pipeline:
   - External Data API: Source of stock data
   - Data Fetcher: Retrieves data from the API
   - Kafka Queue: Manages data flow
   - Data Consumer: Processes queued data

2. Machine Learning:
   - Trainer: Trains AI models on historical data
   - Tradewise Model: Generates stock price forecasts

3. Application Layer:
   - Backend: Serves data and predictions to the frontend
   - User Interface (UI): Presents data and forecasts to users

4. Database:
   - PostgreSQL: Stores historical and forecasted data

## Project Structure

- `/app`: Contains the application code.
  - `/api`: API layer for handling HTTP requests.
  - `/core`: Core business logic and domain models.
  - `/services`: Services for data fetching, processing, and AI training.
  - `/pipelines`: Data pipelines for fetching and ingesting stock data.
  - `/connectors`: Database and message queue connectors.
- `/dashboard`: Flask application for the UI dashboard.
- `/infra`: Infrastructure-related files (Docker, etc.).
- `/migrations`: Database migration scripts.

## Documentation Links

Here are the documentation links for the frameworks and libraries used in this project:

- **Keras**: [Keras Documentation](https://keras.io/)
- **FastAPI**: [FastAPI Documentation](https://fastapi.tiangolo.com/)
- **Uvicorn**: [Uvicorn Documentation](https://www.uvicorn.org/)
- **Pandas**: [Pandas Documentation](https://pandas.pydata.org/docs/)
- **NumPy**: [NumPy Documentation](https://numpy.org/doc/)
- **Kafka-Python**: [Kafka-Python Documentation](https://kafka-python.readthedocs.io/en/master/)
- **Flask**: [Flask Documentation](https://flask.palletsprojects.com/)
- **SQLAlchemy**: [SQLAlchemy Documentation](https://docs.sqlalchemy.org/en/20/)
- **Prophet**: [Prophet Documentation](https://facebook.github.io/prophet/)
- **Pydantic**: [Pydantic Documentation](https://docs.pydantic.dev/)
- **Docker**: [Docker Documentation](https://docs.docker.com/)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up the database:
   ```bash
   make migrate
   ```

3. Start the services:
   ```bash
   make start-services
   ```

## Usage

- API: Access at `http://localhost:8000`
- Dashboard: Access at `http://localhost:8080`
- API Documentation: `http://localhost:8000/docs`

## Development

- Run API server: `make run-api`
- Run Dashboard: `make run-dashboard`
- Run Data Service: `make run-data-service`
- Run Training Service: `make run-training-service`

## Testing

Tests are located in the `/tests` directory. Run tests using the appropriate testing framework.

## Technologies Used

- FastAPI: Backend API framework
- Flask: UI Dashboard
- Kafka: Queue system for data pipeline
- PostgreSQL: Database for storing stock data and predictions
- SQLAlchemy: ORM for database operations
- Pandas, NumPy: Data processing
- Scikit-learn, Keras, Prophet: AI and machine learning libraries
- Docker: Containerization

## Contributing

Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
