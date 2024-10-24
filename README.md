# Stock Trading Application

## Overview

This application provides stock trading suggestions using AI models. It follows clean architecture principles to ensure maintainability and scalability.

## Project Structure

- **/app**: Contains the application code.
  - **/api**: API layer for handling HTTP requests.
  - **/core**: Core business logic and domain models.
  - **/utils**: Utility functions for data loading and AI model operations.

## Documentation Links

Here are the documentation links for the frameworks and libraries used in this project:

- **Keras**: [Keras Documentation](https://keras.io/)
- **FastAPI**: [FastAPI Documentation](https://fastapi.tiangolo.com/)
- **Uvicorn**: [Uvicorn Documentation](https://www.uvicorn.org/)
- **Pandas**: [Pandas Documentation](https://pandas.pydata.org/docs/)
- **Openpyxl**: [Openpyxl Documentation](https://openpyxl.readthedocs.io/en/stable/)
- **NumPy**: [NumPy Documentation](https://numpy.org/doc/)
- **Kafka-Python**: [Kafka-Python Documentation](https://kafka-python.readthedocs.io/en/master/)
- **Flask**: [Flask Documentation](https://flask.palletsprojects.com/)
- **Ray**: [Ray Documentation](https://docs.ray.io/en/latest/)
- **SQLAlchemy**: [SQLAlchemy Documentation](https://docs.sqlalchemy.org/en/20/)
- **Prophet**: [Prophet Documentation](https://facebook.github.io/prophet/)
- **Gym**: [Gym Documentation](https://www.gymlibrary.dev/)
- **Pydantic**: [Pydantic Documentation](https://docs.pydantic.dev/)
- **PyMongo**: [PyMongo Documentation](https://pymongo.readthedocs.io/en/stable/)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   uvicorn app.api.main:app --reload
   ```

## Usage

Access the API at `http://localhost:8000/trade-suggestions` to get trade suggestions.

## Testing

Tests are located in the `/tests` directory. Run tests using:
# Stock Trading Application

## Architecture

This application follows the Clean Code Architecture pattern, which is divided into four layers:
- **Entities**: Contains the business logic related to the Stock entity.
- **Use Cases**: Contains the business rules for generating trade suggestions.
- **Interface Adapters**: Handles interaction with external systems like the Dhan API.
- **Frameworks & Drivers**: The entry point for the application, handling HTTP requests and responses.

## Setup

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the application using `make run-server`.

## Makefile and Docker Compose

- Use the Makefile to run different services.
- Use Docker Compose to set up MongoDB.
