# Stock Trading Application

## Overview

This application provides stock trading suggestions using AI models. It follows clean architecture principles to ensure maintainability and scalability.

## Clean Architecture

The application is organized into the following layers:

1. **Entities**: Core business objects.
2. **Use Cases**: Application-specific business rules.
3. **Interface Adapters**: Convert data from the format most convenient for the use cases and entities to the format most convenient for some external agency such as a database or the web.
4. **Frameworks and Drivers**: Details like UI, database, and external APIs.

## Project Structure

- **/app**: Contains the application code.
  - **/api**: API layer for handling HTTP requests.
  - **/core**: Core business logic and domain models.
    - **/domain**: Contains entity definitions.
    - **/use_cases**: Contains application-specific business rules.
    - **/interface_adapters**: Contains repository interfaces and implementations.
  - **/utils**: Utility functions for data loading and AI model operations.

## Setup
=======
## Documentation Links

Here are the documentation links for the frameworks and libraries used in this project:

- **Keras**: [Keras Documentation](https://keras.io/)
- **FastAPI**: [FastAPI Documentation](https://fastapi.tiangolo.com/)
- **Uvicorn**: [Uvicorn Documentation](https://www.uvicorn.org/)
- **Pandas**: [Pandas Documentation](https://pandas.pydata.org/docs/)
- **Openpyxl**: [Openpyxl Documentation](https://openpyxl.readthedocs.io/en/stable/)
- **NumPy**: [NumPy Documentation](https://numpy.org/doc/)
- **Kafka-Python**: [Kafka-Python Documentation](https://kafka-python.readthedocs.io/en/master/)
- **SQLAlchemy**: [SQLAlchemy Documentation](https://docs.sqlalchemy.org/en/20/)
- **Prophet**: [Prophet Documentation](https://facebook.github.io/prophet/)
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

## API Documentation

The API is documented using OpenAPI standards. You can access the API documentation by visiting `http://localhost:8000/docs` after starting the server.

## UI Dashboard

A UI dashboard is available to view the forecast vs actual tracking. Access it by visiting `http://localhost:8080` after starting the services.

## Usage

### Starting Services

To start all services, use the provided shell script:

```bash
./start_services.sh
```

Access the API at `http://localhost:8000/trade-suggestions` to get trade suggestions.

## Testing

Tests are located in the `/tests` directory. Run tests using:
