# Stock Trading Application

## Overview

This application provides stock trading suggestions using AI models. It follows clean architecture principles to ensure maintainability and scalability.

## Project Structure

- **/app**: Contains the application code.
  - **/api**: API layer for handling HTTP requests.
  - **/core**: Core business logic and domain models.
  - **/utils**: Utility functions for data loading and AI model operations.

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
