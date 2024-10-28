# Stock Trading Application

## Overview

This application provides intelligent stock trading suggestions using advanced AI models. It follows clean architecture principles and implements sophisticated machine learning techniques for market analysis.

## System Architecture

### High-Level Architecture
![Trade Management Architecture](docs/architecture.svg)

### AI Model Architecture
![Tradewise AI Architecture](docs/tradewise_ai.svg)

## Architecture Components

Our application consists of two main architectural layers:

### 1. System Architecture
- **Data Pipeline**:
  - External Data API: Source of stock data
  - Data Fetcher: Retrieves data from API
  - Kafka Queue: Manages data flow
  - Data Consumer: Processes queued data

- **Application Layer**:
  - Backend API: Serves data and predictions
  - Database: PostgreSQL for data storage
  - Model Service: Handles AI predictions

### 2. AI Model Architecture

1. Model Trainer
   - Data Splitting: Separates data into training and evaluation sets
   - Training Set: Used for model training
   - Evaluation Set: Used for model validation
   - Prediction Generation: Creates trading predictions
   - Accuracy Evaluation: Measures model performance

2. Tradewise Model
   - Core Components:
     - Trading Psychology: Analyzes market sentiment and behavior patterns
     - Zone Analysis: Identifies key support/resistance zones
     - Technical Analysis: Processes technical indicators
   
   - Data Processing:
     - Time Series Data: Handles temporal market data
     - Market Data: Processes real-time market information
     - Trading Suggestions: Generates actionable trade recommendations
   
   - Model Logic:
     - Forecasting: Predicts future market movements
     - Prediction Accuracy: Evaluates forecast reliability
     - Performance Metrics: Tracks model effectiveness
   
   - Session Management:
     - Performance Tracking: Monitors trading session performance
     - Model Optimization: Tunes model parameters
     - Model Reinforcement: Improves model through learning

3. Data Storage
   - PostgreSQL Database: Stores historical data, predictions, and performance metrics
   - Enables persistent storage of:
     - Historical market data
     - Model predictions
     - Performance metrics
     - Training results

## Data Flow

1. External data flows through the data pipeline
2. Data is processed and stored in PostgreSQL
3. Model Trainer processes data for AI training
4. Tradewise Model:
   - Processes incoming market data
   - Generates predictions
   - Tracks performance
   - Continuously optimizes itself
5. Results are stored and served via API

## Project Structure

- `/app`: Application code
  - `/core`: Core business logic
    - `/ai`: AI model implementations
    - `/domain`: Domain models and interfaces
    - `/use_cases`: Business logic implementation
  - `/api`: API endpoints
  - `/infrastructure`: External service integrations

- `/docs`: Documentation and diagrams
- `/infra`: Infrastructure configuration
- `/migrations`: Database migrations

## Setup and Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   ```

3. Start services:
   ```bash
   make start-services
   ```

## Running Components

```bash
# Start API server
make run-api

# Run model training
make run-trainer

# Start prediction service
make run-predictions
```

## Technologies Used

- **AI/ML**: 
  - TensorFlow/Keras: Deep learning models
  - Scikit-learn: Machine learning algorithms
  - Pandas: Data processing
  - NumPy: Numerical computations

- **Infrastructure**:
  - PostgreSQL: Data storage
  - Docker: Containerization
  - FastAPI: API framework
  - Kafka: Message queue

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
