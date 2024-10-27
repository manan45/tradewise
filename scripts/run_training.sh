#!/bin/bash

# Activate virtual environment if needed
# source venv/bin/activate

# Set environment variables
export MODEL_PATH="./models"
export SESSION_SAVE_DIR="./sessions"
export LOG_DIR="./logs"

# Run the training service
python -m app.services.training_service --symbol AAPL

