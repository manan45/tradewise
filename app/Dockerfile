# Use an official Python runtime as a parent image
FROM python:3.9-slim as builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ .
COPY app/data/ ./data/
COPY app/utils/ ./utils/
COPY app/core/ ./core/
COPY app/services/ ./services/
COPY app/queue/ ./queue/

# Final stage
FROM python:3.9-slim

WORKDIR /app

COPY --from=builder /app /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
