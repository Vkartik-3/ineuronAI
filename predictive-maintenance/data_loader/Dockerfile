# Dockerfile for C-MAPSS Data Loader
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run Kafka streamer
CMD ["python", "kafka_streamer.py", "--dataset", "FD001", "--train", "--rate", "1.0", "--loop"]
