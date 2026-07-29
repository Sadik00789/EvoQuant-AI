FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install alpaca-py without checking dependency bounds to bypass the websockets < 12 constraint
RUN pip install --no-cache-dir alpaca-py==0.17.0 --no-deps
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
