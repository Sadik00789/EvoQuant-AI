FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app

# Ensure logs flush instantly
ENV PYTHONUNBUFFERED=1

# Install essential C/C++ compilation tools for C-extension packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip build tooling, then install dependencies cleanly
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
