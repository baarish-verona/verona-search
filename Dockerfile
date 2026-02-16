FROM python:3.11-slim

# Build argument for environment
ARG APP_ENV=production

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set HuggingFace cache directory
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model download script and run it
COPY scripts/download_models.py scripts/
RUN python scripts/download_models.py

# Copy application code and config
COPY . .

# Expose port
EXPOSE 3000

# Set environment from build arg
ENV APP_ENV=${APP_ENV}

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]
