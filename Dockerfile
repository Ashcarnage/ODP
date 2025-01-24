# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

ARG CACHE_DATE=unknown

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model file
COPY ./app ./app
COPY ./app/app2.py .
COPY ./app/data ./data
COPY model.py .


# Create a non-root user (fixed version)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port for FastAPI
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.customAPI:app", "--host", "0.0.0.0", "--port", "8000"]