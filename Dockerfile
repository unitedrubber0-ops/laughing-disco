# Step A: Define the base image
FROM python:3.13-slim

# Step B: Install Poppler for pdf2image
RUN apt-get update && \
    apt-get install -y poppler-utils && \
    # Clean up to keep the image size small
    rm -rf /var/lib/apt/lists/*

# Step C: Set up the working directory inside the container
WORKDIR /app

# Step D: Copy your requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step E: Copy the rest of your application code into the container
COPY . .

# Set environment variables for Gunicorn configuration
ENV GUNICORN_TIMEOUT=300
ENV GUNICORN_WORKERS=2
ENV GUNICORN_THREADS=4
ENV MAX_REQUESTS=100
ENV MAX_REQUESTS_JITTER=20
ENV WORKER_CLASS=gthread
ENV WORKER_CONNECTIONS=1000

# Step F: Define the start command with enhanced configuration
CMD ["gunicorn", \
    "--bind", "0.0.0.0:${PORT}", \
    "--timeout", "300", \
    "--workers", "2", \
    "--threads", "4", \
    "--worker-class", "gthread", \
    "--worker-connections", "1000", \
    "--max-requests", "100", \
    "--max-requests-jitter", "20", \
    "--keepalive", "65", \
    "--log-level", "debug", \
    "app:app"]