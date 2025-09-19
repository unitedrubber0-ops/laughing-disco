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

# Load environment variables
ENV GUNICORN_TIMEOUT=${GUNICORN_TIMEOUT:-120}
ENV GUNICORN_WORKERS=${GUNICORN_WORKERS:-2}
ENV GUNICORN_THREADS=${GUNICORN_THREADS:-4}
ENV MAX_REQUESTS=${MAX_REQUESTS:-1000}
ENV MAX_REQUESTS_JITTER=${MAX_REQUESTS_JITTER:-50}

# Step F: Define the start command with enhanced configuration
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", \
     "--timeout", "${GUNICORN_TIMEOUT}", \
     "--workers", "${GUNICORN_WORKERS}", \
     "--threads", "${GUNICORN_THREADS}", \
     "--max-requests", "${MAX_REQUESTS}", \
     "--max-requests-jitter", "${MAX_REQUESTS_JITTER}", \
     "app:app"]