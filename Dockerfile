# Multi-stage build for optimized image size
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up build environment
WORKDIR /app
COPY requirements.txt .

# Build wheels for all requirements
RUN pip wheel --wheel-dir=/wheels -r requirements.txt

# Final stage with minimal runtime dependencies
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libleptonica-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages from pre-built wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy application files
COPY ["MATERIAL WITH STANDARD.xlsx", "./"]
COPY . .

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/

# Expose the port the app runs on
EXPOSE 10000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
