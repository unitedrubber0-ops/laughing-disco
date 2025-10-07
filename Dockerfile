# Use Python 3.11 slim image as the base
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies including Poppler and Tesseract
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libleptonica-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and material database
COPY requirements.txt .
COPY ["MATERIAL WITH STANDARD.xlsx", "./"]

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose the port the app runs on
EXPOSE 10000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
