FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libleptonica-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy Python files first to check for their existence
COPY *.py ./
COPY requirements.txt ./
COPY ["MATERIAL WITH STANDARD.xlsx", "./"]

# List files to verify
RUN echo "Verifying Python files:" && \
    ls -la *.py && \
    echo "Python files verified."

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy remaining application files
COPY templates ./templates/
COPY static ./static/

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=app.py

# Expose port
EXPOSE 10000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
