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

# Copy all application files
COPY . .

# Make debug script executable
RUN chmod +x debug.sh

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create startup script
RUN echo '#!/bin/bash\n\
echo "=== Environment Debug Info ==="\n\
pwd\n\
echo "Directory contents:"\n\
ls -la\n\
echo "Python path:"\n\
python -c "import sys; print(sys.path)"\n\
echo "Starting application..."\n\
exec gunicorn --bind 0.0.0.0:10000 wsgi:app --log-level debug' > /app/start.sh && \
    chmod +x /app/start.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=app.py
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 10000

# Run the application with startup script
CMD ["/app/start.sh"]
