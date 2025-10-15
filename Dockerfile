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
cd /app\n\
echo "Current directory: $(pwd)"\n\
echo "Directory contents:"\n\
ls -la /app\n\
echo "Python files in /app:"\n\
find /app -name "*.py" -type f\n\
echo "Python path:"\n\
PYTHONPATH=/app python -c "import sys; print(\\"\\n\\".join(sys.path))"\n\
echo "Trying to import pdf_processor:"\n\
PYTHONPATH=/app python -c "import pdf_processor; print(f\\"Found pdf_processor at: {pdf_processor.__file__}\\")" || echo "Import failed"\n\
echo "Starting application..."\n\
export PYTHONPATH=/app:${PYTHONPATH}\n\
exec gunicorn --bind 0.0.0.0:10000 wsgi:app --log-level debug --preload' > /app/start.sh && \
    chmod +x /app/start.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=app.py
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONVERBOSE=1

# Verify Python environment
RUN echo "=== Verifying Python environment ==="; \
    cd /app && \
    python3 -c "import sys; print('Python path:'); [print(p) for p in sys.path]"; \
    python3 -c "import os; print('Contents of /app:'); [print(f) for f in os.listdir('.')]"; \
    python3 -c "import pdf_processor; print('pdf_processor loaded successfully')" || echo "Failed to import pdf_processor"

# Create debug verification script
RUN echo 'import os\n\
print("=== Python Module Verification ===")\n\
print("Current directory:", os.getcwd())\n\
print("Directory contents:", os.listdir("."))\n\
print("Attempting imports...")\n\
import pdf_processor\n\
print("pdf_processor imported from:", pdf_processor.__file__)\n\
from pdf_processor import process_pdf_comprehensive\n\
print("process_pdf_comprehensive imported successfully")\n\
' > /app/verify_imports.py

# Expose port
EXPOSE 10000

# Run the application with startup script
CMD ["bash", "-c", "cd /app && python3 verify_imports.py && exec /app/start.sh"]
