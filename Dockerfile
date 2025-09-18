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

# Step F: Define the start command
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]