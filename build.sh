#!/usr/bin/env bash
# exit on error
set -o errexit

# Create build directory
mkdir -p build

# Copy static assets
cp -r static build/
cp -r templates/* build/

# Create a simple index for static files (optional)
cat > build/static_index.html << EOL
<!DOCTYPE html>
<html>
<head>
    <title>Feasibility Analyzer Static Assets</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <h1>Static Assets Directory</h1>
    <script src="/static/script.js"></script>
</body>
</html>
EOL

# Install system dependencies (if needed)
if [ -n "$INSTALL_DEPENDENCIES" ]; then
    apt-get update
    apt-get install -y --no-install-recommends \
        poppler-utils
fi

# Install Python dependencies (if needed)
if [ -n "$INSTALL_PYTHON_DEPS" ]; then
    pip install -r requirements.txt
fi