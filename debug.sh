#!/bin/bash

echo "=== Debugging Python Module Setup ==="
echo "Current directory: $(pwd)"
echo "Python path: $PYTHONPATH"
echo "Directory contents:"
ls -la

echo -e "\nPython files in directory:"
ls -la *.py

echo -e "\nChecking pdf_processor.py content:"
cat pdf_processor.py || echo "pdf_processor.py not found!"

echo -e "\nPython version and path:"
python --version
which python
python -c "import sys; print('Python path:', sys.path)"

echo -e "\nTrying to import pdf_processor:"
python -c "import pdf_processor; print('pdf_processor module found!')" || echo "Failed to import pdf_processor"

echo "=== Debug Info End ==="