# Import wrapper for debugging
import sys
import os

def debug_import(module_name):
    print(f"Attempting to import {module_name}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Directory contents: {os.listdir('.')}")
    
    try:
        module = __import__(module_name)
        print(f"Successfully imported {module_name}")
        return module
    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")
        raise

if __name__ == "__main__":
    # Test importing pdf_processor
    debug_import("pdf_processor")