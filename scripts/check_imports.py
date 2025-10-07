import sys, importlib
print('PYTHON:', sys.executable)
mods = ['pandas','numpy','openpyxl','fitz','PIL','pytesseract','pdf2image','cv2','flask','flask_cors','google.generativeai']
for m in mods:
    try:
        importlib.import_module(m)
        print(m + ' OK')
    except Exception as e:
        print(m + ' ERR', type(e).__name__, str(e))
