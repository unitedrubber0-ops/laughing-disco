document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('drawing-file');
    const fileNameDisplay = document.getElementById('file-name');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultsContainer = document.getElementById('results-container');
    const errorMessage = document.getElementById('error-message');

    // Display the name of the selected file
    fileInput.addEventListener('change', () => {
        fileNameDisplay.textContent = fileInput.files.length > 0 ? fileInput.files[0].name : '';
    });

    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        // Validate file selection
        if (!fileInput.files || fileInput.files.length === 0) {
            displayError('Please select a PDF file to analyze.');
            return;
        }

        const file = fileInput.files[0];
        
        // Validate file type
        if (file.type !== 'application/pdf') {
            displayError('Please select a valid PDF file.');
            return;
        }

        // Validate file size (max 10MB)
        const maxSize = 10 * 1024 * 1024; // 10MB in bytes
        if (file.size > maxSize) {
            displayError('File size exceeds 10MB limit.');
            return;
        }

        // Show loading spinner and hide previous results
        loadingSpinner.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        errorMessage.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', file); // Changed from 'drawing' to 'file' to match backend

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData,
            });

            let errorMessage;
            if (!response.ok) {
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    const errorData = await response.json();
                    errorMessage = errorData.error || `Server error: ${response.status}`;
                } else {
                    errorMessage = `HTTP error! Status: ${response.status}`;
                }
                throw new Error(errorMessage);
            }

            const results = await response.json();
            displayResults(results);

        } catch (error) {
            console.error('Upload error:', error);
            displayError(error.message || 'An unexpected error occurred during file analysis.');
        } finally {
            // Hide loading spinner
            loadingSpinner.classList.add('hidden');
        }
    });

    function displayResults(data) {
        if (data.error) {
            displayError(data.error);
            return;
        }

        document.getElementById('result-part-number').textContent = data.part_number;
        document.getElementById('result-standard').textContent = data.standard;
        document.getElementById('result-grade').textContent = data.grade;
        document.getElementById('result-material').textContent = data.material;
        
        // Add download button for Excel file if available
        if (data.excel_data) {
            // Remove existing download button if it exists
            const existingButton = document.getElementById('download-excel');
            if (existingButton) {
                existingButton.remove();
            }
            
            const downloadButton = document.createElement('button');
            downloadButton.id = 'download-excel';
            downloadButton.textContent = 'Download Excel Sheet';
            downloadButton.style.marginTop = '20px';
            downloadButton.addEventListener('click', () => {
                // Convert base64 to blob and download
                const byteCharacters = atob(data.excel_data);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }
                const byteArray = new Uint8Array(byteNumbers);
                const blob = new Blob([byteArray], {type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'});
                
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'drawing_analysis.xlsx';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            });
            
            resultsContainer.appendChild(downloadButton);
        }
        
        resultsContainer.classList.remove('hidden');
    }

    function displayError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
    }
});